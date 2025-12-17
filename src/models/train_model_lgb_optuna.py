"""
LightGBM Optuna 자동 하이퍼파라미터 튜닝
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle
import optuna

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'


def prepare_features(df: pd.DataFrame) -> tuple:
    """피처 준비 (기존 코드와 동일)"""
    df = df.copy()
    
    # 범주형 인코딩
    zone_x_map = {'defensive': 0, 'midfield': 1, 'attacking': 2}
    zone_y_map = {'left': 0, 'center': 1, 'right': 2}
    zone_combined_map = {
        'defensive_left': 0, 'defensive_center': 1, 'defensive_right': 2,
        'midfield_left': 3, 'midfield_center': 4, 'midfield_right': 5,
        'attacking_left': 6, 'attacking_center': 7, 'attacking_right': 8
    }
    
    if 'zone_x' in df.columns and df['zone_x'].dtype == 'object':
        df['zone_x_encoded'] = df['zone_x'].map(zone_x_map)
        df['zone_y_encoded'] = df['zone_y'].map(zone_y_map)
        df['zone_combined_encoded'] = df['zone_combined'].map(zone_combined_map)
    
    # Phase 1, 2, 3 피처
    phase1_features = [
        'start_x', 'start_y', 'dist_to_target_goal',
        'zone_x_encoded', 'zone_y_encoded', 'zone_combined_encoded',
        'in_penalty_box', 'in_final_third',
    ]
    
    phase2_features = [
        'prev_end_x', 'prev_end_y', 'prev_action_distance',
        'time_since_prev', 'prev_direction_x', 'prev_direction_y',
        'pass_count_in_episode'
    ]
    
    phase3_features = [
        'pass_velocity', 'avg_episode_velocity',
        'touchline_proximity', 'episode_x_range',
        'is_under_pressure', 'rolling_mean_distance_3',
    ]
    
    feature_cols = []
    for feat_list in [phase1_features, phase2_features, phase3_features]:
        for feat in feat_list:
            if feat in df.columns:
                feature_cols.append(feat)
    
    X = df[feature_cols].copy()
    y_x = df['end_x'].copy()
    y_y = df['end_y'].copy()
    
    if X.isna().sum().sum() > 0:
        X = X.fillna(0)
    
    return X, y_x, y_y, feature_cols


def objective(trial, X, y_x, y_y):
    """Optuna 목적 함수"""
    
    # 탐색할 하이퍼파라미터 범위
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        
        # 튜닝할 파라미터
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000, step=100),
        
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    }
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_x_train, y_x_val = y_x.iloc[train_idx], y_x.iloc[val_idx]
        y_y_train, y_y_val = y_y.iloc[train_idx], y_y.iloc[val_idx]
        
        # X 좌표 모델
        train_data_x = lgb.Dataset(X_train, label=y_x_train)
        val_data_x = lgb.Dataset(X_val, label=y_x_val, reference=train_data_x)
        
        model_x = lgb.train(
            params,
            train_data_x,
            valid_sets=[val_data_x],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # Y 좌표 모델
        train_data_y = lgb.Dataset(X_train, label=y_y_train)
        val_data_y = lgb.Dataset(X_val, label=y_y_val, reference=train_data_y)
        
        model_y = lgb.train(
            params,
            train_data_y,
            valid_sets=[val_data_y],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # 예측
        pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)
        
        # 유클리드 거리 RMSE
        euclidean = np.sqrt((y_x_val - pred_x)**2 + (y_y_val - pred_y)**2)
        rmse = np.sqrt(np.mean(euclidean**2))
        cv_scores.append(rmse)
    
    return np.mean(cv_scores)


def main():
    print("="*60)
    print("🔍 LightGBM Optuna 자동 튜닝")
    print("="*60)
    
    # 데이터 로드
    train_path = DATA_DIR / 'train_final_passes_v3.csv'
    df = pd.read_csv(train_path)
    print(f"✅ 데이터 로드: {len(df):,}개")
    
    # 피처 준비
    X, y_x, y_y, feature_cols = prepare_features(df)
    print(f"✅ 피처 수: {len(feature_cols)}")
    
    # Optuna 스터디
    print(f"\n🔄 하이퍼파라미터 탐색 시작...")
    print(f"  - 탐색 횟수: 50회")
    print(f"  - 예상 시간: 2-3시간")
    print(f"  - 각 trial마다 5-Fold CV 수행")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y_x, y_y),
        n_trials=50,  # 50회 탐색
        show_progress_bar=True
    )
    
    # 최적 파라미터
    print("\n" + "="*60)
    print("✅ 튜닝 완료!")
    print("="*60)
    
    print(f"\n🏆 최적 CV RMSE: {study.best_value:.4f}m")
    print(f"\n📊 최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    
    # 최적 파라미터로 최종 모델 학습
    print(f"\n🔧 최종 모델 학습 중...")
    
    best_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42,
        **study.best_params
    }
    
    train_data_x = lgb.Dataset(X, label=y_x)
    final_model_x = lgb.train(best_params, train_data_x)
    
    train_data_y = lgb.Dataset(X, label=y_y)
    final_model_y = lgb.train(best_params, train_data_y)
    
    # 저장
    model_path = MODEL_DIR / 'lgb_model_v3_optuna.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model_x': final_model_x,
            'model_y': final_model_y,
            'features': feature_cols,
            'cv_score': study.best_value,
            'best_params': study.best_params
        }, f)
    
    print(f"✅ 모델 저장: {model_path}")
    
    # 비교
    print("\n" + "="*60)
    print("📊 성능 비교")
    print("="*60)
    print(f"기존 (수동): 18.81m")
    print(f"Optuna:     {study.best_value:.2f}m")
    
    if study.best_value < 18.81:
        improvement = 18.81 - study.best_value
        print(f"✅ 개선: -{improvement:.2f}m")
    else:
        print(f"⚠️  악화: +{study.best_value - 18.81:.2f}m")


if __name__ == '__main__':
    main()