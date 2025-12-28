"""
LightGBM 모델 학습 - Phase 4 버전
Phase 4: 도메인 특화 피처 (선수/팀 통계, 경기 흐름)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'


def prepare_features(df: pd.DataFrame) -> tuple:
    """피처 준비 (v4 버전 - Phase 4 포함)"""
    df = df.copy()
    
    print(f"\n🔧 피처 준비 시작")
    print(f"  - 원본 컬럼 수: {len(df.columns)}")
    
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
    
    # Phase 1
    phase1_features = [
        'start_x', 'start_y',
        'dist_to_target_goal',
        'zone_x_encoded', 'zone_y_encoded', 'zone_combined_encoded',
        'in_penalty_box', 'in_final_third',
    ]
    
    # Phase 2
    phase2_features = [
        'prev_end_x', 'prev_end_y',
        'prev_action_distance',
        'time_since_prev',
        'prev_direction_x', 'prev_direction_y',
        'pass_count_in_episode'
    ]
    
    # Phase 3 (선별 6개)
    phase3_features = [
        'pass_velocity',
        'avg_episode_velocity',
        'touchline_proximity',
        'episode_x_range',
        'is_under_pressure',
        'rolling_mean_distance_3',
    ]
    
    # Phase 4 (도메인 특화 9개) ⭐ NEW!
    phase4_features = [
        # 선수 통계 (4개)
        'player_avg_pass_distance',
        'player_forward_ratio',
        'player_success_rate',
        'player_pass_count',
        # 팀 통계 (2개)
        'team_avg_pass_distance',
        'team_attack_style',
        # 경기 흐름 (3개)
        'score_diff',
        'match_period_normalized',
        'is_late_game',
    ]
    
    # 존재하는 피처만 수집
    feature_cols = []
    for feat_list in [phase1_features, phase2_features, phase3_features, phase4_features]:
        for feat in feat_list:
            if feat in df.columns:
                feature_cols.append(feat)
    
    print(f"  - Phase 1: {len([f for f in phase1_features if f in df.columns])}개")
    print(f"  - Phase 2: {len([f for f in phase2_features if f in df.columns])}개")
    print(f"  - Phase 3: {len([f for f in phase3_features if f in df.columns])}개")
    print(f"  - Phase 4: {len([f for f in phase4_features if f in df.columns])}개 ⭐")
    print(f"  - 최종 피처 수: {len(feature_cols)}개")
    
    # X, y 분리
    X = df[feature_cols].copy()
    y_x = df['end_x'].copy()
    y_y = df['end_y'].copy()
    
    # 결측치 처리
    if X.isna().sum().sum() > 0:
        print(f"  ⚠️  결측치 발견 → 0으로 대체")
        X = X.fillna(0)
    
    return X, y_x, y_y, feature_cols


def train_lgb_model(df: pd.DataFrame, n_folds: int = 5):
    """LightGBM 학습"""
    print("=" * 60)
    print("🚀 LightGBM 모델 학습 시작 (Phase 4)")
    print("=" * 60)
    
    # 피처 준비
    X, y_x, y_y, feature_cols = prepare_features(df)
    
    print(f"\n📊 데이터 정보:")
    print(f"  - 샘플 수: {len(X):,}")
    print(f"  - 피처 수: {len(feature_cols)}")
    
    # LightGBM 파라미터
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"\n⚙️  하이퍼파라미터:")
    for key, value in params.items():
        print(f"  - {key}: {value}")
    
    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_scores_x = []
    cv_scores_y = []
    cv_scores_total = []
    
    print(f"\n🔄 {n_folds}-Fold Cross Validation:")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_x_train, y_x_val = y_x.iloc[train_idx], y_x.iloc[val_idx]
        y_y_train, y_y_val = y_y.iloc[train_idx], y_y.iloc[val_idx]
        
        # end_x 모델
        train_data_x = lgb.Dataset(X_train, label=y_x_train)
        val_data_x = lgb.Dataset(X_val, label=y_x_val, reference=train_data_x)
        
        model_x = lgb.train(
            params,
            train_data_x,
            num_boost_round=800,
            valid_sets=[val_data_x],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # end_y 모델
        train_data_y = lgb.Dataset(X_train, label=y_y_train)
        val_data_y = lgb.Dataset(X_val, label=y_y_val, reference=train_data_y)
        
        model_y = lgb.train(
            params,
            train_data_y,
            num_boost_round=500,
            valid_sets=[val_data_y],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
        )
        
        # 예측
        pred_x = model_x.predict(X_val, num_iteration=model_x.best_iteration)
        pred_y = model_y.predict(X_val, num_iteration=model_y.best_iteration)
        
        # RMSE
        rmse_x = np.sqrt(mean_squared_error(y_x_val, pred_x))
        rmse_y = np.sqrt(mean_squared_error(y_y_val, pred_y))
        
        euclidean_errors = np.sqrt((y_x_val - pred_x)**2 + (y_y_val - pred_y)**2)
        rmse_total = np.sqrt(np.mean(euclidean_errors**2))
        
        cv_scores_x.append(rmse_x)
        cv_scores_y.append(rmse_y)
        cv_scores_total.append(rmse_total)
        
        print(f"  Fold {fold}: RMSE_X={rmse_x:.4f}m, RMSE_Y={rmse_y:.4f}m, Total={rmse_total:.4f}m")
    
    print(f"\n📈 Cross-validation 결과:")
    print(f"  - RMSE_X: {np.mean(cv_scores_x):.4f} ± {np.std(cv_scores_x):.4f}m")
    print(f"  - RMSE_Y: {np.mean(cv_scores_y):.4f} ± {np.std(cv_scores_y):.4f}m")
    print(f"  - RMSE_Total: {np.mean(cv_scores_total):.4f} ± {np.std(cv_scores_total):.4f}m")
    
    # 최종 모델
    print(f"\n🔧 전체 데이터로 최종 모델 학습 중...")
    
    train_data_x = lgb.Dataset(X, label=y_x)
    final_model_x = lgb.train(params, train_data_x, num_boost_round=500)
    
    train_data_y = lgb.Dataset(X, label=y_y)
    final_model_y = lgb.train(params, train_data_y, num_boost_round=500)
    
    print(f"✅ 학습 완료!")
    
    # 피처 중요도
    importance_x = pd.DataFrame({
        'feature': feature_cols,
        'importance_x': final_model_x.feature_importance(importance_type='gain'),
        'importance_y': final_model_y.feature_importance(importance_type='gain')
    }).sort_values('importance_x', ascending=False)
    
    print(f"\n📊 피처 중요도 (Top 15):")
    print(importance_x.head(15).to_string(index=False))
    
    # Phase 4 피처만 따로 확인
    phase4_features = [
        'player_avg_pass_distance', 'player_forward_ratio', 
        'player_success_rate', 'player_pass_count',
        'team_avg_pass_distance', 'team_attack_style',
        'score_diff', 'match_period_normalized', 'is_late_game'
    ]
    phase4_importance = importance_x[
        importance_x['feature'].isin(phase4_features)
    ]
    
    if len(phase4_importance) > 0:
        print(f"\n📊 Phase 4 피처 중요도:")
        print(phase4_importance.to_string(index=False))
    
    # 모델 저장
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / 'lgb_model_v4.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model_x': final_model_x,
            'model_y': final_model_y,
            'features': feature_cols,
            'cv_score': np.mean(cv_scores_total)
        }, f)
    
    print(f"\n💾 모델 저장 완료: {model_path}")
    
    # 이전 버전과 비교
    print("\n" + "="*60)
    print("📊 성능 비교")
    print("="*60)
    print(f"Phase 2 (v2): CV 18.81m")
    print(f"Phase 3 (v3): CV 18.82m")
    print(f"Phase 4 (v4): CV {np.mean(cv_scores_total):.2f}m")
    
    if np.mean(cv_scores_total) < 18.82:
        improvement = 18.82 - np.mean(cv_scores_total)
        print(f"✅ v3 대비 개선: -{improvement:.2f}m ({improvement/18.82*100:.1f}%)")
    else:
        print(f"⚠️  v3 대비 악화: +{np.mean(cv_scores_total) - 18.82:.2f}m")
    
    return {
        'cv_scores_total': cv_scores_total,
        'mean_cv': np.mean(cv_scores_total)
    }


if __name__ == '__main__':
    print("🎯 K리그 패스 좌표 예측 - LightGBM v4 (Phase 4)\n")
    
    # 데이터 로드 (v4로 변경!)
    train_path = DATA_DIR / 'train_final_passes_v4.csv'
    
    if not train_path.exists():
        print(f"❌ v4 데이터 파일이 없습니다: {train_path}")
        exit(1)
    
    df = pd.read_csv(train_path)
    print(f"✅ 데이터 로드 완료: {len(df):,}개 샘플")
    print(f"  - 컬럼 수: {len(df.columns)}")
    
    # Phase 4 피처 확인
    phase4_features = [
        'player_avg_pass_distance', 'team_avg_pass_distance', 
        'score_diff', 'match_period_normalized'
    ]
    has_phase4 = all(feat in df.columns for feat in phase4_features)
    
    print(f"\nPhase 4 피처 존재: {'✓' if has_phase4 else '✗'}")
    
    if not has_phase4:
        print("⚠️  Phase 4 피처가 없습니다!")
        missing = [f for f in phase4_features if f not in df.columns]
        print(f"누락된 피처: {missing}")
        exit(1)
    
    # 학습
    results = train_lgb_model(df, n_folds=5)
    
    print("\n" + "="*60)
    print("🎊 LightGBM Phase 4 학습 완료!")
    print("="*60)
