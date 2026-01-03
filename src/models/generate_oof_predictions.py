"""
OOF (Out-of-Fold) 예측 생성 스크립트

3개 모델(XGBoost, LightGBM, CatBoost)의 OOF 예측을 생성합니다.
Stacking 앙상블의 Meta-Learner 학습에 사용됩니다.

"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tqdm import tqdm

# 경로 설정
DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')
OUTPUT_DIR = Path('data/processed')

print("=" * 60)
print("OOF 예측 생성 시작")
print("=" * 60)

# 1. 데이터 로드
print("\n📂 데이터 로드 중...")
df = pd.read_csv(DATA_DIR / 'train_final_passes_v6.csv')
print(f"✅ Shape: {df.shape}")

# 2. 피처 및 타겟 분리
target_cols = ['end_x', 'end_y']
feature_cols = [col for col in df.columns if col not in target_cols + ['game_episode']]

print(f"\n🔍 범주형 피처 확인 중...")

# 범주형 피처 인코딩
categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    print(f"📝 범주형 피처 발견: {categorical_cols}")
    from sklearn.preprocessing import LabelEncoder
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"✅ 범주형 피처 인코딩 완료")
else:
    print(f"✅ 범주형 피처 없음")

X = df[feature_cols].values
y_x = df['end_x'].values
y_y = df['end_y'].values

print(f"✅ 피처: {len(feature_cols)}개")
print(f"✅ 샘플: {len(X):,}개")

# 3. K-Fold 설정
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print(f"\n🔀 {N_SPLITS}-Fold Cross-Validation")

# 4. OOF 예측 저장 배열 초기화
oof_predictions = {
    'xgb': {'x': np.zeros(len(X)), 'y': np.zeros(len(X))},
    'lgb': {'x': np.zeros(len(X)), 'y': np.zeros(len(X))},
    'cat': {'x': np.zeros(len(X)), 'y': np.zeros(len(X))}
}

# 5. 각 모델별 OOF 생성
print("\n" + "=" * 60)
print("1️⃣ XGBoost OOF 생성")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=N_SPLITS, desc="XGBoost")):
    # Train/Val 분리
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]
    
    # XGBoost 파라미터 (v4 튜닝된 버전)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'random_state': 42
    }
    
    # end_x 예측
    model_x = xgb.XGBRegressor(**params)
    model_x.fit(X_train, y_train_x, verbose=False)
    oof_predictions['xgb']['x'][val_idx] = model_x.predict(X_val)
    
    # end_y 예측
    model_y = xgb.XGBRegressor(**params)
    model_y.fit(X_train, y_train_y, verbose=False)
    oof_predictions['xgb']['y'][val_idx] = model_y.predict(X_val)

# XGBoost OOF RMSE 계산
xgb_rmse_x = np.sqrt(mean_squared_error(y_x, oof_predictions['xgb']['x']))
xgb_rmse_y = np.sqrt(mean_squared_error(y_y, oof_predictions['xgb']['y']))
xgb_rmse = np.sqrt((xgb_rmse_x**2 + xgb_rmse_y**2) / 2)
print(f"\n✅ XGBoost OOF RMSE: {xgb_rmse:.4f}m")
print(f"   - end_x RMSE: {xgb_rmse_x:.4f}m")
print(f"   - end_y RMSE: {xgb_rmse_y:.4f}m")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2️⃣ LightGBM OOF 생성")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=N_SPLITS, desc="LightGBM")):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]
    
    # LightGBM 파라미터 (v4 튜닝된 버전)
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
        'verbosity': -1,
        'random_state': 42
    }
    
    # end_x 예측
    train_data_x = lgb.Dataset(X_train, label=y_train_x)
    model_x = lgb.train(params, train_data_x, num_boost_round=800)
    oof_predictions['lgb']['x'][val_idx] = model_x.predict(X_val)
    
    # end_y 예측
    train_data_y = lgb.Dataset(X_train, label=y_train_y)
    model_y = lgb.train(params, train_data_y, num_boost_round=800)
    oof_predictions['lgb']['y'][val_idx] = model_y.predict(X_val)

# LightGBM OOF RMSE 계산
lgb_rmse_x = np.sqrt(mean_squared_error(y_x, oof_predictions['lgb']['x']))
lgb_rmse_y = np.sqrt(mean_squared_error(y_y, oof_predictions['lgb']['y']))
lgb_rmse = np.sqrt((lgb_rmse_x**2 + lgb_rmse_y**2) / 2)
print(f"\n✅ LightGBM OOF RMSE: {lgb_rmse:.4f}m")
print(f"   - end_x RMSE: {lgb_rmse_x:.4f}m")
print(f"   - end_y RMSE: {lgb_rmse_y:.4f}m")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("3️⃣ CatBoost OOF 생성")
print("=" * 60)

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=N_SPLITS, desc="CatBoost")):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]
    
    # CatBoost 파라미터 (v4)
    params = {
        'iterations': 500,
        'depth': 8,
        'learning_rate': 0.05,
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }
    
    # end_x 예측
    model_x = cb.CatBoostRegressor(**params)
    model_x.fit(X_train, y_train_x, verbose=False)
    oof_predictions['cat']['x'][val_idx] = model_x.predict(X_val)
    
    # end_y 예측
    model_y = cb.CatBoostRegressor(**params)
    model_y.fit(X_train, y_train_y, verbose=False)
    oof_predictions['cat']['y'][val_idx] = model_y.predict(X_val)

# CatBoost OOF RMSE 계산
cat_rmse_x = np.sqrt(mean_squared_error(y_x, oof_predictions['cat']['x']))
cat_rmse_y = np.sqrt(mean_squared_error(y_y, oof_predictions['cat']['y']))
cat_rmse = np.sqrt((cat_rmse_x**2 + cat_rmse_y**2) / 2)
print(f"\n✅ CatBoost OOF RMSE: {cat_rmse:.4f}m")
print(f"   - end_x RMSE: {cat_rmse_x:.4f}m")
print(f"   - end_y RMSE: {cat_rmse_y:.4f}m")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("📊 OOF 성능 요약")
print("=" * 60)
print(f"XGBoost : {xgb_rmse:.4f}m")
print(f"LightGBM: {lgb_rmse:.4f}m")
print(f"CatBoost: {cat_rmse:.4f}m")
print(f"평균    : {(xgb_rmse + lgb_rmse + cat_rmse) / 3:.4f}m")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("💾 OOF 예측 저장")
print("=" * 60)

# OOF DataFrame 생성
oof_df = pd.DataFrame({
    'game_episode': df['game_episode'],
    'true_x': y_x,
    'true_y': y_y,
    'xgb_pred_x': oof_predictions['xgb']['x'],
    'xgb_pred_y': oof_predictions['xgb']['y'],
    'lgb_pred_x': oof_predictions['lgb']['x'],
    'lgb_pred_y': oof_predictions['lgb']['y'],
    'cat_pred_x': oof_predictions['cat']['x'],
    'cat_pred_y': oof_predictions['cat']['y']
})

# 저장
output_path = OUTPUT_DIR / 'oof_predictions_v6.csv'
oof_df.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path}")
print(f"   Shape: {oof_df.shape}")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("🎉 OOF 생성 완료!")
print("=" * 60)
print("\n다음 단계:")
print("1. Meta-Learner 학습: python src/models/train_meta_learner.py")
print("2. Stacking 예측: python src/models/predict_stacking.py")
