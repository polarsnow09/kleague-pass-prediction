"""
Phase 6 모델 학습: 3개 모델 통합 학습 스크립트

에러 분석 기반 타겟팅 피처 포함
- XGBoost
- LightGBM  
- CatBoost
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
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Phase 6: 3개 모델 통합 학습")
print("=" * 60)

# =================================================================
# 1. 데이터 로드
# =================================================================
print("\n📂 데이터 로드 중...")
df = pd.read_csv(DATA_DIR / 'train_final_passes_v6.csv')
print(f"✅ Shape: {df.shape}")

# =================================================================
# 2. 피처 및 타겟 분리
# =================================================================
target_cols = ['end_x', 'end_y']
feature_cols = [col for col in df.columns if col not in target_cols + ['game_episode']]

print(f"\n🔍 피처 확인...")
print(f"   총 피처: {len(feature_cols)}개")

# 범주형 피처 인코딩
categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
if len(categorical_cols) > 0:
    print(f"   범주형 피처: {len(categorical_cols)}개")
    from sklearn.preprocessing import LabelEncoder
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # LabelEncoder 저장
    with open(MODEL_DIR / 'label_encoders_v6.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"   ✅ 범주형 인코딩 완료")

X = df[feature_cols].values
y_x = df['end_x'].values
y_y = df['end_y'].values

print(f"✅ X shape: {X.shape}")
print(f"✅ y_x shape: {y_x.shape}")
print(f"✅ y_y shape: {y_y.shape}")

# =================================================================
# 3. Cross-Validation 설정
# =================================================================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print(f"\n🔀 {N_SPLITS}-Fold Cross-Validation")

# =================================================================
# 4. XGBoost 학습
# =================================================================
print("\n" + "=" * 60)
print("1️⃣ XGBoost 학습")
print("=" * 60)

xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'random_state': 42
}

cv_scores_xgb = []

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=N_SPLITS, desc="XGBoost CV")):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]
    
    # end_x 모델
    model_x = xgb.XGBRegressor(**xgb_params)
    model_x.fit(X_train, y_train_x, verbose=False)
    pred_x = model_x.predict(X_val)
    
    # end_y 모델
    model_y = xgb.XGBRegressor(**xgb_params)
    model_y.fit(X_train, y_train_y, verbose=False)
    pred_y = model_y.predict(X_val)
    
    # RMSE 계산
    rmse_x = np.sqrt(mean_squared_error(y_val_x, pred_x))
    rmse_y = np.sqrt(mean_squared_error(y_val_y, pred_y))
    rmse = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
    
    cv_scores_xgb.append(rmse)

xgb_cv_mean = np.mean(cv_scores_xgb)
xgb_cv_std = np.std(cv_scores_xgb)

print(f"\n✅ XGBoost CV 결과:")
print(f"   평균 RMSE: {xgb_cv_mean:.4f}m (±{xgb_cv_std:.4f})")
print(f"   Fold별: {[f'{s:.4f}' for s in cv_scores_xgb]}")

# 전체 데이터로 최종 학습
print(f"\n🔄 전체 데이터로 최종 학습 중...")
xgb_model_x = xgb.XGBRegressor(**xgb_params)
xgb_model_x.fit(X, y_x, verbose=False)

xgb_model_y = xgb.XGBRegressor(**xgb_params)
xgb_model_y.fit(X, y_y, verbose=False)

# 저장
with open(MODEL_DIR / 'baseline_model_v6.pkl', 'wb') as f:
    pickle.dump({'model_x': xgb_model_x, 'model_y': xgb_model_y}, f)

print(f"✅ 저장: baseline_model_v6.pkl")

# =================================================================
# 5. LightGBM 학습
# =================================================================
print("\n" + "=" * 60)
print("2️⃣ LightGBM 학습")
print("=" * 60)

lgb_params = {
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

cv_scores_lgb = []

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=N_SPLITS, desc="LightGBM CV")):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]
    
    # end_x 모델
    train_data_x = lgb.Dataset(X_train, label=y_train_x)
    model_x = lgb.train(lgb_params, train_data_x, num_boost_round=800)
    pred_x = model_x.predict(X_val)
    
    # end_y 모델
    train_data_y = lgb.Dataset(X_train, label=y_train_y)
    model_y = lgb.train(lgb_params, train_data_y, num_boost_round=800)
    pred_y = model_y.predict(X_val)
    
    # RMSE 계산
    rmse_x = np.sqrt(mean_squared_error(y_val_x, pred_x))
    rmse_y = np.sqrt(mean_squared_error(y_val_y, pred_y))
    rmse = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
    
    cv_scores_lgb.append(rmse)

lgb_cv_mean = np.mean(cv_scores_lgb)
lgb_cv_std = np.std(cv_scores_lgb)

print(f"\n✅ LightGBM CV 결과:")
print(f"   평균 RMSE: {lgb_cv_mean:.4f}m (±{lgb_cv_std:.4f})")
print(f"   Fold별: {[f'{s:.4f}' for s in cv_scores_lgb]}")

# 전체 데이터로 최종 학습
print(f"\n🔄 전체 데이터로 최종 학습 중...")
train_data_x = lgb.Dataset(X, label=y_x)
lgb_model_x = lgb.train(lgb_params, train_data_x, num_boost_round=800)

train_data_y = lgb.Dataset(X, label=y_y)
lgb_model_y = lgb.train(lgb_params, train_data_y, num_boost_round=800)

# 저장
with open(MODEL_DIR / 'lgb_model_v6.pkl', 'wb') as f:
    pickle.dump({'model_x': lgb_model_x, 'model_y': lgb_model_y}, f)

print(f"✅ 저장: lgb_model_v6.pkl")

# =================================================================
# 6. CatBoost 학습
# =================================================================
print("\n" + "=" * 60)
print("3️⃣ CatBoost 학습")
print("=" * 60)

cat_params = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

cv_scores_cat = []

for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=N_SPLITS, desc="CatBoost CV")):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_x, y_val_x = y_x[train_idx], y_x[val_idx]
    y_train_y, y_val_y = y_y[train_idx], y_y[val_idx]
    
    # end_x 모델
    model_x = cb.CatBoostRegressor(**cat_params)
    model_x.fit(X_train, y_train_x, verbose=False)
    pred_x = model_x.predict(X_val)
    
    # end_y 모델
    model_y = cb.CatBoostRegressor(**cat_params)
    model_y.fit(X_train, y_train_y, verbose=False)
    pred_y = model_y.predict(X_val)
    
    # RMSE 계산
    rmse_x = np.sqrt(mean_squared_error(y_val_x, pred_x))
    rmse_y = np.sqrt(mean_squared_error(y_val_y, pred_y))
    rmse = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
    
    cv_scores_cat.append(rmse)

cat_cv_mean = np.mean(cv_scores_cat)
cat_cv_std = np.std(cv_scores_cat)

print(f"\n✅ CatBoost CV 결과:")
print(f"   평균 RMSE: {cat_cv_mean:.4f}m (±{cat_cv_std:.4f})")
print(f"   Fold별: {[f'{s:.4f}' for s in cv_scores_cat]}")

# 전체 데이터로 최종 학습
print(f"\n🔄 전체 데이터로 최종 학습 중...")
cat_model_x = cb.CatBoostRegressor(**cat_params)
cat_model_x.fit(X, y_x, verbose=False)

cat_model_y = cb.CatBoostRegressor(**cat_params)
cat_model_y.fit(X, y_y, verbose=False)

# 저장
with open(MODEL_DIR / 'catboost_model_v6.pkl', 'wb') as f:
    pickle.dump({'model_x': cat_model_x, 'model_y': cat_model_y}, f)

print(f"✅ 저장: catboost_model_v6.pkl")

# =================================================================
# 7. 최종 결과 요약
# =================================================================
print("\n" + "=" * 60)
print("📊 최종 CV 결과 요약")
print("=" * 60)

print(f"\n{'모델':<15} {'v4 CV':>10} {'v6 CV':>10} {'개선':>10}")
print("-" * 50)

# v4 결과 (하드코딩)
v4_xgb = 18.73
v4_lgb = 18.64
v4_cat = 18.73

print(f"{'XGBoost':<15} {v4_xgb:>9.2f}m {xgb_cv_mean:>9.2f}m {xgb_cv_mean - v4_xgb:>+9.2f}m")
print(f"{'LightGBM':<15} {v4_lgb:>9.2f}m {lgb_cv_mean:>9.2f}m {lgb_cv_mean - v4_lgb:>+9.2f}m")
print(f"{'CatBoost':<15} {v4_cat:>9.2f}m {cat_cv_mean:>9.2f}m {cat_cv_mean - v4_cat:>+9.2f}m")

v4_avg = (v4_xgb + v4_lgb + v4_cat) / 3
v6_avg = (xgb_cv_mean + lgb_cv_mean + cat_cv_mean) / 3

print("-" * 50)
print(f"{'평균':<15} {v4_avg:>9.2f}m {v6_avg:>9.2f}m {v6_avg - v4_avg:>+9.2f}m")

# 개선율 계산
improvement = ((v4_avg - v6_avg) / v4_avg) * 100

print(f"\n✨ 총 개선: {improvement:+.2f}%")

if v6_avg < v4_avg:
    print(f"🎉 Phase 6 피처가 효과적입니다!")
else:
    print(f"⚠️ Phase 6 피처 효과가 미미하거나 악화되었습니다.")
    print(f"   → 피처 선택 또는 하이퍼파라미터 튜닝 필요")

print("\n" + "=" * 60)
print("✅ 모든 모델 학습 완료!")
print("=" * 60)

print("\n다음 단계:")
print("  1. OOF 생성: python src/models/generate_oof_predictions_v6.py")
print("  2. Stacking: python src/models/predict_stacking_v6.py")
print("  3. 제출!")
