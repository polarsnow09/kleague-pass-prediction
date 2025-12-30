"""
Meta-Learner 학습 스크립트

OOF 예측을 사용하여 Stacking 앙상블의 Meta-Learner를 학습합니다.

사용법:
    python src/models/train_meta_learner.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import lightgbm as lgb

# 경로 설정
DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Meta-Learner 학습 시작")
print("=" * 60)

# 1. OOF 데이터 로드
print("\n📂 OOF 데이터 로드 중...")
oof_df = pd.read_csv(DATA_DIR / 'oof_predictions.csv')
print(f"✅ Shape: {oof_df.shape}")

# 2. Meta-Features 및 타겟 분리
print("\n🔍 Meta-Features 준비 중...")

# Meta-Features: 3개 모델의 예측값 (6개)
meta_features = [
    'xgb_pred_x', 'xgb_pred_y',
    'lgb_pred_x', 'lgb_pred_y',
    'cat_pred_x', 'cat_pred_y'
]

X_meta = oof_df[meta_features].values
y_true_x = oof_df['true_x'].values
y_true_y = oof_df['true_y'].values

print(f"✅ Meta-Features: {X_meta.shape}")
print(f"   - 샘플: {len(X_meta):,}개")
print(f"   - 피처: {len(meta_features)}개")

# 3. 상관관계 확인
print("\n📊 Base 모델 예측 간 상관관계:")
corr = oof_df[meta_features].corr()
print(corr.round(3))

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("1️⃣ Ridge Regression Meta-Learner")
print("=" * 60)

# Ridge 파라미터 탐색
alphas = [0.1, 1.0, 10.0, 100.0]
best_alpha_x = None
best_alpha_y = None
best_score_x = float('inf')
best_score_y = float('inf')

print("\n🔍 Alpha 튜닝 (end_x)...")
for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    # Negative MSE이므로 음수 제거
    scores = -cross_val_score(ridge, X_meta, y_true_x, 
                               cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(scores.mean())
    print(f"   alpha={alpha:6.1f} → RMSE: {rmse:.4f}m")
    
    if rmse < best_score_x:
        best_score_x = rmse
        best_alpha_x = alpha

print(f"✅ 최적 alpha (end_x): {best_alpha_x} (RMSE: {best_score_x:.4f}m)")

print("\n🔍 Alpha 튜닝 (end_y)...")
for alpha in alphas:
    ridge = Ridge(alpha=alpha, random_state=42)
    scores = -cross_val_score(ridge, X_meta, y_true_y, 
                               cv=5, scoring='neg_mean_squared_error')
    rmse = np.sqrt(scores.mean())
    print(f"   alpha={alpha:6.1f} → RMSE: {rmse:.4f}m")
    
    if rmse < best_score_y:
        best_score_y = rmse
        best_alpha_y = alpha

print(f"✅ 최적 alpha (end_y): {best_alpha_y} (RMSE: {best_score_y:.4f}m)")

# 최적 파라미터로 학습
print("\n🎓 최종 Ridge 모델 학습 중...")
ridge_x = Ridge(alpha=best_alpha_x, random_state=42)
ridge_y = Ridge(alpha=best_alpha_y, random_state=42)

ridge_x.fit(X_meta, y_true_x)
ridge_y.fit(X_meta, y_true_y)

# 학습 데이터 예측 (sanity check)
pred_x = ridge_x.predict(X_meta)
pred_y = ridge_y.predict(X_meta)

rmse_x = np.sqrt(mean_squared_error(y_true_x, pred_x))
rmse_y = np.sqrt(mean_squared_error(y_true_y, pred_y))
rmse_total = np.sqrt((rmse_x**2 + rmse_y**2) / 2)

print(f"\n✅ Ridge 학습 완료!")
print(f"   - end_x RMSE: {rmse_x:.4f}m")
print(f"   - end_y RMSE: {rmse_y:.4f}m")
print(f"   - Total RMSE: {rmse_total:.4f}m")

# Ridge 가중치 출력
print(f"\n📊 Ridge 가중치 (end_x):")
for i, (feat, coef) in enumerate(zip(meta_features, ridge_x.coef_)):
    print(f"   {feat:15s}: {coef:7.4f}")
print(f"   intercept      : {ridge_x.intercept_:7.4f}")

print(f"\n📊 Ridge 가중치 (end_y):")
for i, (feat, coef) in enumerate(zip(meta_features, ridge_y.coef_)):
    print(f"   {feat:15s}: {coef:7.4f}")
print(f"   intercept      : {ridge_y.intercept_:7.4f}")

# 모델 저장
ridge_path_x = MODEL_DIR / 'meta_ridge_x.pkl'
ridge_path_y = MODEL_DIR / 'meta_ridge_y.pkl'

with open(ridge_path_x, 'wb') as f:
    pickle.dump(ridge_x, f)
with open(ridge_path_y, 'wb') as f:
    pickle.dump(ridge_y, f)

print(f"\n💾 Ridge 모델 저장 완료:")
print(f"   - {ridge_path_x}")
print(f"   - {ridge_path_y}")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("2️⃣ LightGBM Meta-Learner (비교용)")
print("=" * 60)

print("\n🎓 LightGBM Meta-Learner 학습 중...")

# LightGBM 파라미터 (가볍게)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 15,  # 작게 (과적합 방지)
    'learning_rate': 0.05,
    'n_estimators': 100,  # 적게
    'verbosity': -1,
    'random_state': 42
}

# end_x 모델
lgb_x = lgb.LGBMRegressor(**params)
lgb_x.fit(X_meta, y_true_x)
pred_x = lgb_x.predict(X_meta)
lgb_rmse_x = np.sqrt(mean_squared_error(y_true_x, pred_x))

# end_y 모델
lgb_y = lgb.LGBMRegressor(**params)
lgb_y.fit(X_meta, y_true_y)
pred_y = lgb_y.predict(X_meta)
lgb_rmse_y = np.sqrt(mean_squared_error(y_true_y, pred_y))

lgb_rmse_total = np.sqrt((lgb_rmse_x**2 + lgb_rmse_y**2) / 2)

print(f"\n✅ LightGBM 학습 완료!")
print(f"   - end_x RMSE: {lgb_rmse_x:.4f}m")
print(f"   - end_y RMSE: {lgb_rmse_y:.4f}m")
print(f"   - Total RMSE: {lgb_rmse_total:.4f}m")

# LightGBM 피처 중요도
print(f"\n📊 LightGBM 피처 중요도 (end_x):")
importances_x = lgb_x.feature_importances_
for feat, imp in sorted(zip(meta_features, importances_x), 
                        key=lambda x: x[1], reverse=True):
    print(f"   {feat:15s}: {imp:7.0f}")

print(f"\n📊 LightGBM 피처 중요도 (end_y):")
importances_y = lgb_y.feature_importances_
for feat, imp in sorted(zip(meta_features, importances_y), 
                        key=lambda x: x[1], reverse=True):
    print(f"   {feat:15s}: {imp:7.0f}")

# 모델 저장
lgb_path_x = MODEL_DIR / 'meta_lgb_x.pkl'
lgb_path_y = MODEL_DIR / 'meta_lgb_y.pkl'

with open(lgb_path_x, 'wb') as f:
    pickle.dump(lgb_x, f)
with open(lgb_path_y, 'wb') as f:
    pickle.dump(lgb_y, f)

print(f"\n💾 LightGBM 모델 저장 완료:")
print(f"   - {lgb_path_x}")
print(f"   - {lgb_path_y}")

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("📊 Meta-Learner 비교")
print("=" * 60)

print(f"\nRidge     : {rmse_total:.4f}m")
print(f"LightGBM  : {lgb_rmse_total:.4f}m")

if rmse_total < lgb_rmse_total:
    print(f"\n🏆 Ridge가 더 우수! (차이: {lgb_rmse_total - rmse_total:.4f}m)")
    recommended = "Ridge"
else:
    print(f"\n🏆 LightGBM이 더 우수! (차이: {rmse_total - lgb_rmse_total:.4f}m)")
    recommended = "LightGBM"

# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("🎉 Meta-Learner 학습 완료!")
print("=" * 60)

print(f"\n✅ 저장된 모델:")
print(f"   - Ridge: meta_ridge_x.pkl, meta_ridge_y.pkl")
print(f"   - LightGBM: meta_lgb_x.pkl, meta_lgb_y.pkl")

print(f"\n💡 권장: {recommended} Meta-Learner 사용")

print(f"\n다음 단계:")
print(f"1. Base 모델 학습 (전체 데이터): python src/models/train_base_models.py")
print(f"2. Stacking 예측: python src/models/predict_stacking.py --meta {recommended.lower()}")
