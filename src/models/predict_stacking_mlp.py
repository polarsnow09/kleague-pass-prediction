"""
MLP Stacking 앙상블 예측 스크립트

기존 predict_stacking.py와 동일하지만 Meta-Learner로 MLP 사용

사용법:
    python src/models/predict_stacking_mlp.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

# 경로 설정
DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path('models')
SUBMISSION_DIR = Path('submissions')
SUBMISSION_DIR.mkdir(exist_ok=True)

# Phase 피처 생성 함수들 임포트
import sys
sys.path.append('src')
from features.build_feature import build_baseline_features, add_previous_action_features

print("=" * 60)
print("MLP Stacking 앙상블 예측 시작")
print("=" * 60)

# ===================================================================
# STEP 0: MLP 모델 클래스 정의 (학습 시와 동일)
# ===================================================================
class MLPMetaLearner(nn.Module):
    def __init__(self, input_dim=6, hidden_dim1=32, hidden_dim2=16, dropout=0.2):
        super(MLPMetaLearner, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# ===================================================================
# STEP 1: Base 모델 학습 (동일)
# ===================================================================
print("\n" + "=" * 60)
print("STEP 1: Base 모델 학습 (전체 Train 데이터)")
print("=" * 60)

print("\n📂 Train 데이터 로드 중...")
train_df = pd.read_csv(PROCESSED_DIR / 'train_final_passes_v4.csv')
print(f"✅ Shape: {train_df.shape}")

# 피처 및 타겟 분리
target_cols = ['end_x', 'end_y']
feature_cols = [col for col in train_df.columns if col not in target_cols + ['game_episode']]

# 범주형 피처 인코딩
categorical_cols = train_df[feature_cols].select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

if len(categorical_cols) > 0:
    print(f"\n📝 범주형 피처 인코딩: {categorical_cols}")
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        label_encoders[col] = le

X_train = train_df[feature_cols].values
y_train_x = train_df['end_x'].values
y_train_y = train_df['end_y'].values

print(f"✅ 피처: {len(feature_cols)}개")
print(f"✅ 샘플: {len(X_train):,}개")

# -------------------------------------------------------------------
print("\n🎓 XGBoost 학습 중...")
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'random_state': 42
}

xgb_model_x = xgb.XGBRegressor(**xgb_params)
xgb_model_y = xgb.XGBRegressor(**xgb_params)

xgb_model_x.fit(X_train, y_train_x, verbose=False)
xgb_model_y.fit(X_train, y_train_y, verbose=False)
print("✅ XGBoost 학습 완료")

# -------------------------------------------------------------------
print("\n🎓 LightGBM 학습 중...")
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

train_data_x = lgb.Dataset(X_train, label=y_train_x)
train_data_y = lgb.Dataset(X_train, label=y_train_y)

lgb_model_x = lgb.train(lgb_params, train_data_x, num_boost_round=800)
lgb_model_y = lgb.train(lgb_params, train_data_y, num_boost_round=800)
print("✅ LightGBM 학습 완료")

# -------------------------------------------------------------------
print("\n🎓 CatBoost 학습 중...")
cat_params = {
    'iterations': 500,
    'depth': 8,
    'learning_rate': 0.05,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

cat_model_x = cb.CatBoostRegressor(**cat_params)
cat_model_y = cb.CatBoostRegressor(**cat_params)

cat_model_x.fit(X_train, y_train_x, verbose=False)
cat_model_y.fit(X_train, y_train_y, verbose=False)
print("✅ CatBoost 학습 완료")

# ===================================================================
# STEP 2: MLP Meta-Learner 로드
# ===================================================================
print("\n" + "=" * 60)
print("STEP 2: MLP Meta-Learner 로드")
print("=" * 60)

print("\n📂 MLP Meta-Learner 로드 중...")
with open(MODEL_DIR / 'meta_mlp_x.pkl', 'rb') as f:
    mlp_x_package = pickle.load(f)
with open(MODEL_DIR / 'meta_mlp_y.pkl', 'rb') as f:
    mlp_y_package = pickle.load(f)

# MLP 모델 재구성
arch = mlp_x_package['architecture']
mlp_model_x = MLPMetaLearner(**arch)
mlp_model_y = MLPMetaLearner(**arch)

mlp_model_x.load_state_dict(mlp_x_package['model_state'])
mlp_model_y.load_state_dict(mlp_y_package['model_state'])

mlp_model_x.eval()
mlp_model_y.eval()

# Scaler 로드
scaler = mlp_x_package['scaler']

print("✅ MLP Meta-Learner 로드 완료")

# ===================================================================
# STEP 3: Test 데이터 예측
# ===================================================================
print("\n" + "=" * 60)
print("STEP 3: Test 데이터 예측")
print("=" * 60)

print("\n📂 Test 데이터 로드 중...")
test = pd.read_csv(RAW_DIR / 'test.csv')
match_info = pd.read_csv(RAW_DIR / 'match_info.csv')
print(f"✅ Test 샘플: {len(test):,}개")

# Phase 4 통계 계산 (동일)
print("\n📊 Phase 4 통계 계산 중...")
train_full = pd.read_csv(RAW_DIR / 'train.csv')
passes = train_full[train_full['type_name'] == 'Pass'].copy()

passes['pass_distance'] = np.sqrt(
    (passes['end_x'] - passes['start_x'])**2 + 
    (passes['end_y'] - passes['start_y'])**2
)
passes['forward_distance'] = np.where(
    passes['is_home'],
    passes['end_x'] - passes['start_x'],
    passes['start_x'] - passes['end_x']
)
passes['is_forward'] = (passes['forward_distance'] > 0).astype(int)
passes['is_success'] = (passes['result_name'] == 'Successful').astype(int)
passes['is_wide'] = ((passes['start_y'] < 20) | (passes['start_y'] > 48)).astype(int)

player_stats = passes.groupby('player_id').agg({
    'pass_distance': 'mean',
    'is_forward': 'mean',
    'is_success': 'mean',
    'player_id': 'count'
}).rename(columns={'player_id': 'pass_count'}).to_dict('index')

team_stats = passes.groupby('team_id').agg({
    'pass_distance': 'mean',
    'is_wide': 'mean'
}).rename(columns={'is_wide': 'attack_style'}).to_dict('index')

global_player = {
    'pass_distance': passes['pass_distance'].mean(),
    'is_forward': passes['is_forward'].mean(),
    'is_success': passes['is_success'].mean(),
    'pass_count': 50
}
global_team = {
    'pass_distance': passes['pass_distance'].mean(),
    'attack_style': passes['is_wide'].mean()
}

print("✅ Phase 4 통계 계산 완료")

# -------------------------------------------------------------------
print("\n🔮 Episode별 예측 시작...")

predictions = []

for idx, row in tqdm(test.iterrows(), total=len(test), desc="예측"):
    game_episode = row['game_episode']
    csv_path = RAW_DIR / row['path']
    
    # Episode 데이터 로드
    episode_df = pd.read_csv(csv_path)
    
    # Phase 1-2 피처 생성
    episode_df = build_baseline_features(episode_df)
    episode_df = add_previous_action_features(episode_df)
    
    # Phase 3 피처 생성
    try:
        from features.advanced_features import build_phase3_features
        episode_df = build_phase3_features(episode_df)
    except ImportError:
        phase3_cols = [
            'rolling_mean_distance_3', 'rolling_std_distance_3', 
            'rolling_mean_direction_x_3', 'rolling_mean_direction_y_3',
            'rolling_mean_distance_5', 'rolling_std_distance_5',
            'rolling_mean_direction_x_5', 'rolling_mean_direction_y_5',
            'cumulative_distance', 'cumulative_forward', 'cumulative_lateral',
            'forward_lateral_ratio', 'pass_velocity', 'avg_episode_velocity',
            'velocity_change', 'recent_3_avg_velocity', 'episode_x_range',
            'episode_y_range', 'touchline_proximity', 'avg_touchline_proximity',
            'is_buildup', 'is_counter', 'is_under_pressure'
        ]
        for col in phase3_cols:
            if col not in episode_df.columns:
                episode_df[col] = 0
    
    # Phase 4 피처 추가 (동일 로직)
    last_pass = episode_df[episode_df['type_name'] == 'Pass'].iloc[-1]
    player_id = last_pass['player_id']
    team_id = last_pass['team_id']
    game_id = last_pass['game_id']
    is_home = last_pass['is_home']
    time_seconds = last_pass['time_seconds']
    
    p_stats = player_stats.get(player_id, global_player)
    episode_df['player_avg_pass_distance'] = p_stats['pass_distance']
    episode_df['player_forward_ratio'] = p_stats['is_forward']
    episode_df['player_success_rate'] = p_stats['is_success']
    episode_df['player_pass_count'] = p_stats['pass_count']
    
    t_stats = team_stats.get(team_id, global_team)
    episode_df['team_avg_pass_distance'] = t_stats['pass_distance']
    episode_df['team_attack_style'] = t_stats['attack_style']
    
    match = match_info[match_info['game_id'] == game_id].iloc[0]
    episode_df['score_diff'] = np.where(
        is_home,
        match['home_score'] - match['away_score'],
        match['away_score'] - match['home_score']
    )
    episode_df['match_period_normalized'] = time_seconds / 5400
    episode_df['is_late_game'] = int(time_seconds >= 4050)
    
    final_pass = episode_df[episode_df['type_name'] == 'Pass'].iloc[-1:].copy()
    
    # 범주형 인코딩
    for col in categorical_cols:
        if col in final_pass.columns:
            le = label_encoders[col]
            val = str(final_pass[col].values[0])
            if val in le.classes_:
                final_pass[col] = le.transform([val])[0]
            else:
                final_pass[col] = 0
    
    X_test = final_pass[feature_cols].values
    
    # Base 모델 예측
    xgb_pred_x = xgb_model_x.predict(X_test)[0]
    xgb_pred_y = xgb_model_y.predict(X_test)[0]
    
    lgb_pred_x = lgb_model_x.predict(X_test)[0]
    lgb_pred_y = lgb_model_y.predict(X_test)[0]
    
    cat_pred_x = cat_model_x.predict(X_test)[0]
    cat_pred_y = cat_model_y.predict(X_test)[0]
    
    # Meta-Features 구성
    meta_features = np.array([[
        xgb_pred_x, xgb_pred_y,
        lgb_pred_x, lgb_pred_y,
        cat_pred_x, cat_pred_y
    ]])
    
    # 정규화 (MLP용)
    meta_features_scaled = scaler.transform(meta_features)
    meta_tensor = torch.FloatTensor(meta_features_scaled)
    
    # MLP 최종 예측
    with torch.no_grad():
        final_x = mlp_model_x(meta_tensor).item()
        final_y = mlp_model_y(meta_tensor).item()
    
    predictions.append({
        'game_episode': game_episode,
        'end_x': final_x,
        'end_y': final_y
    })

print(f"\n✅ 예측 완료: {len(predictions):,}개")

# ===================================================================
# STEP 4: 제출 파일 생성
# ===================================================================
print("\n" + "=" * 60)
print("STEP 4: 제출 파일 생성")
print("=" * 60)

submission = pd.DataFrame(predictions)
submission = submission[['game_episode', 'end_x', 'end_y']]

output_path = SUBMISSION_DIR / 'submission_stacking_mlp.csv'
submission.to_csv(output_path, index=False)

print(f"\n💾 제출 파일 저장 완료:")
print(f"   {output_path}")
print(f"   Shape: {submission.shape}")

print("\n📊 예측 통계:")
print(submission.describe())

# ===================================================================
print("\n" + "=" * 60)
print("🎉 MLP Stacking 앙상블 예측 완료!")
print("=" * 60)

print(f"\n✅ 제출 파일: {output_path}")
print(f"\n다음 단계:")
print(f"1. 제출 파일 확인")
print(f"2. 리더보드 제출")
print(f"3. Phase 5 (LGB) vs Phase 5.1 (MLP) 비교")
