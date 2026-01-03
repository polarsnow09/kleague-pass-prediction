"""
Phase 6 Stacking 앙상블 예측 스크립트

Phase 6 피처 포함:
- 구역별 특화 피처
- 최종 구역 미진입 타겟팅
- 측면 vs 중앙 차별화
- 득점 상황별 전술 변화
- 극단값 특수 처리
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# 경로 설정
DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MODEL_DIR = Path('models')
SUBMISSION_DIR = Path('submissions')
SUBMISSION_DIR.mkdir(exist_ok=True)

# Phase 6 피처 생성 함수 임포트
import sys
sys.path.append('src')
from features.build_feature import build_baseline_features, add_previous_action_features
from features.build_phase6_features import build_phase6_features

print("=" * 60)
print("Phase 6 Stacking 앙상블 예측 시작")
print("=" * 60)

# ===================================================================
# STEP 1: 학습된 모델 로드
# ===================================================================
print("\n" + "=" * 60)
print("STEP 1: 학습된 모델 로드")
print("=" * 60)

print("\n📂 Base 모델 로드 중...")
with open(MODEL_DIR / 'baseline_model_v6.pkl', 'rb') as f:
    xgb_models = pickle.load(f)
    xgb_model_x = xgb_models['model_x']
    xgb_model_y = xgb_models['model_y']

with open(MODEL_DIR / 'lgb_model_v6.pkl', 'rb') as f:
    lgb_models = pickle.load(f)
    lgb_model_x = lgb_models['model_x']
    lgb_model_y = lgb_models['model_y']

with open(MODEL_DIR / 'catboost_model_v6.pkl', 'rb') as f:
    cat_models = pickle.load(f)
    cat_model_x = cat_models['model_x']
    cat_model_y = cat_models['model_y']

print("✅ Base 모델 로드 완료")

print("\n📂 Meta-Learner 로드 중...")
with open(MODEL_DIR / 'meta_lgb_x.pkl', 'rb') as f:
    meta_model_x = pickle.load(f)
with open(MODEL_DIR / 'meta_lgb_y.pkl', 'rb') as f:
    meta_model_y = pickle.load(f)

print("✅ Meta-Learner 로드 완료")

print("\n📂 LabelEncoder 로드 중...")
with open(MODEL_DIR / 'label_encoders_v6.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

print("✅ LabelEncoder 로드 완료")

# ===================================================================
# STEP 2: Phase 4 통계 계산 (Train 전체)
# ===================================================================
print("\n" + "=" * 60)
print("STEP 2: Phase 4 통계 계산")
print("=" * 60)

print("\n📊 Train 데이터 분석 중...")
train_full = pd.read_csv(RAW_DIR / 'train.csv')
passes = train_full[train_full['type_name'] == 'Pass'].copy()

# 통계 변수 생성
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

# 선수 통계
player_stats = passes.groupby('player_id').agg({
    'pass_distance': 'mean',
    'is_forward': 'mean',
    'is_success': 'mean',
    'player_id': 'count'
}).rename(columns={'player_id': 'pass_count'}).to_dict('index')

# 팀 통계
team_stats = passes.groupby('team_id').agg({
    'pass_distance': 'mean',
    'is_wide': 'mean'
}).rename(columns={'is_wide': 'attack_style'}).to_dict('index')

# 전체 평균
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

# ===================================================================
# STEP 3: Test 데이터 로드
# ===================================================================
print("\n" + "=" * 60)
print("STEP 3: Test 데이터 로드")
print("=" * 60)

print("\n📂 Test 데이터 로드 중...")
test = pd.read_csv(RAW_DIR / 'test.csv')
match_info = pd.read_csv(RAW_DIR / 'match_info.csv')
print(f"✅ Test 샘플: {len(test):,}개")

# ===================================================================
# STEP 4: Episode별 예측
# ===================================================================
print("\n" + "=" * 60)
print("STEP 4: Episode별 예측")
print("=" * 60)

# 피처 목록 (Phase 6)
train_v6 = pd.read_csv(PROCESSED_DIR / 'train_final_passes_v6.csv')
feature_cols = [col for col in train_v6.columns if col not in ['end_x', 'end_y', 'game_episode']]

print(f"\n📋 사용할 피처: {len(feature_cols)}개")

predictions = []

for idx, row in tqdm(test.iterrows(), total=len(test), desc="예측"):
    game_episode = row['game_episode']
    csv_path = RAW_DIR / row['path']
    
    # Episode 데이터 로드
    episode_df = pd.read_csv(csv_path)
    
    # Phase 1-2 피처 생성
    episode_df = build_baseline_features(episode_df)
    episode_df = add_previous_action_features(episode_df)
    
    # Phase 3 피처 생성 (고급 시계열)
    try:
        from features.advanced_features import build_phase3_features
        episode_df = build_phase3_features(episode_df)
    except ImportError:
        # Phase 3 피처 기본값
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
    
    # Phase 4 피처 추가
    last_pass = episode_df[episode_df['type_name'] == 'Pass'].iloc[-1]
    player_id = last_pass['player_id']
    team_id = last_pass['team_id']
    game_id = last_pass['game_id']
    is_home = last_pass['is_home']
    time_seconds = last_pass['time_seconds']
    
    # 선수 통계
    p_stats = player_stats.get(player_id, global_player)
    episode_df['player_avg_pass_distance'] = p_stats['pass_distance']
    episode_df['player_forward_ratio'] = p_stats['is_forward']
    episode_df['player_success_rate'] = p_stats['is_success']
    episode_df['player_pass_count'] = p_stats['pass_count']
    
    # 팀 통계
    t_stats = team_stats.get(team_id, global_team)
    episode_df['team_avg_pass_distance'] = t_stats['pass_distance']
    episode_df['team_attack_style'] = t_stats['attack_style']
    
    # 경기 흐름
    match = match_info[match_info['game_id'] == game_id].iloc[0]
    episode_df['score_diff'] = np.where(
        is_home,
        match['home_score'] - match['away_score'],
        match['away_score'] - match['home_score']
    )
    episode_df['match_period_normalized'] = time_seconds / 5400
    episode_df['is_late_game'] = int(time_seconds >= 4050)
    
    # ⭐ Phase 6 피처 생성
    episode_df = build_phase6_features(episode_df)
    
    # 최종 Pass 선택
    final_pass = episode_df[episode_df['type_name'] == 'Pass'].iloc[-1:].copy()
    
    # 범주형 인코딩
    for col, le in label_encoders.items():
        if col in final_pass.columns:
            val = str(final_pass[col].values[0])
            if val in le.classes_:
                final_pass[col] = le.transform([val])[0]
            else:
                final_pass[col] = 0  # unseen 값
    
    # 피처 추출 (누락 피처는 0으로 채우기)
    X_test = []
    for col in feature_cols:
        if col in final_pass.columns:
            X_test.append(final_pass[col].values[0])
        else:
            X_test.append(0)  # 누락 피처 기본값
    
    X_test = np.array([X_test])
    
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
    
    # Meta-Learner 최종 예측
    final_x = meta_model_x.predict(meta_features)[0]
    final_y = meta_model_y.predict(meta_features)[0]
    
    predictions.append({
        'game_episode': game_episode,
        'end_x': final_x,
        'end_y': final_y
    })

print(f"\n✅ 예측 완료: {len(predictions):,}개")

# ===================================================================
# STEP 5: 제출 파일 생성
# ===================================================================
print("\n" + "=" * 60)
print("STEP 5: 제출 파일 생성")
print("=" * 60)

submission = pd.DataFrame(predictions)
submission = submission[['game_episode', 'end_x', 'end_y']]

output_path = SUBMISSION_DIR / 'submission_stacking_v6.csv'
submission.to_csv(output_path, index=False)

print(f"\n💾 제출 파일 저장 완료:")
print(f"   {output_path}")
print(f"   Shape: {submission.shape}")

print("\n📊 예측 통계:")
print(submission[['end_x', 'end_y']].describe())

# ===================================================================
print("\n" + "=" * 60)
print("🎉 Phase 6 Stacking 앙상블 예측 완료!")
print("=" * 60)

print(f"\n✅ 제출 파일: {output_path}")
print(f"\n예상 결과:")
print(f"   v4 LB: 16.83m")
print(f"   v6 LB: 16.5~16.7m (예상)")
print(f"\n다음 단계:")
print(f"1. 제출 파일 확인")
print(f"2. 리더보드 제출")
print(f"3. 결과 대기!")
