import pandas as pd
from pathlib import Path
import sys

sys.path.append('.')
from src.features.build_feature import build_baseline_features, add_previous_action_features

# Test 파일 하나 로드
test_file = Path('data/raw/test/153363/153363_1.csv')
df = pd.read_csv(test_file)

print("원본 데이터:")
print(f"  - 컬럼: {df.columns.tolist()}")
print(f"  - 샘플:\n{df.head()}")

# 필요한 컬럼 추가
df['end_x'] = df['start_x']
df['end_y'] = df['start_y']
df['game_episode'] = 'test'
df['is_home'] = 1

print("\n추가 후:")
print(f"  - 컬럼: {df.columns.tolist()}")

# Phase 1 피처 생성
df = build_baseline_features(df)

print("\nPhase 1 후:")
print(f"  - 컬럼 수: {len(df.columns)}")
print(f"  - zone_x 존재: {'zone_x' in df.columns}")
print(f"  - zone_x_encoded 존재: {'zone_x_encoded' in df.columns}")

if 'zone_x' in df.columns:
    print(f"  - zone_x 샘플: {df['zone_x'].head()}")

# Phase 2 피처 생성
df = add_previous_action_features(df)

print("\nPhase 2 후:")
print(f"  - 컬럼 수: {len(df.columns)}")
print(f"  - prev_end_x 존재: {'prev_end_x' in df.columns}")

# 마지막 행
last_row = df.iloc[[-1]].copy()  # DataFrame으로 유지
print("\n마지막 행 (예측 대상):")
print(last_row[['start_x', 'start_y', 'zone_x', 'dist_to_target_goal']])

# 인코딩 테스트
print("\n인코딩 테스트:")

zone_x_map = {'defensive': 0, 'midfield': 1, 'attacking': 2}
zone_y_map = {'left': 0, 'center': 1, 'right': 2}
zone_combined_map = {
    'defensive_left': 0, 'defensive_center': 1, 'defensive_right': 2,
    'midfield_left': 3, 'midfield_center': 4, 'midfield_right': 5,
    'attacking_left': 6, 'attacking_center': 7, 'attacking_right': 8
}

last_row['zone_x_encoded'] = last_row['zone_x'].astype(str).map(zone_x_map)
last_row['zone_y_encoded'] = last_row['zone_y'].astype(str).map(zone_y_map)
last_row['zone_combined_encoded'] = last_row['zone_combined'].astype(str).map(zone_combined_map)

print(f"  - zone_x: {last_row['zone_x'].values[0]} → {last_row['zone_x_encoded'].values[0]}")
print(f"  - zone_y: {last_row['zone_y'].values[0]} → {last_row['zone_y_encoded'].values[0]}")
print(f"  - zone_combined: {last_row['zone_combined'].values[0]} → {last_row['zone_combined_encoded'].values[0]}")

# 모델 피처 확인
model_features = [
    'start_x', 'start_y', 'dist_to_target_goal',
    'zone_x_encoded', 'zone_y_encoded', 'zone_combined_encoded',
    'in_penalty_box', 'in_final_third',
    'prev_end_x', 'prev_end_y', 'prev_action_distance',
    'time_since_prev', 'prev_direction_x', 'prev_direction_y',
    'pass_count_in_episode'
]

print("\n모델 입력 피처 확인:")
missing = [f for f in model_features if f not in last_row.columns]
if missing:
    print(f"  ⚠️  누락: {missing}")
else:
    print(f"  ✓ 모든 피처 존재 ({len(model_features)}개)")
    print(f"\n최종 피처 값:")
    print(last_row[model_features].T)