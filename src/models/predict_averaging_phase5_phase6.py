"""
Averaging Ensemble v2: Phase 5 (Stacking) + Phase 6 (Error Analysis)

전략:
- Phase 5 (Stacking): 16.5316m - 최고 기록
- Phase 6 (Error Analysis): 16.5622m - 차이 0.03m만!
- Phase 6의 에러 타겟팅 피처가 일부 케이스에서 효과 있을 수 있음

최종 예측 = 0.8 * Phase 5 + 0.2 * Phase 6

예상 효과: LB 16.50-16.54m
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 경로 설정
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SUBMISSION_DIR = BASE_DIR / 'submissions'

print("=" * 60)
print("🎯 Averaging Ensemble v2: Phase 5 + Phase 6")
print("=" * 60)

# ============================================
# 기존 예측 결과 로드
# ============================================
print("\n📂 기존 예측 결과 로드 중...")

# Phase 5: Stacking (LightGBM Meta-Learner)
phase5_path = SUBMISSION_DIR / 'submission_stacking_lgb.csv'
if not phase5_path.exists():
    raise FileNotFoundError(f"❌ Phase 5 파일 없음: {phase5_path}")
phase5_pred = pd.read_csv(phase5_path)
print(f"   ✅ Phase 5 (Stacking): {len(phase5_pred)} episodes (LB 16.5316)")

# Phase 6: Error Analysis
phase6_path = SUBMISSION_DIR / 'submission_stacking_v6.csv'
if not phase6_path.exists():
    raise FileNotFoundError(f"❌ Phase 6 파일 없음: {phase6_path}")
phase6_pred = pd.read_csv(phase6_path)
print(f"   ✅ Phase 6 (Error): {len(phase6_pred)} episodes (LB 16.5622)")

# ============================================
# ID 컬럼명 자동 감지
# ============================================
print("\n🔍 데이터 검증 중...")

# 가능한 ID 컬럼명들
possible_id_cols = ['game_episode', 'ID', 'id', 'episode_id']

id_col = None
for col in possible_id_cols:
    if col in phase5_pred.columns and col in phase6_pred.columns:
        id_col = col
        print(f"   ✅ ID 컬럼 발견: '{id_col}'")
        break

if id_col is None:
    print("   ⚠️ ID 컬럼 없음 → 인덱스 기준으로 진행")
    if len(phase5_pred) != len(phase6_pred):
        raise ValueError(f"❌ 행 개수 불일치! Phase 5: {len(phase5_pred)}, Phase 6: {len(phase6_pred)}")
    print(f"   ✅ 행 개수 일치: {len(phase5_pred)} episodes")
else:
    # ID 일치 확인
    if not (phase5_pred[id_col] == phase6_pred[id_col]).all():
        raise ValueError(f"❌ {id_col} 불일치! Phase 5와 Phase 6의 순서가 다릅니다.")
    print(f"   ✅ {id_col} 일치 확인 완료")

# ============================================
# Averaging Ensemble
# ============================================
print("\n🔄 Averaging Ensemble 수행 중...")

# 가중치: Phase 5 (80%) + Phase 6 (20%)
# Phase 5가 훨씬 좋으므로 높은 비중
W5, W6 = 0.8, 0.2

final_pred = phase5_pred.copy()
final_pred['end_x'] = W5 * phase5_pred['end_x'] + W6 * phase6_pred['end_x']
final_pred['end_y'] = W5 * phase5_pred['end_y'] + W6 * phase6_pred['end_y']

print(f"   📊 가중치: Phase 5 ({W5}) + Phase 6 ({W6})")
print(f"   💡 Phase 5 위주 (차이 0.03m만 나므로 보수적 접근)")

# ============================================
# 통계 분석
# ============================================
print("\n📈 예측 통계:")
print(f"   Phase 5 평균: end_x={phase5_pred['end_x'].mean():.2f}, end_y={phase5_pred['end_y'].mean():.2f}")
print(f"   Phase 6 평균: end_x={phase6_pred['end_x'].mean():.2f}, end_y={phase6_pred['end_y'].mean():.2f}")
print(f"   최종 평균:    end_x={final_pred['end_x'].mean():.2f}, end_y={final_pred['end_y'].mean():.2f}")

# ============================================
# 저장
# ============================================
output_path = SUBMISSION_DIR / 'submission_averaging_v2_phase5_phase6.csv'
final_pred.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("✅ Averaging Ensemble v2 완료!")
print("=" * 60)
print(f"💾 저장 경로: {output_path}")
print(f"📊 예측 개수: {len(final_pred)} episodes")
print(f"\n🎯 예상 LB: 16.50-16.54m")
print(f"🎯 Phase 5 (16.5316) 대비: ±0.02m 내외 예상")
print("=" * 60)

# ============================================
# 추가 분석: 차이 통계
# ============================================
print("\n📊 Phase 5 vs Phase 6 차이 분석:")
diff_x = np.abs(phase5_pred['end_x'] - phase6_pred['end_x'])
diff_y = np.abs(phase5_pred['end_y'] - phase6_pred['end_y'])

print(f"   end_x 평균 차이: {diff_x.mean():.2f}m (std: {diff_x.std():.2f}m)")
print(f"   end_y 평균 차이: {diff_y.mean():.2f}m (std: {diff_y.std():.2f}m)")
print(f"   최대 차이: end_x={diff_x.max():.2f}m, end_y={diff_y.max():.2f}m")
print(f"   최소 차이: end_x={diff_x.min():.2f}m, end_y={diff_y.min():.2f}m")

# 차이가 큰 케이스
large_diff = (diff_x > 5.0) | (diff_y > 5.0)
print(f"\n   💡 차이 5m 이상 케이스: {large_diff.sum()}개 ({large_diff.sum()/len(final_pred)*100:.1f}%)")

if large_diff.sum() > 0:
    print(f"      → Phase 6의 에러 타겟팅 피처가 이 케이스들에 효과 있을 수 있음")
else:
    print(f"      → Phase 5와 Phase 6가 거의 비슷한 예측")

# Phase 5 대비 변화량
change_x = np.abs(final_pred['end_x'] - phase5_pred['end_x'])
change_y = np.abs(final_pred['end_y'] - phase5_pred['end_y'])
print(f"\n📊 Phase 5 대비 최종 예측 변화:")
print(f"   평균 변화: end_x={change_x.mean():.2f}m, end_y={change_y.mean():.2f}m")
print(f"   최대 변화: end_x={change_x.max():.2f}m, end_y={change_y.max():.2f}m")

print("\n" + "=" * 60)
print("🎉 다음 단계:")
print("   1. submissions/submission_averaging_v2_phase5_phase6.csv 제출")
print("   2. LB 스코어 확인")
print("   3. 결과에 따라:")
print("      - 개선 시: 가중치 미세 조정 (0.85/0.15, 0.75/0.25)")
print("      - 악화 시: 옵션 B (5-Model Stacking)로 전환")
print("=" * 60)
