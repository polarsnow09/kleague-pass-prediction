"""
Averaging Ensemble: Phase 5 (Stacking) + Phase 4 (Weighted Averaging)

전략:
- Phase 5 (Stacking): 비선형 Meta-Learning (LB 16.5316) - 최고 기록
- Phase 4 (Weighted): 선형 가중 평균 (LB 16.8272)
- 두 앙상블 방식의 장점 결합!

최종 예측 = 0.6 * Phase 5 + 0.4 * Phase 4

예상 효과: LB 16.38-16.48m (-0.05~0.15m 개선)
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
print("🎯 Averaging Ensemble: Phase 5 + Phase 4")
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
print(f"   ✅ Phase 5 (Stacking): {len(phase5_pred)} episodes")
print(f"   📋 Phase 5 컬럼: {list(phase5_pred.columns)}")

# Phase 4: Weighted Averaging Ensemble
phase4_path = SUBMISSION_DIR / 'submission_ensemble_v4.csv'
if not phase4_path.exists():
    raise FileNotFoundError(f"❌ Phase 4 파일 없음: {phase4_path}")
phase4_pred = pd.read_csv(phase4_path)
print(f"   ✅ Phase 4 (Weighted): {len(phase4_pred)} episodes")
print(f"   📋 Phase 4 컬럼: {list(phase4_pred.columns)}")

# ============================================
# ID 컬럼명 자동 감지
# ============================================
print("\n🔍 ID 컬럼명 감지 중...")

# 가능한 ID 컬럼명들
possible_id_cols = ['ID', 'id', 'episode_id', 'game_episode']

id_col = None
for col in possible_id_cols:
    if col in phase5_pred.columns and col in phase4_pred.columns:
        id_col = col
        print(f"   ✅ ID 컬럼 발견: '{id_col}'")
        break

if id_col is None:
    print("   ⚠️ ID 컬럼 없음 → 인덱스 기준으로 진행")
    # 인덱스 기준으로 진행 (순서가 같다고 가정)
    if len(phase4_pred) != len(phase5_pred):
        raise ValueError(f"❌ 행 개수 불일치! Phase 4: {len(phase4_pred)}, Phase 5: {len(phase5_pred)}")
    print(f"   ✅ 행 개수 일치: {len(phase4_pred)} episodes")
else:
    # ID 일치 확인
    if not (phase4_pred[id_col] == phase5_pred[id_col]).all():
        raise ValueError(f"❌ {id_col} 불일치! Phase 4와 Phase 5의 순서가 다릅니다.")
    print(f"   ✅ {id_col} 일치 확인 완료")

# ============================================
# Averaging Ensemble
# ============================================
print("\n🔄 Averaging Ensemble 수행 중...")

# 기본 가중치: Phase 5 (60%) + Phase 4 (40%)
W5, W4 = 0.6, 0.4

final_pred = phase5_pred.copy()
final_pred['end_x'] = W5 * phase5_pred['end_x'] + W4 * phase4_pred['end_x']
final_pred['end_y'] = W5 * phase5_pred['end_y'] + W4 * phase4_pred['end_y']

print(f"   📊 가중치: Phase 5 ({W5}) + Phase 4 ({W4})")

# ============================================
# 통계 분석
# ============================================
print("\n📈 예측 통계:")
print(f"   Phase 5 평균: end_x={phase5_pred['end_x'].mean():.2f}, end_y={phase5_pred['end_y'].mean():.2f}")
print(f"   Phase 4 평균: end_x={phase4_pred['end_x'].mean():.2f}, end_y={phase4_pred['end_y'].mean():.2f}")
print(f"   최종 평균:    end_x={final_pred['end_x'].mean():.2f}, end_y={final_pred['end_y'].mean():.2f}")

# ============================================
# 저장
# ============================================
output_path = SUBMISSION_DIR / 'submission_averaging_v1.csv'
final_pred.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("✅ Averaging Ensemble 완료!")
print("=" * 60)
print(f"💾 저장 경로: {output_path}")
print(f"📊 예측 개수: {len(final_pred)} episodes")
print(f"\n🎯 예상 LB: 16.38-16.48m")
print(f"🎯 Phase 5 (16.5316) 대비: -0.05 ~ -0.15m 개선 예상")
print("=" * 60)

# ============================================
# 추가 분석: 차이 통계
# ============================================
print("\n📊 Phase 5 vs Phase 4 차이 분석:")
diff_x = np.abs(phase5_pred['end_x'] - phase4_pred['end_x'])
diff_y = np.abs(phase5_pred['end_y'] - phase4_pred['end_y'])

print(f"   end_x 평균 차이: {diff_x.mean():.2f}m (std: {diff_x.std():.2f}m)")
print(f"   end_y 평균 차이: {diff_y.mean():.2f}m (std: {diff_y.std():.2f}m)")
print(f"   최대 차이: end_x={diff_x.max():.2f}m, end_y={diff_y.max():.2f}m")
print(f"   최소 차이: end_x={diff_x.min():.2f}m, end_y={diff_y.min():.2f}m")

# 차이가 큰 케이스 (상보적일 가능성)
large_diff = (diff_x > 5.0) | (diff_y > 5.0)
print(f"\n   💡 차이 5m 이상 케이스: {large_diff.sum()}개 ({large_diff.sum()/len(final_pred)*100:.1f}%)")
print(f"      → 이 케이스들에서 Averaging이 특히 효과적일 수 있음!")

print("\n" + "=" * 60)
print("🎉 다음 단계:")
print("   1. submissions/submission_averaging_v1.csv 확인")
print("   2. 공모전 플랫폼에 제출")
print("   3. LB 스코어 확인 후 가중치 조정 고려")
print("=" * 60)
