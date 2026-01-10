"""
Averaging Ensemble - Grid Search v2: 추가 공격적 가중치 탐색

현재 패턴:
- 0.80/0.20: 16.5065m
- 0.78/0.22: 16.5049m
- 0.75/0.25: 16.5029m
- 0.70/0.30: 16.5003m ← 현재 최고!

가설: Phase 6 비중 증가 = 성능 향상
목표: 최적점 발견 (0.65/0.35? 0.60/0.40?)
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

print("=" * 70)
print("🔍 Averaging Ensemble - Grid Search v2 (추가 공격적 가중치)")
print("=" * 70)

# ============================================
# 기존 예측 결과 로드
# ============================================
print("\n📂 기존 예측 결과 로드 중...")

phase5_path = SUBMISSION_DIR / 'submission_stacking_lgb.csv'
phase6_path = SUBMISSION_DIR / 'submission_stacking_v6.csv'

phase5_pred = pd.read_csv(phase5_path)
phase6_pred = pd.read_csv(phase6_path)

print(f"   ✅ Phase 5: {len(phase5_pred)} episodes (LB 16.5316)")
print(f"   ✅ Phase 6: {len(phase6_pred)} episodes (LB 16.5622)")

# ============================================
# 현재까지 결과 요약
# ============================================
print("\n📊 현재까지 발견한 패턴:")
print("   ┌──────────────────────────────────────┐")
print("   │ Phase 5  Phase 6    LB Score         │")
print("   ├──────────────────────────────────────┤")
print("   │  0.80  /  0.20  →  16.5065m          │")
print("   │  0.78  /  0.22  →  16.5049m          │")
print("   │  0.75  /  0.25  →  16.5029m          │")
print("   │  0.70  /  0.30  →  16.5003m  ⭐      │")
print("   └──────────────────────────────────────┘")
print("\n   💡 패턴: Phase 6 비중 ↑ = 성능 ↑")
print("   💡 가설: 0.65/0.35가 최적점일 가능성!")

# ============================================
# 추가 Grid Search 가중치 정의
# ============================================
print("\n🎯 추가 Grid Search 가중치 (3개):")

# 더 공격적인 가중치
additional_weights = [
    (0.65, 0.35, "Phase 6 35% (최적점 예상)"),
    (0.60, 0.40, "Phase 6 40% (동등 수준)"),
    (0.55, 0.45, "Phase 6 45% (Phase 6 우세)"),
]

for i, (w5, w6, desc) in enumerate(additional_weights, 6):
    print(f"   {i}. Phase 5 ({w5:.2f}) + Phase 6 ({w6:.2f}) - {desc}")

print("\n" + "=" * 70)

# ============================================
# 각 가중치별로 예측 생성
# ============================================
print("\n🔄 가중치별 예측 생성 중...\n")

results = []

for i, (w5, w6, desc) in enumerate(additional_weights, 6):
    print(f"📊 조합 {i}: Phase 5 ({w5:.2f}) + Phase 6 ({w6:.2f})")
    print(f"   설명: {desc}")
    
    # 예측 생성
    final_pred = phase5_pred.copy()
    final_pred['end_x'] = w5 * phase5_pred['end_x'] + w6 * phase6_pred['end_x']
    final_pred['end_y'] = w5 * phase5_pred['end_y'] + w6 * phase6_pred['end_y']
    
    # 통계
    avg_x = final_pred['end_x'].mean()
    avg_y = final_pred['end_y'].mean()
    
    # Phase 5 대비 변화
    change_x = np.abs(final_pred['end_x'] - phase5_pred['end_x']).mean()
    change_y = np.abs(final_pred['end_y'] - phase5_pred['end_y']).mean()
    
    # Phase 6에 얼마나 가까운지
    distance_to_p6_x = np.abs(final_pred['end_x'] - phase6_pred['end_x']).mean()
    distance_to_p6_y = np.abs(final_pred['end_y'] - phase6_pred['end_y']).mean()
    
    print(f"   평균: end_x={avg_x:.2f}, end_y={avg_y:.2f}")
    print(f"   Phase 5 대비 변화: end_x={change_x:.2f}m, end_y={change_y:.2f}m")
    print(f"   Phase 6까지 거리: end_x={distance_to_p6_x:.2f}m, end_y={distance_to_p6_y:.2f}m")
    
    # 저장
    output_name = f'submission_averaging_grid_{i}_w5{int(w5*100)}_w6{int(w6*100)}.csv'
    output_path = SUBMISSION_DIR / output_name
    final_pred.to_csv(output_path, index=False)
    
    print(f"   💾 저장: {output_name}")
    print()
    
    # 결과 기록
    results.append({
        'num': i,
        'w5': w5,
        'w6': w6,
        'desc': desc,
        'filename': output_name,
        'avg_x': avg_x,
        'avg_y': avg_y,
        'change_x': change_x,
        'change_y': change_y,
        'dist_to_p6_x': distance_to_p6_x,
        'dist_to_p6_y': distance_to_p6_y
    })

# ============================================
# 결과 요약
# ============================================
print("=" * 70)
print("✅ Grid Search v2 완료!")
print("=" * 70)

print("\n📋 생성된 제출 파일 목록:\n")
for r in results:
    print(f"   {r['num']}. {r['filename']}")
    print(f"      가중치: Phase 5 ({r['w5']:.2f}) + Phase 6 ({r['w6']:.2f})")
    print(f"      설명: {r['desc']}")
    print(f"      예측: end_x={r['avg_x']:.2f}m, end_y={r['avg_y']:.2f}m")
    print(f"      Phase 5 대비 변화: {r['change_x']:.2f}m")
    print()

print("=" * 70)
print("🎯 다음 단계:")
print("=" * 70)

print("\n1️⃣ 3개 파일을 순서대로 제출")
print("   - submission_averaging_grid_6_w565_w635.csv (우선 추천 ⭐)")
print("   - submission_averaging_grid_7_w560_w640.csv")
print("   - submission_averaging_grid_8_w555_w645.csv")

print("\n2️⃣ 결과 기록 및 분석")

print("\n3️⃣ 최종 결정")
print("   ┌─────────────────────────────────────────────┐")
print("   │ IF 조합 6-8 중 개선:                        │")
print("   │    → 최고 점수 선택                         │")
print("   │    → 문서화 시작                            │")
print("   │                                             │")
print("   │ IF 모두 16.5003m 이하 (악화 or 비슷):      │")
print("   │    → 0.70/0.30 (16.5003m) 확정             │")
print("   │    → 문서화 시작                            │")
print("   └─────────────────────────────────────────────┘")

print("\n💡 예상 결과:")
print("   시나리오 A (40%): 조합 6 (0.65/0.35) = 16.48-16.49m ⭐")
print("   시나리오 B (40%): 모두 16.50-16.51m (0.70/0.30 최적)")
print("   시나리오 C (20%): 계속 개선 → 0.55/0.45까지!")

print("\n⚠️  중요: 이번이 마지막!")
print("   - 이 3개 시도 후 결과와 무관하게 문서화 단계로!")
print("   - 최적점 발견하든 못하든 여기서 멈춤!")

print("\n" + "=" * 70)
print("🎉 Grid Search v2 완료! 마지막 제출 시작하세요!")
print("=" * 70)

# ============================================
# CSV로 결과 요약 저장
# ============================================
results_df = pd.DataFrame(results)
results_csv_path = SUBMISSION_DIR / 'grid_search_v2_summary.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"\n📊 요약 저장: {results_csv_path}")

# ============================================
# 전체 가중치 추세 분석
# ============================================
print("\n" + "=" * 70)
print("📈 전체 가중치 추세 분석 (문서화용)")
print("=" * 70)

all_weights = [
    # (0.85, 0.15, "미제출"),
    # (0.82, 0.18, "미제출"),
    (0.80, 0.20, "16.5065m"),
    (0.78, 0.22, "16.5049m"),
    (0.75, 0.25, "16.5029m"),
    (0.70, 0.30, "16.5003m"),
    (0.65, 0.35, "제출 예정"),
    (0.60, 0.40, "제출 예정"),
    (0.55, 0.45, "제출 예정"),
]

print("\n   Phase 5  Phase 6    결과")
print("   ────────────────────────────")
for w5, w6, result in all_weights:
    marker = "⭐" if "16.500" in result else ""
    print(f"    {w5:.2f}  /  {w6:.2f}  →  {result} {marker}")

print("\n   💡 이 데이터로 완벽한 그래프 작성 가능!")
print("   💡 선형 관계 vs 최적점 존재 여부 확인")

print("\n" + "=" * 70)