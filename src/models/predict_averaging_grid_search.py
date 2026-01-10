"""
Averaging Ensemble - Grid Search: Phase 5 + Phase 6 가중치 최적화

현재 최고: (0.8, 0.2) = 16.5065m

목표: 최적 가중치 발견
전략: 5개 가중치 조합 자동 생성
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
print("🔍 Averaging Ensemble - Grid Search: Phase 5 + Phase 6")
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
# Grid Search 가중치 정의
# ============================================
print("\n🎯 Grid Search 가중치:")

# 현재 최고: (0.8, 0.2)
# 주변 탐색
weight_combinations = [
    (0.85, 0.15, "더 보수적 (Phase 5 중시)"),
    (0.82, 0.18, "미세 조정 1"),
    (0.78, 0.22, "미세 조정 2"),
    (0.75, 0.25, "약간 공격적 (Phase 6 증가)"),
    (0.70, 0.30, "더 공격적 (Phase 6 대폭 증가)"),
]

for i, (w5, w6, desc) in enumerate(weight_combinations, 1):
    print(f"   {i}. Phase 5 ({w5:.2f}) + Phase 6 ({w6:.2f}) - {desc}")

print("\n" + "=" * 70)

# ============================================
# 각 가중치별로 예측 생성
# ============================================
print("\n🔄 가중치별 예측 생성 중...\n")

results = []

for i, (w5, w6, desc) in enumerate(weight_combinations, 1):
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
    
    print(f"   평균: end_x={avg_x:.2f}, end_y={avg_y:.2f}")
    print(f"   Phase 5 대비 평균 변화: end_x={change_x:.2f}m, end_y={change_y:.2f}m")
    
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
        'change_y': change_y
    })

# ============================================
# 결과 요약
# ============================================
print("=" * 70)
print("✅ Grid Search 완료!")
print("=" * 70)

print("\n📋 생성된 제출 파일 목록:\n")
for r in results:
    print(f"   {r['num']}. {r['filename']}")
    print(f"      가중치: Phase 5 ({r['w5']:.2f}) + Phase 6 ({r['w6']:.2f})")
    print(f"      설명: {r['desc']}")
    print(f"      예측: end_x={r['avg_x']:.2f}m, end_y={r['avg_y']:.2f}m")
    print(f"      변화: end_x={r['change_x']:.2f}m, end_y={r['change_y']:.2f}m")
    print()

print("=" * 70)
print("🎯 다음 단계:")
print("=" * 70)
print("\n1️⃣ 5개 파일을 순서대로 제출")
print("   - submission_averaging_grid_1_w585_w615.csv")
print("   - submission_averaging_grid_2_w582_w618.csv")
print("   - submission_averaging_grid_3_w578_w622.csv")
print("   - submission_averaging_grid_4_w575_w625.csv")
print("   - submission_averaging_grid_5_w570_w630.csv")

print("\n2️⃣ LB 스코어 기록")
print("   - 각 가중치별 점수 확인")
print("   - 최고 점수 선택")

print("\n3️⃣ 결과 분석")
print("   - 최적 가중치 발견 시: 프로젝트 정리")
print("   - 개선 없으면: 옵션 B (5-Model Stacking) 고려")

print("\n💡 예상 결과:")
print("   - 현재 최고: 0.8/0.2 = 16.5065m")
print("   - 목표: 16.48-16.50m")
print("   - 가능성: 조합 3-4가 유력 (0.78/0.22 or 0.75/0.25)")

print("\n" + "=" * 70)
print("🎉 Grid Search 완료! 제출 시작하세요!")
print("=" * 70)

# ============================================
# CSV로 결과 요약 저장
# ============================================
results_df = pd.DataFrame(results)
results_csv_path = SUBMISSION_DIR / 'grid_search_summary.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"\n📊 요약 저장: {results_csv_path}")