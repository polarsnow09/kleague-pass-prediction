"""
Phase 6 피처 생성: 안전한 버전 (디버깅 강화)

수정 내용:
1. dtype 체크 및 안전한 변환
2. 결측치/NaN 처리
3. 디버깅 메시지 추가
"""

import pandas as pd
import numpy as np
from pathlib import Path

def build_phase6_features(df):
    """
    Phase 6 피처 생성 (안전한 버전)
    """
    
    print("\n" + "=" * 60)
    print("Phase 6 피처 생성 시작 (안전 모드)")
    print("=" * 60)
    
    # =================================================================
    # 전략 1: 구역별 특화 피처
    # =================================================================
    print("\n🎯 전략 1: 구역별 특화 피처")
    
    # 1-1. 수비 구역 여부
    df['is_defensive_zone'] = (df['start_x'] < 35).astype(int)
    
    # 1-2. 수비 구역 불확실성
    df['defensive_uncertainty'] = (
        df['is_defensive_zone'] * (1 - df['in_final_third'])
    )
    
    # 1-3. 구역별 선수 스타일 차이
    df['player_style_in_defense'] = np.where(
        df['start_x'] < 35,
        df['player_avg_pass_distance'] / (df['dist_to_target_goal'] + 1),
        0
    )
    
    # 1-4. 중앙 수비 구역
    df['is_defensive_center'] = (
        (df['start_x'] < 35) & 
        (df['start_y'] >= 20) & 
        (df['start_y'] <= 48)
    ).astype(int)
    
    # 1-5. 구역별 압박 효과 (⭐⭐⭐ 초간단 안전 버전)
    print("  🔍 디버깅: 구역 피처 생성 중...")
    
    # is_under_pressure 피처 확인
    if 'is_under_pressure' not in df.columns:
        print("  ⚠️ is_under_pressure 피처 없음 → 기본값 0 생성")
        df['is_under_pressure'] = 0
    
    # ⭐⭐⭐ zone_x 완전 무시하고 start_x로 새로 생성 (가장 안전)
    print("  📍 start_x 기반 구역 생성 (기존 zone_x 무시)")
    
    # start_x 기반 3구역 분류 (단순하게!)
    conditions = [
        df['start_x'] < 35,              # 수비 구역
        (df['start_x'] >= 35) & (df['start_x'] < 70),  # 중원
        df['start_x'] >= 70              # 공격 구역
    ]
    choices = [0, 1, 2]
    df['zone_x_encoded'] = np.select(conditions, choices, default=1).astype(np.int32)
    
    print(f"  ✅ zone_x_encoded 생성 완료 (dtype: {df['zone_x_encoded'].dtype})")
    print(f"     분포: {dict(pd.Series(df['zone_x_encoded']).value_counts())}")
    
    # 최종 상호작용 피처 생성
    df['pressure_zone_interaction'] = (
        df['is_under_pressure'].astype(int) * df['zone_x_encoded']
    ).astype(np.int32)
    
    print(f"  ✅ pressure_zone_interaction 생성 완료")
    
    print(f"  ✅ 5개 피처 생성 완료")
    
    # =================================================================
    # 전략 2: 최종 구역 미진입 타겟팅
    # =================================================================
    print("\n🎯 전략 2: 최종 구역 미진입 타겟팅")
    
    df['attack_failure_risk'] = (
        (1 - df['in_final_third']) * 
        (df['dist_to_target_goal'] / 105)
    )
    
    df['stuck_in_midfield'] = (
        (df['start_x'] >= 35) & 
        (df['start_x'] <= 70) & 
        (df['in_final_third'] == 0)
    ).astype(int)
    
    df['buildup_style'] = np.where(
        df['in_final_third'] == 0,
        df['pass_count_in_episode'] / (df['time_since_prev'] + 1),
        0
    )
    
    if 'prev_direction_x' in df.columns:
        df['attack_momentum'] = np.where(
            df['in_final_third'] == 0,
            df['prev_direction_x'] / (df['time_since_prev'] + 0.1),
            0
        )
    else:
        df['attack_momentum'] = 0
    
    print(f"  ✅ 4개 피처 생성 완료")
    
    # =================================================================
    # 전략 3: 측면 vs 중앙 차별화
    # =================================================================
    print("\n🎯 전략 3: 측면 vs 중앙 차별화")
    
    df['central_uncertainty'] = np.where(
        (df['start_y'] >= 20) & (df['start_y'] <= 48),
        df['touchline_proximity'] / 34,
        0
    )
    
    df['wing_attack_pattern'] = (
        ((df['start_y'] < 20) | (df['start_y'] > 48)) & 
        (df['start_x'] > 70)
    ).astype(int)
    
    df['cross_likelihood'] = np.where(
        ((df['start_y'] < 15) | (df['start_y'] > 53)) & (df['start_x'] > 70),
        1 - (df['start_y'] - 34)**2 / 34**2,
        0
    )
    
    if 'team_attack_style' in df.columns:
        df['wing_central_balance'] = (
            df['team_attack_style'] * 
            ((df['start_y'] < 20) | (df['start_y'] > 48)).astype(int)
        )
    else:
        df['wing_central_balance'] = 0
    
    print(f"  ✅ 4개 피처 생성 완료")
    
    # =================================================================
    # 전략 4: 득점 상황별 전술 변화
    # =================================================================
    print("\n🎯 전략 4: 득점 상황별 전술 변화")
    
    df['leading_defensive'] = np.where(
        df['score_diff'] > 0,
        (1 - df['in_final_third']) * df['time_since_prev'],
        0
    )
    
    df['losing_aggressive'] = np.where(
        df['score_diff'] < 0,
        df['in_final_third'] * (1 / (df['time_since_prev'] + 0.1)),
        0
    )
    
    df['endgame_pressure'] = (
        df['match_period_normalized'] * 
        np.abs(df['score_diff']) * 
        (1 - df['in_final_third'])
    )
    
    print(f"  ✅ 3개 피처 생성 완료")
    
    # =================================================================
    # 전략 5: 극단값 특수 처리
    # =================================================================
    print("\n🎯 전략 5: 극단값 특수 처리")
    
    df['near_boundary'] = (
        (df['start_x'] < 5) | (df['start_x'] > 100) |
        (df['start_y'] < 5) | (df['start_y'] > 63)
    ).astype(int)
    
    df['extreme_pass'] = (
        (df['player_avg_pass_distance'] > 30) | 
        (df['prev_action_distance'] > 40)
    ).astype(int)
    
    df['abnormal_situation'] = (
        df['near_boundary'] | 
        df['extreme_pass'] |
        (df['touchline_proximity'] > 30)
    ).astype(int)
    
    print(f"  ✅ 3개 피처 생성 완료")
    
    # =================================================================
    # 추가 피처: 상호작용
    # =================================================================
    print("\n🎯 보너스: 피처 상호작용")
    
    df['zone_final_interaction'] = (
        df['is_defensive_zone'] * (1 - df['in_final_third'])
    )
    
    df['wing_pressure_interaction'] = (
        df['wing_attack_pattern'] * df['is_under_pressure']
    )
    
    df['player_zone_interaction'] = (
        df['player_avg_pass_distance'] * df['is_defensive_zone']
    )
    
    print(f"  ✅ 3개 피처 생성 완료")
    
    # =================================================================
    # 최종 검증
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 6 피처 검증")
    print("=" * 60)
    
    phase6_features = [
        'is_defensive_zone', 'defensive_uncertainty', 
        'player_style_in_defense', 'is_defensive_center',
        'pressure_zone_interaction',
        'attack_failure_risk', 'stuck_in_midfield', 
        'buildup_style', 'attack_momentum',
        'central_uncertainty', 'wing_attack_pattern',
        'cross_likelihood', 'wing_central_balance',
        'leading_defensive', 'losing_aggressive', 'endgame_pressure',
        'near_boundary', 'extreme_pass', 'abnormal_situation',
        'zone_final_interaction', 'wing_pressure_interaction',
        'player_zone_interaction'
    ]
    
    phase6_features = [f for f in phase6_features if f in df.columns]
    
    print(f"\n✅ 총 {len(phase6_features)}개 Phase 6 피처 생성")
    
    # 결측치 확인
    missing = df[phase6_features].isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ 결측치 발견:")
        print(missing[missing > 0])
        # 결측치를 0으로 채움
        df[phase6_features] = df[phase6_features].fillna(0)
        print(f"✅ 결측치를 0으로 대체")
    else:
        print(f"✅ 결측치: 없음")
    
    # Inf 확인
    inf_count = np.isinf(df[phase6_features].select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"\n⚠️ Inf 값 발견: {inf_count}개")
        df[phase6_features] = df[phase6_features].replace([np.inf, -np.inf], 0)
        print(f"✅ Inf 값을 0으로 대체")
    else:
        print(f"✅ Inf 값: 없음")
    
    # dtype 검증
    print(f"\n🔍 dtype 검증:")
    for feat in ['zone_x_encoded', 'pressure_zone_interaction', 'is_under_pressure']:
        if feat in df.columns:
            print(f"  {feat:30s}: {df[feat].dtype}")
    
    print("\n" + "=" * 60)
    print("✅ Phase 6 피처 생성 완료!")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    
    DATA_DIR = Path('data/processed')
    
    print("=" * 60)
    print("Phase 6 피처 생성 스크립트 (안전 모드)")
    print("=" * 60)
    
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv(DATA_DIR / 'train_final_passes_v4.csv')
    print(f"✅ Shape: {df.shape}")
    
    df = build_phase6_features(df)
    
    output_path = DATA_DIR / 'train_final_passes_v6.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 저장 완료: {output_path}")
    print(f"   Shape: {df.shape}")
