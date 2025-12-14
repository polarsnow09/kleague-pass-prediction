"""
Phase 3: 고급 시계열 피처
"""

import numpy as np
import pandas as pd
from typing import Dict


def add_rolling_features(df: pd.DataFrame, windows: list = [3, 5]) -> pd.DataFrame:
    """
    롤링 통계 피처
    
    Args:
        df: 데이터프레임 (episode별 정렬 필요)
        windows: 윈도우 크기 리스트
    """
    df = df.copy()
    
    for window in windows:
        # 거리 롤링
        df[f'rolling_mean_distance_{window}'] = df.groupby('game_episode')['prev_action_distance'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        df[f'rolling_std_distance_{window}'] = df.groupby('game_episode')['prev_action_distance'].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
        
        # X 방향 롤링
        df[f'rolling_mean_direction_x_{window}'] = df.groupby('game_episode')['prev_direction_x'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        
        # Y 방향 롤링
        df[f'rolling_mean_direction_y_{window}'] = df.groupby('game_episode')['prev_direction_y'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    return df


def add_cumulative_features(df: pd.DataFrame) -> pd.DataFrame:
    """누적 통계 피처 (수정됨)"""
    df = df.copy()
    
    # 누적 거리
    df['cumulative_distance'] = df.groupby('game_episode')['prev_action_distance'].cumsum()
    
    # 누적 전진 (X)
    df['cumulative_forward'] = df.groupby('game_episode')['prev_direction_x'].cumsum()
    
    # 누적 측면 (Y) - transform 사용
    df['cumulative_lateral'] = df.groupby('game_episode')['prev_direction_y'].transform(
        lambda x: x.abs().cumsum()
    )
    
    # 비율 (0으로 나누기 방지)
    df['forward_lateral_ratio'] = df['cumulative_forward'] / (df['cumulative_lateral'] + 1e-6)
    
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """속도 관련 피처"""
    df = df.copy()
    
    # 속도 (m/s) - 0으로 나누기 방지
    df['pass_velocity'] = df['prev_action_distance'] / (df['time_since_prev'] + 0.1)
    
    # Episode 평균 속도
    df['avg_episode_velocity'] = df.groupby('game_episode')['pass_velocity'].transform('mean')
    
    # 속도 변화율
    df['velocity_change'] = df.groupby('game_episode')['pass_velocity'].transform(
        lambda x: x.diff().fillna(0)
    )
    
    # 최근 3개 평균 속도
    df['recent_3_avg_velocity'] = df.groupby('game_episode')['pass_velocity'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """공간 활용 피처"""
    df = df.copy()
    
    # X 범위 (누적 최대 - 최소)
    df['episode_x_range'] = df.groupby('game_episode')['start_x'].transform(
        lambda x: x.expanding().max() - x.expanding().min()
    )
    
    # Y 범위 (누적 최대 - 최소)
    df['episode_y_range'] = df.groupby('game_episode')['start_y'].transform(
        lambda x: x.expanding().max() - x.expanding().min()
    )
    
    # 터치라인 근접도
    df['touchline_proximity'] = df['start_y'].apply(lambda y: min(y, 68-y))
    
    # 평균 터치라인 근접도
    df['avg_touchline_proximity'] = df.groupby('game_episode')['touchline_proximity'].transform('mean')
    
    return df


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """패턴 인식 피처"""
    df = df.copy()
    
    # 속도 피처가 있는지 확인
    if 'pass_velocity' in df.columns and 'prev_action_distance' in df.columns:
        # 빌드업 (느린 속도, 짧은 거리)
        df['is_buildup'] = (
            (df['pass_velocity'] < 5.0) & 
            (df['prev_action_distance'] < 15.0)
        ).astype(int)
        
        # 역습 (빠른 속도, 긴 거리)
        df['is_counter'] = (
            (df['pass_velocity'] > 15.0) & 
            (df['prev_action_distance'] > 20.0)
        ).astype(int)
        
        # 압박
        df['is_under_pressure'] = (
            (df['time_since_prev'] < 2.0) & 
            (df['prev_action_distance'] < 10.0)
        ).astype(int)
    
    return df


def build_phase3_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 3 전체 피처 생성
    
    Args:
        df: Phase 2까지 완료된 데이터프레임
    """
    df = df.copy()
    
    print("🔧 Phase 3 피처 생성 중...")
    
    # Episode별 정렬 (필수!)
    df = df.sort_values(['game_episode', 'time_seconds']).reset_index(drop=True)
    
    # 1. 롤링 피처
    print("  - 롤링 통계...")
    df = add_rolling_features(df, windows=[3, 5])
    print("     롤링 통계 end")
    
    # 2. 누적 피처
    print("  - 누적 통계...")
    df = add_cumulative_features(df)
    print("     누적 통계 end")
    
    # 3. 속도 피처
    print("  - 속도 분석...")
    df = add_velocity_features(df)
    print("     속도 분석 end")
    
    # 4. 공간 피처
    print("  - 공간 활용...")
    df = add_spatial_features(df)
    print("     공간 활용 end")
    
    # 5. 패턴 피처
    print("  - 패턴 인식...")
    df = add_pattern_features(df)
    print("     패턴 인식 end")
    
    print("✅ Phase 3 완료!")
    
    return df


if __name__ == '__main__':
    # 테스트
    print("Phase 3 피처 모듈 테스트")
    
    # 샘플 데이터
    sample = pd.DataFrame({
        'game_episode': ['ep1'] * 10,
        'time_seconds': range(10),
        'start_x': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        'start_y': [34, 30, 35, 32, 38, 34, 30, 35, 32, 34],
        'prev_action_distance': [5, 7, 6, 8, 7, 6, 9, 7, 8, 7],
        'prev_direction_x': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        'prev_direction_y': [-4, 5, -3, 6, -4, -4, 5, -3, 2, 0],
        'time_since_prev': [1, 1.5, 1.2, 2, 1.8, 1.5, 2.5, 1.8, 2, 1.5]
    })
    
    result = build_phase3_features(sample)
    
    print(f"\n생성된 피처 수: {len(result.columns) - len(sample.columns)}")
    print(f"총 피처: {len(result.columns)}")
    
    new_features = [col for col in result.columns if col not in sample.columns]
    print(f"\n새 피처 ({len(new_features)}개):")
    for feat in new_features:
        print(f"  - {feat}")
    
    # 샘플 값 확인
    print(f"\n샘플 값 (첫 3행):")
    print(result[new_features[:5]].head(3))