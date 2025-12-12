"""
피처 엔지니어링 모듈
K리그 패스 좌표 예측 - Phase 1 베이스라인 피처
"""

import numpy as np
import pandas as pd
from typing import Tuple

# 경기장 상수
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
PENALTY_BOX_LENGTH = 16.5
CENTER_X = FIELD_LENGTH / 2


def add_distance_to_goal(df: pd.DataFrame) -> pd.DataFrame:
    """
    골대까지의 거리 계산 (is_home에 따라 자동 처리)
    
    Args:
        df: 데이터프레임 (is_home 컬럼 필요)
    
    Returns:
        dist_to_target_goal 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    
    # 홈팀/원정팀별 타겟 골대 위치
    # 홈팀: 오른쪽 골대 (105, 34)
    # 원정팀: 왼쪽 골대 (0, 34)
    target_goal_x = np.where(df['is_home'] == 1, FIELD_LENGTH, 0)
    target_goal_y = FIELD_WIDTH / 2
    
    # 유클리드 거리
    df['dist_to_target_goal'] = np.sqrt(
        (df['start_x'] - target_goal_x) ** 2 + 
        (df['start_y'] - target_goal_y) ** 2
    )
    
    return df


def add_field_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    경기장을 9개 구역으로 분류
    
    X축 3구역: 수비(0-35), 중원(35-70), 공격(70-105)
    Y축 3구역: 좌측(0-22.67), 중앙(22.67-45.33), 우측(45.33-68)
    
    Returns:
        zone_x, zone_y, zone_combined 컬럼 추가
    """
    df = df.copy()
    
    # X축 구역
    df['zone_x'] = pd.cut(
        df['start_x'],
        bins=[0, 35, 70, 105],
        labels=['defensive', 'midfield', 'attacking'],
        include_lowest=True
    )
    
    # Y축 구역
    df['zone_y'] = pd.cut(
        df['start_y'],
        bins=[0, 22.67, 45.33, 68],
        labels=['left', 'center', 'right'],
        include_lowest=True
    )
    
    # 조합 구역
    df['zone_combined'] = df['zone_x'].astype(str) + '_' + df['zone_y'].astype(str)
    
    return df


def add_penalty_box_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    페널티 박스 진입 여부
    
    Returns:
        in_penalty_box, in_final_third 컬럼 추가
    """
    df = df.copy()
    
    # 페널티 박스 (양쪽)
    df['in_penalty_box'] = (
        (df['start_x'] < PENALTY_BOX_LENGTH) |  # 왼쪽 박스
        (df['start_x'] > FIELD_LENGTH - PENALTY_BOX_LENGTH)  # 오른쪽 박스
    ).astype(int)
    
    # 최종 3구역 (공격 방향 기준)
    df['in_final_third'] = (df['start_x'] > 70).astype(int)
    
    return df


def add_episode_progress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Episode 내 진행률 계산
    
    Returns:
        episode_progress 컬럼 추가 (0~1 사이 값)
    """
    df = df.copy()
    
    # 각 episode 내에서의 순서
    df['action_number'] = df.groupby('game_episode').cumcount() + 1
    df['episode_total_actions'] = df.groupby('game_episode')['game_episode'].transform('count')
    
    # 진행률
    df['episode_progress'] = df['action_number'] / df['episode_total_actions']
    
    return df


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1 베이스라인 피처 전체 생성
    
    Args:
        df: 원본 데이터프레임 (start_x, start_y, is_home 필요)
    
    Returns:
        모든 베이스라인 피처가 추가된 데이터프레임
    """
    df = df.copy()
    
    # is_home 체크
    if 'is_home' not in df.columns:
        print("⚠️  Warning: 'is_home' 컬럼이 없습니다. 홈팀으로 가정합니다.")
        df['is_home'] = 1
    
    # 모든 피처 생성
    df = add_distance_to_goal(df) 
    df = add_field_zones(df)
    df = add_penalty_box_feature(df)
    df = add_episode_progress(df)
    
    return df


if __name__ == '__main__':
    # 테스트 코드
    print("피처 엔지니어링 모듈 테스트")
    
    # 샘플 데이터
    sample = pd.DataFrame({
        'start_x': [50, 85, 15],
        'start_y': [34, 10, 60],
        'is_home': [1, 1, 0],
        'game_episode': ['ep1', 'ep1', 'ep2']
    })
    
    result = build_baseline_features(sample)
    print("\n생성된 피처:")
    print(result.columns.tolist())
    print("\n샘플 결과:")
    print(result[['start_x', 'start_y', 'is_home', 'dist_to_target_goal', 
                  'zone_combined', 'in_final_third', 'episode_progress']])