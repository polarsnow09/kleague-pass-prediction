"""
앙상블 예측 (XGBoost + LightGBM + CatBoost)
K리그 패스 좌표 예측 - Phase 4 버전
Phase 4: 도메인 특화 피처 (선수/팀 통계, 경기 흐름)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import pickle

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.features.build_feature import build_baseline_features, add_previous_action_features
from src.features.advanced_features import build_phase3_features

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'submissions'

OUTPUT_DIR.mkdir(exist_ok=True)


class Phase4Statistics:
    """Phase 4 통계 계산기 (선수/팀 통계)"""
    
    def __init__(self, train_csv: Path, match_info_csv: Path):
        """
        Args:
            train_csv: 학습 데이터 경로
            match_info_csv: 경기 정보 경로
        """
        print("\n📊 Phase 4 통계 준비 중...")
        
        # Train 데이터 로드
        train = pd.read_csv(train_csv)
        passes = train[train['type_name'] == 'Pass'].copy()
        
        # 패스 거리
        passes['pass_distance'] = np.sqrt(
            (passes['end_x'] - passes['start_x'])**2 + 
            (passes['end_y'] - passes['start_y'])**2
        )
        
        # 전진 패스
        passes['is_forward'] = np.where(
            passes['is_home'],
            passes['end_x'] - passes['start_x'],
            passes['start_x'] - passes['end_x']
        ) > 0
        
        # 패스 성공
        passes['is_success'] = (passes['result_name'] == 'Successful').astype(int)
        
        # 측면 패스
        passes['is_wide'] = ((passes['start_y'] < 20) | (passes['start_y'] > 48)).astype(int)
        
        # 선수별 통계
        self.player_stats = passes.groupby('player_id').agg({
            'pass_distance': 'mean',
            'is_forward': 'mean',
            'is_success': 'mean',
            'player_id': 'count'
        }).rename(columns={'player_id': 'pass_count'}).to_dict('index')
        
        # 팀별 통계
        self.team_stats = passes.groupby('team_id').agg({
            'pass_distance': 'mean',
            'is_wide': 'mean'
        }).rename(columns={'is_wide': 'attack_style'}).to_dict('index')
        
        # 전체 평균 (신규 선수/팀용)
        self.global_player = {
            'pass_distance': passes['pass_distance'].mean(),
            'is_forward': passes['is_forward'].mean(),
            'is_success': passes['is_success'].mean(),
            'pass_count': 50
        }
        
        self.global_team = {
            'pass_distance': passes['pass_distance'].mean(),
            'attack_style': passes['is_wide'].mean()
        }
        
        # 경기 정보 (득점)
        self.match_info = pd.read_csv(match_info_csv)
        
        print(f"  ✅ 선수 통계: {len(self.player_stats):,}명")
        print(f"  ✅ 팀 통계: {len(self.team_stats):,}팀")
        print(f"  ✅ 경기 정보: {len(self.match_info):,}경기")
    
    def get_player_stats(self, player_id: int) -> dict:
        """선수 통계 가져오기"""
        if player_id in self.player_stats:
            stats = self.player_stats[player_id]
            return {
                'player_avg_pass_distance': stats['pass_distance'],
                'player_forward_ratio': stats['is_forward'],
                'player_success_rate': stats['is_success'],
                'player_pass_count': stats['pass_count']
            }
        else:
            return {
                'player_avg_pass_distance': self.global_player['pass_distance'],
                'player_forward_ratio': self.global_player['is_forward'],
                'player_success_rate': self.global_player['is_success'],
                'player_pass_count': self.global_player['pass_count']
            }
    
    def get_team_stats(self, team_id: int) -> dict:
        """팀 통계 가져오기"""
        if team_id in self.team_stats:
            stats = self.team_stats[team_id]
            return {
                'team_avg_pass_distance': stats['pass_distance'],
                'team_attack_style': stats['attack_style']
            }
        else:
            return {
                'team_avg_pass_distance': self.global_team['pass_distance'],
                'team_attack_style': self.global_team['attack_style']
            }
    
    def get_match_stats(self, game_id: int, team_id: int, is_home: bool, time_seconds: float) -> dict:
        """경기 흐름 통계 가져오기"""
        match = self.match_info[self.match_info['game_id'] == game_id]
        
        if len(match) == 0:
            return {
                'score_diff': 0,
                'match_period_normalized': time_seconds / 5400,
                'is_late_game': int(time_seconds >= 4050)
            }
        
        match = match.iloc[0]
        
        # 득점차
        if is_home:
            score_diff = match['home_score'] - match['away_score']
        else:
            score_diff = match['away_score'] - match['home_score']
        
        return {
            'score_diff': score_diff,
            'match_period_normalized': time_seconds / 5400,
            'is_late_game': int(time_seconds >= 4050)
        }


class EnsemblePredictor:
    """앙상블 예측기 (Phase 4 버전)"""
    
    def __init__(self, model_paths: list, weights: list = None, phase4_stats: Phase4Statistics = None):
        """
        Args:
            model_paths: 모델 파일 경로 리스트
            weights: 각 모델의 가중치
            phase4_stats: Phase 4 통계 계산기
        """
        self.models = []
        self.feature_cols = None
        self.phase4_stats = phase4_stats
        
        print(f"\n📦 모델 로딩 ({len(model_paths)}개)")
        
        for i, path in enumerate(model_paths, 1):
            print(f"  {i}. {path.name}")
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models.append({
                'model_x': model_data['model_x'],
                'model_y': model_data['model_y'],
                'feature_cols': model_data.get('feature_cols') or model_data.get('features'),
                'name': path.stem
            })
            
            if self.feature_cols is None:
                self.feature_cols = model_data.get('feature_cols') or model_data.get('features')
        
        # 가중치
        if weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            self.weights = weights
        
        print(f"✅ 앙상블 준비 완료")
        print(f"  - 모델 수: {len(self.models)}")
        print(f"  - 가중치: {self.weights}")
        print(f"  - 피처 수: {len(self.feature_cols)}")
    
    def load_test_episode(self, csv_path: Path) -> pd.DataFrame:
        """Test episode CSV 로드"""
        if not csv_path.exists():
            raise FileNotFoundError(f"파일 없음: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        if 'end_x' not in df.columns:
            df['end_x'] = df['start_x']
        if 'end_y' not in df.columns:
            df['end_y'] = df['start_y']
        
        df['game_episode'] = 'temp'
        
        if 'is_home' not in df.columns:
            df['is_home'] = 1
        
        return df
    
    def add_phase4_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Phase 4 피처 추가"""
        if self.phase4_stats is None:
            # Phase 4 통계 없으면 기본값
            df['player_avg_pass_distance'] = 16.84
            df['player_forward_ratio'] = 0.50
            df['player_success_rate'] = 0.86
            df['player_pass_count'] = 50
            df['team_avg_pass_distance'] = 16.84
            df['team_attack_style'] = 0.60
            df['score_diff'] = 0
            df['match_period_normalized'] = df['time_seconds'] / 5400
            df['is_late_game'] = (df['time_seconds'] >= 4050).astype(int)
            return df
        
        # 최종 Pass의 정보 추출
        last_pass = df[df['type_name'] == 'Pass'].iloc[-1]
        
        player_id = int(last_pass['player_id'])
        team_id = int(last_pass['team_id'])
        game_id = int(last_pass['game_id'])
        is_home = bool(last_pass['is_home'])
        time_seconds = float(last_pass['time_seconds'])
        
        # 선수 통계
        player_stats = self.phase4_stats.get_player_stats(player_id)
        for key, value in player_stats.items():
            df[key] = value
        
        # 팀 통계
        team_stats = self.phase4_stats.get_team_stats(team_id)
        for key, value in team_stats.items():
            df[key] = value
        
        # 경기 흐름
        match_stats = self.phase4_stats.get_match_stats(game_id, team_id, is_home, time_seconds)
        for key, value in match_stats.items():
            df[key] = value
        
        return df
    
    def preprocess_episode(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 생성 (Phase 1 + 2 + 3 + 4)"""
        # Phase 1
        df = build_baseline_features(df)
        
        # Phase 2
        df = add_previous_action_features(df)
        
        # Phase 3
        df = build_phase3_features(df)
        
        # Phase 4 ⭐ NEW!
        df = self.add_phase4_features(df)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모델 입력 피처 준비"""
        df = df.copy()
        
        # 범주형 인코딩
        zone_x_map = {'defensive': 0, 'midfield': 1, 'attacking': 2}
        zone_y_map = {'left': 0, 'center': 1, 'right': 2}
        zone_combined_map = {
            'defensive_left': 0, 'defensive_center': 1, 'defensive_right': 2,
            'midfield_left': 3, 'midfield_center': 4, 'midfield_right': 5,
            'attacking_left': 6, 'attacking_center': 7, 'attacking_right': 8
        }
        
        if 'zone_x' in df.columns:
            df['zone_x_encoded'] = df['zone_x'].astype(str).map(zone_x_map)
            df['zone_y_encoded'] = df['zone_y'].astype(str).map(zone_y_map)
            df['zone_combined_encoded'] = df['zone_combined'].astype(str).map(zone_combined_map)
        
        # 누락된 피처 처리
        for feat in self.feature_cols:
            if feat not in df.columns:
                df[feat] = 0
        
        X = df[self.feature_cols].copy()
        X = X.fillna(0)
        
        return X
    
    def predict(self, X: pd.DataFrame) -> tuple:
        """앙상블 예측"""
        pred_x_list = []
        pred_y_list = []
        
        for model_info, weight in zip(self.models, self.weights):
            # LightGBM 특수 처리
            if 'lgb' in model_info['name'].lower():
                if hasattr(model_info['model_x'], 'best_iteration'):
                    pred_x = model_info['model_x'].predict(X, num_iteration=model_info['model_x'].best_iteration)
                    pred_y = model_info['model_y'].predict(X, num_iteration=model_info['model_y'].best_iteration)
                else:
                    pred_x = model_info['model_x'].predict(X)
                    pred_y = model_info['model_y'].predict(X)
            else:
                pred_x = model_info['model_x'].predict(X)
                pred_y = model_info['model_y'].predict(X)
            
            pred_x_list.append(pred_x * weight)
            pred_y_list.append(pred_y * weight)
        
        # 가중 평균
        final_pred_x = np.sum(pred_x_list, axis=0)
        final_pred_y = np.sum(pred_y_list, axis=0)
        
        return final_pred_x, final_pred_y
    
    def predict_episode(self, csv_path: Path, debug: bool = False) -> tuple:
        """Episode 예측"""
        df = self.load_test_episode(csv_path)
        
        if debug:
            print(f"\n  🔍 {csv_path.name}")
            print(f"    - 원본 행 수: {len(df)}")
        
        df = self.preprocess_episode(df)
        
        if debug:
            print(f"    - 전처리 후: {len(df.columns)}개 컬럼")
            # Phase 4 피처 확인
            phase4_cols = ['player_avg_pass_distance', 'team_attack_style', 'score_diff']
            for col in phase4_cols:
                if col in df.columns:
                    print(f"    - {col}: {df[col].iloc[-1]:.2f}")
        
        # 최종 Pass만 선택
        pass_rows = df[df['type_name'] == 'Pass']
        if len(pass_rows) > 0:
            last_row = pass_rows.iloc[[-1]].copy()
        else:
            last_row = df.iloc[[-1]].copy()
        
        X = self.prepare_features(last_row)
        
        if debug:
            print(f"    - 피처 준비 완료: {X.shape}")
        
        pred_x, pred_y = self.predict(X)
        
        if debug:
            print(f"    - 예측: ({pred_x[0]:.2f}, {pred_y[0]:.2f})")
        
        return pred_x[0], pred_y[0]


def create_ensemble_submission(
    test_csv: Path,
    train_csv: Path,
    match_info_csv: Path,
    model_paths: list,
    weights: list = None,
    output_filename: str = 'submission_ensemble_v4.csv',
    debug_first_n: int = 5
) -> None:
    """앙상블 제출 파일 생성 (Phase 4)"""
    print("=" * 60)
    print("🎯 K리그 패스 좌표 예측 - 앙상블 제출 (v4)")
    print("=" * 60)
    
    # Phase 4 통계 준비
    phase4_stats = Phase4Statistics(train_csv, match_info_csv)
    
    # Test 데이터 로드
    print(f"\n📂 Test 데이터 로딩: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"  - 예측 대상: {len(test_df)}개 episode")
    
    # 앙상블 예측기 초기화
    predictor = EnsemblePredictor(model_paths, weights, phase4_stats)
    
    # 예측 수행
    print(f"\n🔮 예측 시작...")
    predictions = []
    errors = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="예측 중"):
        game_episode = row['game_episode']
        
        relative_path = row['path']
        if relative_path.startswith('./'):
            relative_path = relative_path[2:]
        
        csv_path = DATA_DIR / 'raw' / relative_path
        
        try:
            debug = (idx < debug_first_n)
            end_x, end_y = predictor.predict_episode(csv_path, debug=debug)
            
            predictions.append({
                'game_episode': game_episode,
                'end_x': end_x,
                'end_y': end_y
            })
            
        except FileNotFoundError:
            errors.append(str(csv_path))
            predictions.append({
                'game_episode': game_episode,
                'end_x': 52.5,
                'end_y': 34.0
            })
            
        except Exception as e:
            print(f"\n⚠️  오류 ({game_episode}): {e}")
            predictions.append({
                'game_episode': game_episode,
                'end_x': 52.5,
                'end_y': 34.0
            })
    
    if errors:
        print(f"\n⚠️  파일 없음: {len(errors)}개")
    
    # 제출 파일 생성
    print(f"\n💾 제출 파일 생성 중...")
    submission = pd.DataFrame(predictions)
    
    # 좌표 범위 체크
    submission['end_x'] = submission['end_x'].clip(0, 105)
    submission['end_y'] = submission['end_y'].clip(0, 68)
    
    # 저장
    output_path = OUTPUT_DIR / output_filename
    submission.to_csv(output_path, index=False)
    
    print(f"✅ 저장 완료: {output_path}")
    print(f"\n📊 예측 통계:")
    print(submission[['end_x', 'end_y']].describe())
    
    success_rate = (len(test_df) - len(errors)) / len(test_df) * 100
    print(f"\n성공률: {success_rate:.1f}% ({len(test_df) - len(errors)}/{len(test_df)})")
    
    print("\n" + "=" * 60)
    print("🎊 앙상블 제출 파일 생성 완료!")
    print("=" * 60)


if __name__ == '__main__':
    # 설정
    TEST_CSV = DATA_DIR / 'raw' / 'test.csv'
    TRAIN_CSV = DATA_DIR / 'raw' / 'train.csv'
    MATCH_INFO_CSV = DATA_DIR / 'raw' / 'match_info.csv'
    
    # v4 모델들 ⭐ Phase 4 버전!
    MODEL_PATHS = [
        MODEL_DIR / 'baseline_model_v4.pkl',  # XGBoost v4
        MODEL_DIR / 'lgb_model_v4.pkl',       # LightGBM v4
        MODEL_DIR / 'catboost_model_v4.pkl',  # CatBoost v4
    ]
    
    # 모델 존재 확인
    for path in MODEL_PATHS:
        if not path.exists():
            print(f"❌ 모델 파일이 없습니다: {path}")
            print(f"\n사용 가능한 모델:")
            for model_file in MODEL_DIR.glob('*.pkl'):
                print(f"  - {model_file.name}")
            exit(1)
    
    if not TEST_CSV.exists():
        print(f"❌ Test 데이터가 없습니다: {TEST_CSV}")
        exit(1)
    
    if not TRAIN_CSV.exists():
        print(f"❌ Train 데이터가 없습니다: {TRAIN_CSV}")
        exit(1)
    
    if not MATCH_INFO_CSV.exists():
        print(f"❌ match_info 데이터가 없습니다: {MATCH_INFO_CSV}")
        exit(1)
    
    # 앙상블 제출 파일 생성
    weights = [0.2, 0.4, 0.4]  # 검증된 최적 가중치
    output_name = 'submission_ensemble_v4.csv'
    
    create_ensemble_submission(
        test_csv=TEST_CSV,
        train_csv=TRAIN_CSV,
        match_info_csv=MATCH_INFO_CSV,
        model_paths=MODEL_PATHS,
        weights=weights,
        output_filename=output_name,
        debug_first_n=5  # 첫 5개만 디버그 출력
    )
