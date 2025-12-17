"""
앙상블 예측 (XGBoost + LightGBM + CatBoost)
K리그 패스 좌표 예측 - Phase 3 버전
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
from src.features.advanced_features import build_phase3_features  # ⭐ Phase 3 추가!

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'submissions'

OUTPUT_DIR.mkdir(exist_ok=True)


class EnsemblePredictor:
    """앙상블 예측기 (여러 모델 평균)"""
    
    def __init__(self, model_paths: list, weights: list = None):
        """
        Args:
            model_paths: 모델 파일 경로 리스트
            weights: 각 모델의 가중치 (None이면 균등 평균)
        """
        self.models = []
        self.feature_cols = None
        
        print(f"📦 모델 로딩 ({len(model_paths)}개)")
        
        for i, path in enumerate(model_paths, 1):
            print(f"  {i}. {path.name}")
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models.append({
                'model_x': model_data['model_x'],
                'model_y': model_data['model_y'],
                'feature_cols': model_data.get('feature_cols') or model_data.get('features'),  # 호환성
                'name': path.stem
            })
            
            # 첫 번째 모델의 피처 컬럼 사용
            if self.feature_cols is None:
                self.feature_cols = model_data.get('feature_cols') or model_data.get('features')
        
        # 가중치 설정
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
    
    def preprocess_episode(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 생성 (Phase 1 + 2 + 3)"""
        # Phase 1: 기본 피처
        df = build_baseline_features(df)
        
        # Phase 2: 시계열 피처
        df = add_previous_action_features(df)
        
        # Phase 3: 고급 시계열 피처 ⭐ 새로 추가!
        df = build_phase3_features(df)
        
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
    model_paths: list,
    weights: list = None,
    output_filename: str = 'submission_ensemble_v3.csv',
    debug_first_n: int = 5
) -> None:
    """앙상블 제출 파일 생성"""
    print("=" * 60)
    print("🎯 K리그 패스 좌표 예측 - 앙상블 제출 (v3)")
    print("=" * 60)
    
    # Test 데이터 로드
    print(f"\n📂 Test 데이터 로딩: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"  - 예측 대상: {len(test_df)}개 episode")
    
    # 앙상블 예측기 초기화
    predictor = EnsemblePredictor(model_paths, weights)
    
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
    
    # v3 모델들 ⭐ 경로 변경!
    MODEL_PATHS = [
        MODEL_DIR / 'baseline_model_v3.pkl',  # XGBoost v3
        MODEL_DIR / 'lgb_model_v3_optuna.pkl',       # LightGBM v3
        MODEL_DIR / 'catboost_model_v3.pkl',  # CatBoost v3
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
    
    # 앙상블 제출 파일 생성
    weights = [0.2, 0.4, 0.4]  # 검증된 최적 가중치
    output_name = 'submission_ensemble_v3_optuna.csv'
    
    create_ensemble_submission(
        test_csv=TEST_CSV,
        model_paths=MODEL_PATHS,
        weights=weights, 
        output_filename=output_name
    )