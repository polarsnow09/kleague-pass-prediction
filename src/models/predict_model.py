"""
Test 데이터 예측 및 제출 파일 생성
K리그 패스 좌표 예측
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.features.build_feature import build_baseline_features, add_previous_action_features

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_DIR = PROJECT_ROOT / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'submissions'

OUTPUT_DIR.mkdir(exist_ok=True)


class PassPredictor:
    """패스 좌표 예측기"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: 학습된 모델 파일 경로
        """
        import pickle
        
        print(f"📦 모델 로딩: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_x = model_data['model_x']
        self.model_y = model_data['model_y']
        self.feature_cols = model_data['feature_cols']
        
        print(f"✅ 모델 로드 완료")
        print(f"  - 피처 수: {len(self.feature_cols)}")
        print(f"  - 피처 목록: {self.feature_cols[:5]}...")
    
    def load_test_episode(self, csv_path: Path) -> pd.DataFrame:
        """
        Test episode CSV 로드 및 전처리
        
        Args:
            csv_path: episode CSV 파일 경로
            
        Returns:
            전처리된 데이터프레임
        """
        # 파일 존재 확인
        if not csv_path.exists():
            raise FileNotFoundError(f"파일 없음: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # end_x, end_y 추가 (전처리용 임시값)
        if 'end_x' not in df.columns:
            df['end_x'] = df['start_x']
        if 'end_y' not in df.columns:
            df['end_y'] = df['start_y']
        
        # game_episode 추가
        df['game_episode'] = 'temp'
        
        # is_home 추가 (없으면 홈팀 가정)
        if 'is_home' not in df.columns:
            df['is_home'] = 1
        
        return df
    
    def preprocess_episode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Episode 데이터 피처 생성"""
        # Phase 1 피처
        df = build_baseline_features(df)
        
        # Phase 2 피처 (시계열)
        df = add_previous_action_features(df)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모델 입력 피처 준비
        
        Args:
            df: 전처리된 데이터프레임
            
        Returns:
            모델 입력용 피처 데이터프레임
        """
        df = df.copy()
        
        # 범주형 인코딩 (항상 수행)
        zone_x_map = {'defensive': 0, 'midfield': 1, 'attacking': 2}
        zone_y_map = {'left': 0, 'center': 1, 'right': 2}
        zone_combined_map = {
            'defensive_left': 0, 'defensive_center': 1, 'defensive_right': 2,
            'midfield_left': 3, 'midfield_center': 4, 'midfield_right': 5,
            'attacking_left': 6, 'attacking_center': 7, 'attacking_right': 8
        }
        
        # zone_x가 문자열(category)이면 인코딩
        if 'zone_x' in df.columns:
            # astype(str)로 변환 후 매핑
            df['zone_x_encoded'] = df['zone_x'].astype(str).map(zone_x_map)
            df['zone_y_encoded'] = df['zone_y'].astype(str).map(zone_y_map)
            df['zone_combined_encoded'] = df['zone_combined'].astype(str).map(zone_combined_map)
        else:
            print("  ⚠️  경고: zone_x 컬럼이 없습니다!")
            # 기본값 (중원-중앙)
            df['zone_x_encoded'] = 1
            df['zone_y_encoded'] = 1
            df['zone_combined_encoded'] = 4
        
        # 필요한 피처만 선택
        available_features = []
        missing_features = []
        
        for feat in self.feature_cols:
            if feat in df.columns:
                available_features.append(feat)
            else:
                missing_features.append(feat)
                df[feat] = 0  # 누락된 피처는 0으로 채움
        
        if missing_features:
            print(f"  ⚠️  누락된 피처 ({len(missing_features)}개)를 0으로 대체: {missing_features[:3]}...")
        
        X = df[self.feature_cols].copy()
        
        # 결측치 처리
        X = X.fillna(0)
        
        return X
    
    def predict(self, X: pd.DataFrame) -> tuple:
        """좌표 예측"""
        pred_x = self.model_x.predict(X)
        pred_y = self.model_y.predict(X)
        
        return pred_x, pred_y
    
    def predict_episode(self, csv_path: Path, debug: bool = False) -> tuple:
        """
        Episode의 최종 패스 좌표 예측
        
        Args:
            csv_path: episode CSV 파일 경로
            debug: 디버깅 출력 여부
            
        Returns:
            (end_x, end_y) 튜플
        """
        # 데이터 로드
        df = self.load_test_episode(csv_path)
        
        if debug:
            print(f"\n  🔍 {csv_path.name}")
            print(f"    - 원본 행 수: {len(df)}")
        
        # 전처리
        df = self.preprocess_episode(df)
        
        if debug:
            print(f"    - 전처리 후: {len(df.columns)}개 컬럼")
        
        # 마지막 행만 사용 (최종 패스)
        last_row = df.iloc[[-1]].copy()
        
        # 피처 준비
        X = self.prepare_features(last_row)
        
        if debug:
            print(f"    - 피처 준비 완료: {X.shape}")
        
        # 예측
        pred_x, pred_y = self.predict(X)
        
        if debug:
            print(f"    - 예측: ({pred_x[0]:.2f}, {pred_y[0]:.2f})")
        
        return pred_x[0], pred_y[0]

def create_submission(
    test_csv: Path,
    model_path: Path,
    output_filename: str = 'submission_v2.csv',
    debug_first_n: int = 5  # 처음 N개만 디버깅 출력
) -> None:
    """
    제출 파일 생성
    
    Args:
        test_csv: test.csv 파일 경로
        model_path: 학습된 모델 파일 경로
        output_filename: 출력 파일명
    """
    print("="*60)
    print("🎯 K리그 패스 좌표 예측 - 제출 파일 생성")
    print("="*60)
    
    # Test 데이터 로드
    print(f"\n📂 Test 데이터 로딩: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"  - 예측 대상: {len(test_df)}개 episode")
    
    # 예측기 초기화
    predictor = PassPredictor(str(model_path))
    
    # 예측 수행
    print(f"\n🔮 예측 시작...")
    predictions = []
    errors = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="예측 중"):
        game_episode = row['game_episode']
        
        # 경로 수정
        relative_path = row['path']
        if relative_path.startswith('./'):
            relative_path = relative_path[2:]
        
        csv_path = DATA_DIR / 'raw' / relative_path
        
        try:
            # 디버깅 출력 (처음 N개만)
            debug = (idx < debug_first_n)
            
            # Episode 예측
            end_x, end_y = predictor.predict_episode(csv_path, debug=debug)
            
            predictions.append({
                'game_episode': game_episode,
                'end_x': end_x,
                'end_y': end_y
            })
            
        except Exception as e:
            print(f"\n⚠️  오류 ({game_episode}): {e}")
            predictions.append({
                'game_episode': game_episode,
                'end_x': 52.5,
                'end_y': 34.0
            })
    
    # 오류 요약
    if errors:
        print(f"\n⚠️  파일을 찾을 수 없는 경우: {len(errors)}개")
        print(f"  - 처음 3개: {errors[:3]}")
    
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
    
    # 성공률
    success_rate = (len(test_df) - len(errors)) / len(test_df) * 100
    print(f"\n성공률: {success_rate:.1f}% ({len(test_df) - len(errors)}/{len(test_df)})")
    
    print("\n" + "="*60)
    print("🎊 제출 파일 생성 완료!")
    print("="*60)
    print(f"\n다음 단계:")
    print(f"1. {output_path} 파일 확인")
    print(f"2. 공모전 사이트에 제출")
    print(f"3. Public LB 점수 확인")

if __name__ == '__main__':
    # 설정
    TEST_CSV = DATA_DIR / 'raw' / 'test.csv'
    MODEL_PATH = MODEL_DIR / 'baseline_model_v2_temporal.pkl'
    
    # 파일 존재 확인
    if not TEST_CSV.exists():
        print(f"❌ Test 데이터가 없습니다: {TEST_CSV}")
        exit(1)
    
    if not MODEL_PATH.exists():
        print(f"❌ 모델 파일이 없습니다: {MODEL_PATH}")
        print(f"사용 가능한 모델:")
        for model_file in MODEL_DIR.glob('*.pkl'):
            print(f"  - {model_file.name}")
        exit(1)
    
    # 제출 파일 생성
    create_submission(
        test_csv=str(TEST_CSV),
        model_path=str(MODEL_PATH),
        output_filename='submission_v2_temporal.csv'
    )