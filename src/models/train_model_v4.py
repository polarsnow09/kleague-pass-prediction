"""
베이스라인 모델 학습 - Phase 4 버전
K리그 패스 좌표 예측 - XGBoost 멀티 타겟 회귀
Phase 4: 도메인 특화 피처 (선수/팀 통계, 경기 흐름)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle
from typing import Tuple, List

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'


class PassCoordinateModel:
    """패스 좌표 예측 모델"""
    
    def __init__(self):
        self.model_x = None  # end_x 예측 모델
        self.model_y = None  # end_y 예측 모델
        self.feature_cols = []
        
    def prepare_features(self, df: pd.DataFrame, use_temporal: bool = True, 
                        use_phase3: bool = True, use_phase4: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        학습용 피처 준비
        
        Args:
            df: 데이터프레임
            use_temporal: Phase 2 시계열 피처 사용 여부
            use_phase3: Phase 3 고급 피처 사용 여부
            use_phase4: Phase 4 도메인 피처 사용 여부
        
        Returns:
            X, y_x, y_y
        """
        df = df.copy()
        
        print(f"\n🔧 피처 준비 시작")
        print(f"  - Phase 2: {'ON' if use_temporal else 'OFF'}")
        print(f"  - Phase 3: {'ON' if use_phase3 else 'OFF'}")
        print(f"  - Phase 4: {'ON' if use_phase4 else 'OFF'}")
        print(f"  - 원본 컬럼 수: {len(df.columns)}")
        
        # 범주형 피처 인코딩
        zone_x_map = {'defensive': 0, 'midfield': 1, 'attacking': 2}
        zone_y_map = {'left': 0, 'center': 1, 'right': 2}
        zone_combined_map = {
            'defensive_left': 0, 'defensive_center': 1, 'defensive_right': 2,
            'midfield_left': 3, 'midfield_center': 4, 'midfield_right': 5,
            'attacking_left': 6, 'attacking_center': 7, 'attacking_right': 8
        }
        
        if 'zone_x' in df.columns and df['zone_x'].dtype == 'object':
            df['zone_x_encoded'] = df['zone_x'].map(zone_x_map)
            df['zone_y_encoded'] = df['zone_y'].map(zone_y_map)
            df['zone_combined_encoded'] = df['zone_combined'].map(zone_combined_map)
        
        # Phase 1 피처 (기본)
        phase1_features = [
            'start_x', 'start_y',
            'dist_to_target_goal',
            'zone_x_encoded', 'zone_y_encoded', 'zone_combined_encoded',
            'in_penalty_box', 'in_final_third',
        ]
        
        self.feature_cols = [f for f in phase1_features if f in df.columns]
        print(f"  - Phase 1 피처: {len(self.feature_cols)}개")
        
        # Phase 2 피처 (시계열)
        if use_temporal:
            phase2_features = [
                'prev_end_x', 'prev_end_y',
                'prev_action_distance',
                'time_since_prev',
                'prev_direction_x', 'prev_direction_y',
                'pass_count_in_episode'
            ]
            
            temporal_added = []
            for feat in phase2_features:
                if feat in df.columns:
                    self.feature_cols.append(feat)
                    temporal_added.append(feat)
            
            print(f"  - Phase 2 피처: {len(temporal_added)}개")
        
        # Phase 3 피처 (선별 버전)
        if use_phase3:
            phase3_features = [
                # 속도 (핵심 2개만)
                'pass_velocity',          # 중요도 0.8%
                'avg_episode_velocity',   # Episode 평균
                
                # 공간 (핵심 2개만)
                'touchline_proximity',    # 중요도 0.9%, 측면 압박
                'episode_x_range',        # X축 활용도
                
                # 패턴 (핵심 1개만)
                'is_under_pressure',      # 중요도 0.6%, 압박 상황
                
                # 롤링 평균 (대표 1개만)
                'rolling_mean_distance_3', # 최근 패스 거리 추세
            ]
            
            phase3_added = []
            for feat in phase3_features:
                if feat in df.columns:
                    self.feature_cols.append(feat)
                    phase3_added.append(feat)
            
            print(f"  - Phase 3 피처: {len(phase3_added)}개")
            if not phase3_added:
                print(f"    ⚠️  Phase 3 피처를 찾을 수 없습니다!")
        
        # Phase 4 피처 (도메인 특화) ⭐ NEW!
        if use_phase4:
            phase4_features = [
                # 선수 통계 (4개)
                'player_avg_pass_distance',  # 선수 평균 패스 거리
                'player_forward_ratio',      # 전진 패스 비율
                'player_success_rate',       # 패스 성공률
                'player_pass_count',         # 패스 횟수 (경험치)
                
                # 팀 통계 (2개)
                'team_avg_pass_distance',    # 팀 평균 패스 거리
                'team_attack_style',         # 측면/중앙 선호도
                
                # 경기 흐름 (3개)
                'score_diff',                # 득점차
                'match_period_normalized',   # 경기 진행률
                'is_late_game',              # 후반 75분 이후
            ]
            
            phase4_added = []
            for feat in phase4_features:
                if feat in df.columns:
                    self.feature_cols.append(feat)
                    phase4_added.append(feat)
            
            print(f"  - Phase 4 피처: {len(phase4_added)}개")
            if not phase4_added:
                print(f"    ⚠️  Phase 4 피처를 찾을 수 없습니다!")
        
        print(f"  - 최종 피처 수: {len(self.feature_cols)}개")
        
        # NaN 확인
        X = df[self.feature_cols].copy()
        nan_counts = X.isna().sum()
        if nan_counts.sum() > 0:
            print(f"\n⚠️  결측치 발견:")
            print(nan_counts[nan_counts > 0])
            print(f"  → 0으로 대체합니다.")
            X = X.fillna(0)
        
        y_x = df['end_x'].copy()
        y_y = df['end_y'].copy()
        
        return X, y_x, y_y
    
    def train(self, df: pd.DataFrame, n_folds: int = 5) -> dict:
        """
        Cross-validation으로 모델 학습
        
        Args:
            df: 학습 데이터
            n_folds: CV fold 수
            
        Returns:
            학습 결과 딕셔너리
        """
        print("=" * 60)
        print("🚀 XGBoost 모델 학습 시작 (Phase 4)")
        print("=" * 60)
        
        # 피처 준비 (Phase 4 포함!)
        X, y_x, y_y = self.prepare_features(df, use_temporal=True, use_phase3=True, use_phase4=True)
        
        print(f"\n📊 데이터 정보:")
        print(f"  - 샘플 수: {len(X):,}")
        print(f"  - 피처 수: {len(self.feature_cols)}")
        
        # XGBoost 파라미터 (v3와 동일)
        params = {
            'max_depth': 8,           # 6 → 8
            'learning_rate': 0.05,    # 0.1 → 0.05  
            'n_estimators': 300,      # 100 → 300
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        
        print(f"\n⚙️  하이퍼파라미터:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        # Cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_scores_x = []
        cv_scores_y = []
        cv_scores_total = []
        
        print(f"\n🔄 {n_folds}-Fold Cross Validation:")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_x_train, y_x_val = y_x.iloc[train_idx], y_x.iloc[val_idx]
            y_y_train, y_y_val = y_y.iloc[train_idx], y_y.iloc[val_idx]
            
            # end_x 모델 학습
            model_x = xgb.XGBRegressor(**params, early_stopping_rounds=10)
            model_x.fit(X_train, y_x_train, 
                       eval_set=[(X_val, y_x_val)],
                       verbose=False)
            
            # end_y 모델 학습
            model_y = xgb.XGBRegressor(**params, early_stopping_rounds=10)
            model_y.fit(X_train, y_y_train,
                       eval_set=[(X_val, y_y_val)],
                       verbose=False)
            
            # 예측
            pred_x = model_x.predict(X_val)
            pred_y = model_y.predict(X_val)
            
            # RMSE 계산
            rmse_x = np.sqrt(mean_squared_error(y_x_val, pred_x))
            rmse_y = np.sqrt(mean_squared_error(y_y_val, pred_y))
            
            # 유클리드 거리 RMSE
            euclidean_errors = np.sqrt((y_x_val - pred_x)**2 + (y_y_val - pred_y)**2)
            rmse_total = np.sqrt(np.mean(euclidean_errors**2))
            
            cv_scores_x.append(rmse_x)
            cv_scores_y.append(rmse_y)
            cv_scores_total.append(rmse_total)
            
            print(f"  Fold {fold}: RMSE_X={rmse_x:.4f}m, RMSE_Y={rmse_y:.4f}m, Total={rmse_total:.4f}m")
        
        print(f"\n📈 Cross-validation 결과:")
        print(f"  - RMSE_X: {np.mean(cv_scores_x):.4f} ± {np.std(cv_scores_x):.4f}m")
        print(f"  - RMSE_Y: {np.mean(cv_scores_y):.4f} ± {np.std(cv_scores_y):.4f}m")
        print(f"  - RMSE_Total: {np.mean(cv_scores_total):.4f} ± {np.std(cv_scores_total):.4f}m")
        
        # 전체 데이터로 최종 모델 학습
        print(f"\n🔧 전체 데이터로 최종 모델 학습 중...")
        
        self.model_x = xgb.XGBRegressor(**params)
        self.model_x.fit(X, y_x, verbose=False)
        
        self.model_y = xgb.XGBRegressor(**params)
        self.model_y.fit(X, y_y, verbose=False)
        
        print(f"✅ 학습 완료!")
        
        # 피처 중요도
        feature_importance_x = pd.DataFrame({
            'feature': self.feature_cols,
            'importance_x': self.model_x.feature_importances_,
            'importance_y': self.model_y.feature_importances_
        }).sort_values('importance_x', ascending=False)
        
        print(f"\n📊 피처 중요도 (Top 15):")
        print(feature_importance_x.head(15).to_string(index=False))
        
        # Phase 4 피처만 따로 확인
        phase4_features = [
            'player_avg_pass_distance', 'player_forward_ratio', 
            'player_success_rate', 'player_pass_count',
            'team_avg_pass_distance', 'team_attack_style',
            'score_diff', 'match_period_normalized', 'is_late_game'
        ]
        phase4_importance = feature_importance_x[
            feature_importance_x['feature'].isin(phase4_features)
        ]
        
        if len(phase4_importance) > 0:
            print(f"\n📊 Phase 4 피처 중요도:")
            print(phase4_importance.to_string(index=False))
        
        # 결과 저장
        results = {
            'cv_rmse_x': cv_scores_x,
            'cv_rmse_y': cv_scores_y,
            'cv_rmse_total': cv_scores_total,
            'mean_rmse_x': np.mean(cv_scores_x),
            'mean_rmse_y': np.mean(cv_scores_y),
            'mean_rmse_total': np.mean(cv_scores_total),
            'feature_importance': feature_importance_x,
            'feature_cols': self.feature_cols
        }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        좌표 예측
        
        Args:
            X: 피처 데이터프레임
            
        Returns:
            예측 결과 (end_x, end_y)
        """
        if self.model_x is None or self.model_y is None:
            raise ValueError("모델이 학습되지 않았습니다!")
        
        X_features = X[self.feature_cols]
        
        pred_x = self.model_x.predict(X_features)
        pred_y = self.model_y.predict(X_features)
        
        result = pd.DataFrame({
            'end_x': pred_x,
            'end_y': pred_y
        })
        
        return result
    
    def save(self, filename: str = 'baseline_model.pkl'):
        """모델 저장"""
        MODEL_DIR.mkdir(exist_ok=True)
        filepath = MODEL_DIR / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_x': self.model_x,
                'model_y': self.model_y,
                'feature_cols': self.feature_cols
            }, f)
        
        print(f"\n💾 모델 저장 완료: {filepath}")
    
    def load(self, filename: str = 'baseline_model.pkl'):
        """모델 로드"""
        filepath = MODEL_DIR / filename
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model_x = data['model_x']
        self.model_y = data['model_y']
        self.feature_cols = data['feature_cols']
        
        print(f"✅ 모델 로드 완료: {filepath}")


if __name__ == '__main__':
    print("🎯 K리그 패스 좌표 예측 - v4 모델 (도메인 특화 피처)\n")
    
    # 데이터 로드 (v4로 변경!)
    train_path = DATA_DIR / 'train_final_passes_v4.csv'
    
    if not train_path.exists():
        print(f"❌ v4 데이터 파일이 없습니다: {train_path}")
        print("먼저 Phase 4 피처를 생성해주세요!")
        exit(1)
    
    df = pd.read_csv(train_path)
    print(f"✅ 데이터 로드 완료: {len(df):,}개 샘플")
    print(f"  - 컬럼 수: {len(df.columns)}")
    
    # Phase 4 피처 확인
    phase4_features = [
        'player_avg_pass_distance', 'team_avg_pass_distance', 
        'score_diff', 'match_period_normalized'
    ]
    has_phase4 = all(feat in df.columns for feat in phase4_features)
    
    print(f"\nPhase 4 피처 존재: {'✓' if has_phase4 else '✗'}")
    
    if not has_phase4:
        print("⚠️  Phase 4 피처가 없습니다!")
        print("v3 데이터를 사용 중인 것 같습니다. v4 데이터를 사용해주세요.")
        missing = [f for f in phase4_features if f not in df.columns]
        print(f"누락된 피처: {missing}")
        exit(1)
    
    # 모델 학습
    model = PassCoordinateModel()
    results = model.train(df, n_folds=5)
    
    # 모델 저장 (v4로 변경!)
    model.save('baseline_model_v4.pkl')
    
    # 이전 버전과 비교
    print("\n" + "="*60)
    print("📊 성능 비교")
    print("="*60)
    print(f"Phase 2 (v2): CV 18.88m")
    print(f"Phase 3 (v3): CV 18.91m")
    print(f"Phase 4 (v4): CV {results['mean_rmse_total']:.2f}m")
    
    if results['mean_rmse_total'] < 18.91:
        improvement = 18.91 - results['mean_rmse_total']
        print(f"✅ v3 대비 개선: -{improvement:.2f}m ({improvement/18.91*100:.1f}%)")
    else:
        print(f"⚠️  v3 대비 악화: +{results['mean_rmse_total'] - 18.91:.2f}m")
    
    print("\n" + "="*60)
    print("🎊 XGBoost Phase 4 학습 완료!")
    print("="*60)
