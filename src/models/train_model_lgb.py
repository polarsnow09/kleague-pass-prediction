"""
LightGBM 모델 학습
K리그 패스 좌표 예측
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle
from typing import Tuple

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'


class PassCoordinateModelLGB:
    """LightGBM 기반 패스 좌표 예측 모델"""
    
    def __init__(self):
        self.model_x = None
        self.model_y = None
        self.feature_cols = []
        
    def prepare_features(self, df: pd.DataFrame, use_temporal: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """학습용 피처 준비 (XGBoost와 동일)"""
        df = df.copy()
        
        print(f"\n🔧 피처 준비 시작 (시계열: {'ON' if use_temporal else 'OFF'})")
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
        
        # Phase 1 피처
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
            if temporal_added:
                print(f"    추가된 피처: {temporal_added}")
        
        print(f"  - 최종 피처 수: {len(self.feature_cols)}개")
        
        # NaN 처리
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
        """Cross-validation으로 모델 학습"""
        print("=" * 60)
        print("🚀 LightGBM 모델 학습 시작")
        print("=" * 60)
        
        # 피처 준비
        X, y_x, y_y = self.prepare_features(df)
        
        print(f"\n📊 데이터 정보:")
        print(f"  - 샘플 수: {len(X)}")
        print(f"  - 피처 수: {len(self.feature_cols)}")
        
        # LightGBM 파라미터
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
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
            
            # end_x 모델
            model_x = lgb.LGBMRegressor(**params)
            model_x.fit(
                X_train, y_x_train,
                eval_set=[(X_val, y_x_val)],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # end_y 모델
            model_y = lgb.LGBMRegressor(**params)
            model_y.fit(
                X_train, y_y_train,
                eval_set=[(X_val, y_y_val)],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
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
        
        self.model_x = lgb.LGBMRegressor(**params)
        self.model_x.fit(X, y_x)
        
        self.model_y = lgb.LGBMRegressor(**params)
        self.model_y.fit(X, y_y)
        
        print(f"✅ 학습 완료!")
        
        # 피처 중요도
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance_x': self.model_x.feature_importances_,
            'importance_y': self.model_y.feature_importances_
        }).sort_values('importance_x', ascending=False)
        
        print(f"\n📊 피처 중요도 (Top 5):")
        print(feature_importance.head())
        
        # 결과 저장
        results = {
            'cv_rmse_x': cv_scores_x,
            'cv_rmse_y': cv_scores_y,
            'cv_rmse_total': cv_scores_total,
            'mean_rmse_x': np.mean(cv_scores_x),
            'mean_rmse_y': np.mean(cv_scores_y),
            'mean_rmse_total': np.mean(cv_scores_total),
            'feature_importance': feature_importance,
            'feature_cols': self.feature_cols
        }
        
        return results
    
    def save(self, filename: str = 'lgb_model.pkl'):
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


if __name__ == '__main__':
    print("🎯 K리그 패스 좌표 예측 - LightGBM 모델\n")
    
    # 데이터 로드
    train_path = DATA_DIR / 'train_final_passes_v2.csv'
    
    if not train_path.exists():
        print(f"❌ 데이터 파일이 없습니다: {train_path}")
        exit(1)
    
    df = pd.read_csv(train_path)
    print(f"✅ 데이터 로드 완료: {len(df)}개 샘플\n")
    
    # 모델 학습
    model = PassCoordinateModelLGB()
    results = model.train(df, n_folds=5)
    
    # 모델 저장
    model.save('lgb_model_v1.pkl')
    
    print("\n" + "="*60)
    print("🎊 학습 완료!")
    print("="*60)