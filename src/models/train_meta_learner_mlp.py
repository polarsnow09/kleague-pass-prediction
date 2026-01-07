"""
MLP Meta-Learner 학습 스크립트

OOF 예측을 사용하여 Neural Network 기반 Meta-Learner를 학습합니다.

사용법:
    python src/models/train_meta_learner_mlp.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 경로 설정
DATA_DIR = Path('data/processed')
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("MLP Meta-Learner 학습 시작")
print("=" * 60)

# ===================================================================
# STEP 1: 데이터 로드 및 전처리
# ===================================================================
print("\n📂 OOF 데이터 로드 중...")
oof_df = pd.read_csv(DATA_DIR / 'oof_predictions.csv')
print(f"✅ Shape: {oof_df.shape}")

# Meta-Features 및 타겟 분리
meta_features = [
    'xgb_pred_x', 'xgb_pred_y',
    'lgb_pred_x', 'lgb_pred_y',
    'cat_pred_x', 'cat_pred_y'
]

X_meta = oof_df[meta_features].values
y_true_x = oof_df['true_x'].values
y_true_y = oof_df['true_y'].values

print(f"\n✅ Meta-Features: {X_meta.shape}")
print(f"   - 샘플: {len(X_meta):,}개")
print(f"   - 피처: {len(meta_features)}개")

# ===================================================================
# STEP 2: 데이터 정규화 (Neural Network용)
# ===================================================================
print("\n🔧 데이터 정규화 중...")
scaler = StandardScaler()
X_meta_scaled = scaler.fit_transform(X_meta)

print("✅ 정규화 완료")

# ===================================================================
# STEP 3: PyTorch Dataset 생성
# ===================================================================
print("\n🔧 PyTorch Dataset 생성 중...")

# NumPy → Torch Tensor
X_tensor = torch.FloatTensor(X_meta_scaled)
y_x_tensor = torch.FloatTensor(y_true_x).unsqueeze(1)
y_y_tensor = torch.FloatTensor(y_true_y).unsqueeze(1)

# Train/Val 분리 (80:20)
from sklearn.model_selection import train_test_split

X_train, X_val, y_x_train, y_x_val = train_test_split(
    X_tensor, y_x_tensor, test_size=0.2, random_state=42
)
_, _, y_y_train, y_y_val = train_test_split(
    X_tensor, y_y_tensor, test_size=0.2, random_state=42
)

print(f"✅ Train: {len(X_train):,}개")
print(f"✅ Val:   {len(X_val):,}개")

# DataLoader 생성
batch_size = 256
train_dataset_x = TensorDataset(X_train, y_x_train)
train_loader_x = DataLoader(train_dataset_x, batch_size=batch_size, shuffle=True)

train_dataset_y = TensorDataset(X_train, y_y_train)
train_loader_y = DataLoader(train_dataset_y, batch_size=batch_size, shuffle=True)

# ===================================================================
# STEP 4: MLP 모델 정의
# ===================================================================
print("\n🏗️ MLP 모델 정의 중...")

class MLPMetaLearner(nn.Module):
    """
    Simple 2-layer MLP for Meta-Learning
    
    Architecture:
        Input (6) → Hidden (32) → ReLU → Dropout → Hidden (16) → ReLU → Output (1)
    """
    def __init__(self, input_dim=6, hidden_dim1=32, hidden_dim2=16, dropout=0.2):
        super(MLPMetaLearner, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 모델 초기화
mlp_x = MLPMetaLearner(input_dim=6)
mlp_y = MLPMetaLearner(input_dim=6)

print("✅ MLP 구조:")
print(mlp_x)

# ===================================================================
# STEP 5: 학습 설정
# ===================================================================
print("\n⚙️ 학습 설정...")

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer_x = optim.Adam(mlp_x.parameters(), lr=0.001, weight_decay=1e-4)
optimizer_y = optim.Adam(mlp_y.parameters(), lr=0.001, weight_decay=1e-4)

# Learning Rate Scheduler
scheduler_x = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_x, mode='min', factor=0.5, patience=10
)
scheduler_y = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_y, mode='min', factor=0.5, patience=10
)

print("✅ Optimizer: Adam (lr=0.001)")
print("✅ Loss: MSE")
print("✅ Scheduler: ReduceLROnPlateau")

# ===================================================================
# STEP 6: 학습 함수
# ===================================================================
def train_epoch(model, loader, criterion, optimizer):
    """1 epoch 학습"""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        
        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(X_batch)
    
    return total_loss / len(loader.dataset)

def validate(model, X_val, y_val):
    """검증"""
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val)
        mse = criterion(y_pred, y_val).item()
    return np.sqrt(mse)  # RMSE

# ===================================================================
# STEP 7: end_x 모델 학습
# ===================================================================
print("\n" + "=" * 60)
print("🎓 end_x MLP 학습 시작")
print("=" * 60)

n_epochs = 100
best_val_rmse_x = float('inf')
patience_counter = 0
early_stop_patience = 20

for epoch in range(n_epochs):
    # Train
    train_loss = train_epoch(mlp_x, train_loader_x, criterion, optimizer_x)
    
    # Validate
    val_rmse = validate(mlp_x, X_val, y_x_val)
    
    # Scheduler step
    scheduler_x.step(val_rmse)
    
    # Early stopping
    if val_rmse < best_val_rmse_x:
        best_val_rmse_x = val_rmse
        patience_counter = 0
        # 최고 모델 저장
        best_mlp_x_state = mlp_x.state_dict()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
              f"Val RMSE: {val_rmse:.4f}m | Best: {best_val_rmse_x:.4f}m")
    
    if patience_counter >= early_stop_patience:
        print(f"\n⚠️ Early stopping at epoch {epoch+1}")
        break

# 최고 모델 복원
mlp_x.load_state_dict(best_mlp_x_state)
print(f"\n✅ end_x 학습 완료! Best Val RMSE: {best_val_rmse_x:.4f}m")

# ===================================================================
# STEP 8: end_y 모델 학습
# ===================================================================
print("\n" + "=" * 60)
print("🎓 end_y MLP 학습 시작")
print("=" * 60)

best_val_rmse_y = float('inf')
patience_counter = 0

for epoch in range(n_epochs):
    # Train
    train_loss = train_epoch(mlp_y, train_loader_y, criterion, optimizer_y)
    
    # Validate
    val_rmse = validate(mlp_y, X_val, y_y_val)
    
    # Scheduler step
    scheduler_y.step(val_rmse)
    
    # Early stopping
    if val_rmse < best_val_rmse_y:
        best_val_rmse_y = val_rmse
        patience_counter = 0
        best_mlp_y_state = mlp_y.state_dict()
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | "
              f"Val RMSE: {val_rmse:.4f}m | Best: {best_val_rmse_y:.4f}m")
    
    if patience_counter >= early_stop_patience:
        print(f"\n⚠️ Early stopping at epoch {epoch+1}")
        break

# 최고 모델 복원
mlp_y.load_state_dict(best_mlp_y_state)
print(f"\n✅ end_y 학습 완료! Best Val RMSE: {best_val_rmse_y:.4f}m")

# ===================================================================
# STEP 9: 전체 데이터 평가
# ===================================================================
print("\n" + "=" * 60)
print("📊 전체 데이터 평가")
print("=" * 60)

mlp_x.eval()
mlp_y.eval()

with torch.no_grad():
    X_full_tensor = torch.FloatTensor(X_meta_scaled)
    
    pred_x = mlp_x(X_full_tensor).squeeze().numpy()
    pred_y = mlp_y(X_full_tensor).squeeze().numpy()

rmse_x = np.sqrt(mean_squared_error(y_true_x, pred_x))
rmse_y = np.sqrt(mean_squared_error(y_true_y, pred_y))
rmse_total = np.sqrt((rmse_x**2 + rmse_y**2) / 2)

print(f"\n✅ MLP Meta-Learner 성능:")
print(f"   - end_x RMSE: {rmse_x:.4f}m")
print(f"   - end_y RMSE: {rmse_y:.4f}m")
print(f"   - Total RMSE: {rmse_total:.4f}m")

# ===================================================================
# STEP 10: 모델 저장
# ===================================================================
print("\n💾 모델 저장 중...")

# PyTorch 모델 + Scaler를 함께 저장
mlp_x_package = {
    'model_state': mlp_x.state_dict(),
    'scaler': scaler,
    'architecture': {
        'input_dim': 6,
        'hidden_dim1': 32,
        'hidden_dim2': 16,
        'dropout': 0.2
    }
}

mlp_y_package = {
    'model_state': mlp_y.state_dict(),
    'scaler': scaler,
    'architecture': {
        'input_dim': 6,
        'hidden_dim1': 32,
        'hidden_dim2': 16,
        'dropout': 0.2
    }
}

mlp_path_x = MODEL_DIR / 'meta_mlp_x.pkl'
mlp_path_y = MODEL_DIR / 'meta_mlp_y.pkl'

with open(mlp_path_x, 'wb') as f:
    pickle.dump(mlp_x_package, f)
with open(mlp_path_y, 'wb') as f:
    pickle.dump(mlp_y_package, f)

print(f"✅ 저장 완료:")
print(f"   - {mlp_path_x}")
print(f"   - {mlp_path_y}")

# ===================================================================
# STEP 11: 기존 Meta-Learner와 비교
# ===================================================================
print("\n" + "=" * 60)
print("📊 Meta-Learner 비교")
print("=" * 60)

# Ridge/LightGBM 결과 불러오기 (train_meta_learner.py 실행 필요)
try:
    # Ridge
    with open(MODEL_DIR / 'meta_ridge_x.pkl', 'rb') as f:
        ridge_x = pickle.load(f)
    with open(MODEL_DIR / 'meta_ridge_y.pkl', 'rb') as f:
        ridge_y = pickle.load(f)
    
    ridge_pred_x = ridge_x.predict(X_meta)
    ridge_pred_y = ridge_y.predict(X_meta)
    ridge_rmse_x = np.sqrt(mean_squared_error(y_true_x, ridge_pred_x))
    ridge_rmse_y = np.sqrt(mean_squared_error(y_true_y, ridge_pred_y))
    ridge_rmse = np.sqrt((ridge_rmse_x**2 + ridge_rmse_y**2) / 2)
    
    print(f"\nRidge     : {ridge_rmse:.4f}m")
except FileNotFoundError:
    ridge_rmse = None
    print(f"\nRidge     : (모델 없음)")

try:
    # LightGBM
    with open(MODEL_DIR / 'meta_lgb_x.pkl', 'rb') as f:
        lgb_x = pickle.load(f)
    with open(MODEL_DIR / 'meta_lgb_y.pkl', 'rb') as f:
        lgb_y = pickle.load(f)
    
    lgb_pred_x = lgb_x.predict(X_meta)
    lgb_pred_y = lgb_y.predict(X_meta)
    lgb_rmse_x = np.sqrt(mean_squared_error(y_true_x, lgb_pred_x))
    lgb_rmse_y = np.sqrt(mean_squared_error(y_true_y, lgb_pred_y))
    lgb_rmse = np.sqrt((lgb_rmse_x**2 + lgb_rmse_y**2) / 2)
    
    print(f"LightGBM  : {lgb_rmse:.4f}m")
except FileNotFoundError:
    lgb_rmse = None
    print(f"LightGBM  : (모델 없음)")

print(f"MLP       : {rmse_total:.4f}m ← 새로 추가!")

# 최고 성능 찾기
results = {
    'Ridge': ridge_rmse,
    'LightGBM': lgb_rmse,
    'MLP': rmse_total
}
results = {k: v for k, v in results.items() if v is not None}

if results:
    best_model = min(results, key=results.get)
    best_score = results[best_model]
    
    print(f"\n🏆 최고 성능: {best_model} ({best_score:.4f}m)")
    
    # MLP 개선폭 계산
    if lgb_rmse is not None:
        improvement = lgb_rmse - rmse_total
        if improvement > 0:
            print(f"\n✨ MLP가 LightGBM 대비 {improvement:.4f}m 개선! (+{improvement/lgb_rmse*100:.2f}%)")
        else:
            print(f"\n⚠️ MLP가 LightGBM 대비 {abs(improvement):.4f}m 나쁨 ({improvement/lgb_rmse*100:.2f}%)")

# ===================================================================
print("\n" + "=" * 60)
print("🎉 MLP Meta-Learner 학습 완료!")
print("=" * 60)

print(f"\n✅ 저장된 모델:")
print(f"   - MLP: meta_mlp_x.pkl, meta_mlp_y.pkl")

print(f"\n다음 단계:")
print(f"1. Stacking 예측 (MLP): python src/models/predict_stacking_mlp.py")
print(f"2. 결과 비교 및 분석")
