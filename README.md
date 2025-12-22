# K리그 최종 패스 좌표 예측 🎯⚽

> K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발

## 📌 프로젝트 개요
- **목표**: K리그의 실제 경기 데이터를 기반으로, 단편적인 이벤트의 나열을 넘어 특정 상황의 맥락을 AI가 학습하고, 이어지는 패스가 도달할 최적의 위치를 예측하는 것. 나아가 이를 통해 데이터 기반의 선수 평가 및 전술 분석에 대한 새로운 가능성을 발굴하고자 함.
- **기간**: 2025.12.10 ~ 2026.01.12
- **역할**: 데이터 분석, 모델링, AI 도구 활용 전략 수립
- **성과**: (추후 작성)

## 🛠️ 기술 스택
- **언어**: Python 3.10
- **라이브러리**: pandas, numpy, scikit-learn
- **모델**: XGBoost, LightGBM, CatBoost
- **기법**: 시계열 피처 엔지니어링, 앙상블 모델링, K-Fold CV

## 📂 프로젝트 구조
```
kleague-pass-prediction/
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 데이터
│       ├── train_final_passes_v2.csv  # Phase 2
│       └── train_final_passes_v3.csv  # Phase 3
├── models/                     # 학습된 모델 (.pkl)
│   ├── baseline_model_v3.pkl
│   ├── lgb_model_v3.pkl
│   └── catboost_model_v3.pkl
├── src/
│   ├── features/               # 피처 생성 모듈
│   │   ├── build_feature.py    # Phase 1, 2
│   │   └── advanced_features.py # Phase 3
│   └── models/                 # 모델 학습/예측
│       ├── train_model.py
│       └── predict_ensemble.py
├── reports/
│   ├── figures/                # 시각화
│   └── prompts/                # AI 협업 로그
├── submissions/                # 제출 파일
└── README.md
```

## 🚀 실행 방법

### 환경 설정
```bash
pip install -r requirements.txt
```

### 모델 학습
```bash
# 개별 모델 학습 (Phase 3)
python src/models/train_model.py          # XGBoost
python src/models/train_model_lgb.py      # LightGBM
python src/models/train_model_lgb_optuna.py    # LightGBM Optuna 튜닝
python src/models/train_model_catboost.py # CatBoost
```

### 앙상블 예측
```bash
python src/models/predict_ensemble.py
# 출력: submissions/submission_ensemble_v3_tuned.csv
```

## 📊 주요 결과

### 성능 개선 과정
| 단계 | CV RMSE | LB Score | 개선 | 비고 |
|------|---------|----------|------|------|
| Phase 1 (Baseline) | 20.36m | - | - | 위치 피처만 |
| Phase 2 (Temporal) | 18.88m | 17.23m | -7.3% | 시계열 피처 추가 |
| Phase 2 + 2-model | - | 17.13m | -0.6% | XGB + LGB |
| Phase 2 + 3-model | - | 17.01m | -0.7% | + CatBoost |
| Phase 3 (Advanced) | 18.85m | 16.98m | -0.2% | 고급 시계열 |
| **Phase 3 + 튜닝** | **18.83m** | **16.9724** | **최고** | 수동 튜닝 |
| Phase 3 + Optuna | 18.76m | 16.9776 | -0.003% ❌ | 개별↑ 앙상블↓ |

### 개별 모델 성능
| 모델 | Phase 2 (v2) | Phase 3 (v3) | v3 수동 튜닝 | v3 Optuna | 변화 |
|------|--------------|--------------|-------------|-----------|------|
| **XGBoost** | 18.88m | 18.91m | **18.87m** | ❌ | - 0.04m |
| **LightGBM** | 18.81m | 18.82m | 18.81m | **18.76m** | - 0.05m |
| **CatBoost** | 18.97m | **18.82m** | 18.87m | ❌ | 기존 유지 |
| **평균** | 18.89m | 18.85m | 18.85m | **18.83m** | -0.02m |
**핵심 발견**: CatBosst 이외의 나머지 모델들은 튜닝으로 성능 향상을 보임

### 공모전 제출
- **Public LB**: 16.9724 RMSE
- **순위**: 374/687 (상위 약 54%)
- **일반화 성능**: 베이스라인 대비 약 -16.6% 개선 

### 피처 개발
#### Phase 1: 위치 기반 피처 (8개)
```python
- start_x, start_y              # 시작 좌표
- dist_to_target_goal           # 골대까지 거리
- zone_x, zone_y, zone_combined # 구역 분류 (9 zones)
- in_penalty_box                # 페널티 박스 여부
- in_final_third                # 최종 3구역 여부
```
**효과**: CV 20.36m (베이스라인)

#### Phase 2: 시계열 피처 (7개)
```python
- prev_end_x, prev_end_y        # 이전 액션 종료 위치
- prev_action_distance          # 이전 액션과의 거리
- time_since_prev               # 이전 액션과의 시간 간격
- prev_direction_x, prev_direction_y # 공격 방향
- pass_count_in_episode         # Episode 내 패스 카운트
```
**효과**: CV 18.88m (**-7.3%** 개선)
**핵심 인사이트**:
- Episode 맥락이 좌표 예측에 결정적
- 공격 방향성이 최종 패스 위치 결정

#### Phase 3: 고급 시계열 피처 (선별 6개)
```python
# 속도 관련
- pass_velocity                 # 패스 속도 (m/s)
- avg_episode_velocity          # Episode 평균 속도

# 공간 활용
- touchline_proximity           # 터치라인 근접도
- episode_x_range               # X축 활용 범위

# 패턴 인식
- is_under_pressure             # 압박 상황 여부
- rolling_mean_distance_3       # 최근 3개 평균 거리
```
**효과**: CV 18.85m (**-0.2%** 추가 개선)
**개발 과정**:
1. 23개 고급 피처 생성 → 성능 악화 (18.99m)
2. 피처 중요도 분석 → 6개 선별
3. 재학습 → 성능 회복 (18.85m)
**교훈**: "피처 품질 > 피처 수량"

## 📈 피처 중요도 분석
### XGBoost (zone 중심)
```
1. zone_x_encoded (80%)      ← 압도적!
2. start_x (10%)
3. in_penalty_box (1.4%)
```

### LightGBM (균형잡힌 분포)
```
1. start_x (36.5M)
2. zone_x_encoded (8.1M)
3. time_since_prev (2.7M)
4. touchline_proximity (1.7M) ← Phase 3
```

### CatBoost (공간 피처 활용)
```
1. start_x (28.3)
2. time_since_prev (8.0)
3. touchline_proximity (6.7)  ← Phase 3 효과!
```
**결론**: 각 모델이 다른 패턴 학습 → 앙상블 효과 극대화

## 🤖 AI 협업 전략
### Claude 활용 방법
1. **피처 아이디어 생성**: 30+ 프롬프트
2. **코드 리뷰 및 디버깅**: 실시간 오류 수정
3. **전략 수립**: 앙상블 가중치, 피처 선택
4. **문서화**: 체계적 프롬프트 로그

### 프롬프트 로그 구조
```
reports/prompts/
├── 01_data_understanding.md    # 데이터 구조 파악
├── 02_feature_engineering.md   # 피처 설계
├── 03_model_ensemble.md        # 앙상블 전략
├── 04_phase3_advanced_features.md # 고급 시계열 피처
└── 05_hyperparameter_tuning.md # 하이퍼파라미터 & 가중치 최적화
```
상세: [AI Collaboration Log](reports/prompts/05_hyperparameter_tuning.md)

## 📝 회고

### Day 1-3 (2025-12-11 ~ 12-13)
- ✅ 프로젝트 구조 설계 및 Git 설정
- ✅ 탐색적 데이터 분석 (EDA)
- ✅ Episode 구조 완전 파악
- ✅ 2단계 피처 엔지니어링
- ✅ 베이스라인 → 개선 모델 개발
- ✅ End-to-end ML 파이프라인 구축
- ✅ 첫 제출 성공 (LB 17.23)

#### 핵심 학습
1. **시계열 피처의 중요성**: 7개 피처로 15.6% 성능 개선
2. **과적합 방지**: CV 18.88 vs LB 17.23 (일반화 우수)
3. **파이프라인 구축**: 재현 가능한 ML 워크플로우
4. **AI 활용 전략**: 체계적 프롬프트 엔지니어링

### Day 3 (2025-12-13)
- ✅ LightGBM 모델 개발 (CV RMSE 18.81m)
- ✅ 앙상블 파이프라인 구축
- ✅ LB 점수 개선 (17.23 → 17.13)
- ✅ 순위 상승 (282 → 278)

#### 주요 학습
1. **모델 다양성의 중요성**
   - XGBoost: zone 중심
   - LightGBM: 시계열 피처 활용
   - 앙상블로 각 모델 장점 결합

2. **작은 개선의 누적**
   - 0.1m 개선도 의미 있음
   - 여러 기법 조합으로 큰 효과

### Day 4 (2025-12-14)
- ✅ CatBoost 모델 개발 (CV RMSE 18.97m)
- ✅ 앙상블 파이프라인 추가
- ✅ LB 점수 개선 (17.13 → 17.03)
- ✅ 가중치 실험 2: XGBoost 감소 [0.2, 0.4, 0.4]
   - 실험 1: [0.25, 0.5, 0.25]
   - 실험 2: [0.2, 0.4, 0.4] => LB: 17.0111116245로 성능 개선 
   - 실험 3: [0.2, 0.6, 0.2]

#### 주요 학습
1. **3개 모델 다양성 효과 검증**
   - 앙상블로 각 모델 장점 결합
   - 피처 중요도 차이 (3개 모델이 완전히 다른 패턴 학습!✨)
      - CatBoost: start_x 압도적 (37.4)
      - LightGBM: start_x, 시계열 균형
      - XGBoost: zone 중심
   - 가중치 실험 : XGBoost 감소 [0.2, 0.4, 0.4] => 기존 균등 가중치보다 성능 개선 보임

2. **작은 개선의 누적**
   - 예상 범위 내 개선 : 0.1m 개선
   - 여러 기법 조합으로 큰 효과

### Day 5 (2025-12-15)
- ✅ Phase 3 피처 23개 생성
- ✅ 극단값 문제 발견 및 수정
- ✅ 피처 선택 (23개 → 6개)
- ✅ XGBoost v3 학습 (18.91m)

#### 주요 학습
1. **극단값 처리의 중요성**
   - pass_velocity: 853 m/s → 40 m/s
   - 데이터 품질 = 모델 품질

2. **피처 선택의 가치**
   - 23개 전체: 18.99m (악화)
   - 6개 선별: 18.91m (회복)

### Day 6 (2025-12-16)
- ✅ LightGBM v3 학습 (18.82m)
- ✅ CatBoost v3 학습 (18.82m)
- ✅ 3개 모델 종합 분석
- ✅ v3 채택 결정
- ✅ 앙상블 예측 (16.98m)
- ✅ 프로젝트 문서화

#### 주요 학습
1. **CatBoost의 Phase 3 활용**
   - touchline_proximity 중요도 12.0
   - 유일하게 개선 (-0.15m)

2. **앙상블 효과 검증**
   - v2: 17.01m
   - v3: 16.98m (-0.03m)
   - 예상 범위 내 달성

### Day 7 (2025-12-17~2025-12-22)
- ✅ 하이퍼파라미터 수동 튜닝
  - XGBoost: 18.91m → 18.87m (-0.04m)
  - LightGBM: 18.82m → 18.81m (-0.01m)
  - CatBoost: 유지 (18.82m)
  - 앙상블 결과 : **[0.2, 0.4, 0.4] : 16.9724**
- ✅ Optuna 자동 튜닝 (LightGBM)
  - 최적 파라미터 탐색 (100회 trial)
  - CV 성능: 18.81m → 18.76m (-0.05m)
- ✅ Optuna 튜닝 후 앙상블 가중치 최적화 실험 (4개 조합)
  - [0.3, 0.3, 0.4]: 16.9912 ❌
  - [0.15, 0.45, 0.4]: 16.9778 ❌
  - [0.2, 0.5, 0.3]: 16.9905 ❌
  - [0.2, 0.35, 0.45]: 16.9776 ❌
- ✅ 기존 수동 튜닝 후 기존 가중치로 앙상블을 진행한 결과가 가장 좋은 결과를 보임 
  - **기존 [0.2, 0.4, 0.4]: 16.9724 ✅ 여전히 최고**

#### 주요 학습
1. **개별 성능 ≠ 앙상블 기여도**
   - Optuna LGB(18.76m)가 개별 최고
   - 앙상블에서는 기존 LGB(18.81m)와 동일
   - 다양성 > 개별 우수성

2. **하이퍼파라미터 튜닝의 한계**
   - 수동: -0.02m 개선
   - Optuna: -0.05m 개선 (개별)
   - 앙상블: ±0.00m (점수가 나빠짐)
   - → 일정 수준 이상에선 피처가 더 중요

3. **실험의 가치**
   - 실패한 실험도 중요한 정보
   - 가중치 최적화 불필요 확인
   - 다음 방향성 명확화 (피처 개발)


상세 내용: [reports/prompts/README.md](reports/prompts/README.md)

## 📝 종료 후 회고
(프로젝트 종료 후 작성)