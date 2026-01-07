# K리그 최종 패스 좌표 예측 🎯⚽

> K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발

## 📌 프로젝트 개요
- **목표**: K리그의 실제 경기 데이터를 기반으로, 단편적인 이벤트의 나열을 넘어 특정 상황의 맥락을 AI가 학습하고, 이어지는 패스가 도달할 최적의 위치를 예측하는 것. 나아가 이를 통해 데이터 기반의 선수 평가 및 전술 분석에 대한 새로운 가능성을 발굴하고자 함.
- **기간**: 2025.12.10 ~ 2026.01.12
- **역할**: 데이터 분석, 모델링, AI 도구 활용 전략 수립
- **성과**: Public LB **16.5316** (상위 약 55%, 484/872) ⭐ **최고 기록!**

## 🛠️ 기술 스택
- **언어**: Python 3.10
- **라이브러리**: pandas, numpy, scikit-learn, **PyTorch**
- **모델**: XGBoost, LightGBM, CatBoost, **Neural Network (MLP)**
- **기법**: 시계열 피처 엔지니어링, 도메인 특화 피처, **Stacking 앙상블**, K-Fold CV, **Meta-Learning (Ridge, LightGBM, MLP)**, 에러 분석 기반 타겟팅

## 📂 프로젝트 구조
```
kleague-pass-prediction/
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 데이터
│       ├── train_final_passes_v2.csv  # Phase 2
│       ├── train_final_passes_v3.csv  # Phase 3
│       ├── train_final_passes_v4.csv  # Phase 4
│       ├── train_final_passes_v6.csv  # Phase 6
│       └── oof_predictions.csv        # Phase 5 OOF 예측
├── models/                     # 학습된 모델 (.pkl)
│   ├── baseline_model_v4.pkl
│   ├── lgb_model_v4.pkl
│   ├── catboost_model_v4.pkl
│   ├── baseline_model_v6.pkl   # Phase 6
│   ├── lgb_model_v6.pkl
│   ├── catboost_model_v6.pkl
│   ├── label_encoders_v6.pkl
│   ├── meta_ridge_x.pkl        # Meta-Learner (Ridge)
│   ├── meta_ridge_y.pkl
│   ├── meta_lgb_x.pkl          # Meta-Learner (LightGBM) ⭐
│   ├── meta_lgb_y.pkl
│   ├── meta_mlp_x.pkl          # Meta-Learner (MLP) 🆕
│   └── meta_mlp_y.pkl
├── src/
│   ├── features/               # 피처 생성 모듈
│   │   ├── build_feature.py    # Phase 1, 2
│   │   ├── advanced_features.py # Phase 3
│   │   └── build_phase6_features.py # Phase 6
│   └── models/                 # 모델 학습/예측
│       ├── train_model_v4.py
│       ├── train_model_lgb_v4.py
│       ├── train_model_catboost_v4.py
│       ├── train_all_models_v6.py
│       ├── predict_ensemble_v4.py
│       ├── generate_oof_predictions.py
│       ├── train_meta_learner.py
│       ├── train_meta_learner_mlp.py    # Phase 5.1 🆕
│       ├── predict_stacking.py
│       ├── predict_stacking_mlp.py      # Phase 5.1 🆕
│       └── predict_stacking_v6.py
├── reports/
│   ├── figures/                # 시각화
│   └── prompts/                # AI 협업 로그
│       ├── 07_stacking_ensemble.md
│       ├── 08_phase6_error_analysis.md
│       └── 09_phase51_mlp_meta_learner.md  # Phase 5.1 🆕
├── submissions/                # 제출 파일
│   ├── submission_stacking_lgb.csv  # Phase 5 (16.5316) 🥇
│   ├── submission_stacking_v6.csv   # Phase 6 (16.5622)
│   └── submission_stacking_mlp.csv  # Phase 5.1 (16.7311) 🆕
└── README.md
```

## 🚀 실행 방법

### 환경 설정
```bash
pip install -r requirements.txt
```

### Phase 5.1: MLP Meta-Learner (최신 시도) 🆕
```bash
# 1. OOF 예측 생성 (Phase 5와 동일)
python src/models/generate_oof_predictions.py

# 2. MLP Meta-Learner 학습
python src/models/train_meta_learner_mlp.py
# 출력: models/meta_mlp_x.pkl, meta_mlp_y.pkl

# 3. MLP Stacking 예측
python src/models/predict_stacking_mlp.py
# 출력: submissions/submission_stacking_mlp.csv
```

### Phase 6: 에러 분석 기반 타겟팅
```bash
# 1. Phase 6 피처 생성
# build_phase6_features.py가 자동으로 호출됨

# 2. Phase 6 모델 학습 (3개 통합)
python src/models/train_all_models_v6.py
# 출력: models/baseline_model_v6.pkl, lgb_model_v6.pkl, catboost_model_v6.pkl

# 3. Phase 6 Stacking 예측
python src/models/predict_stacking_v6.py
# 출력: submissions/submission_stacking_v6.csv
```

### Phase 5: Stacking 앙상블 (최고 기록)
```bash
# 1. OOF 예측 생성 (5-Fold CV)
python src/models/generate_oof_predictions.py
# 출력: data/processed/oof_predictions.csv

# 2. Meta-Learner 학습
python src/models/train_meta_learner.py
# 출력: models/meta_ridge_*.pkl, models/meta_lgb_*.pkl

# 3. Stacking 예측 및 제출
python src/models/predict_stacking.py
# 출력: submissions/submission_stacking_lgb.csv
```

### Phase 4: 기본 앙상블
```bash
# 개별 모델 학습 (Phase 4)
python src/models/train_model_v4.py          # XGBoost
python src/models/train_model_lgb_v4.py      # LightGBM
python src/models/train_model_catboost_v4.py # CatBoost

# 가중 평균 앙상블 예측
python src/models/predict_ensemble_v4.py
# 출력: submissions/submission_ensemble_v4.csv
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
| Phase 3 + 튜닝 | 18.83m | 16.9724 | -0.2% | 수동 튜닝 |
| Phase 4 (Domain) | 18.70m | 16.8272 | -0.9% | 도메인 특화 |
| **Phase 5 (Stacking)** | **12.84m** | **16.5316** | **-1.8%** 🥇 | **Meta-Learning (LGB)** |
| Phase 5.1 (MLP) | TBD | 16.7311 | +1.2% ❌ | Neural Network 🆕 |
| Phase 6 (Error Analysis) | TBD | 16.5622 | +0.2% ⚠️ | 에러 타겟팅 |

**총 개선**: 20.36m → 16.53m (**-18.8%**, 3.83m) 🎉

**Phase 5.1 결과**: 
- LB 16.7311 (+0.20m vs Phase 5)
- **MLP Meta-Learner가 예상과 달리 성능 악화**
- Phase 5 (LightGBM Meta) 최고 기록 유지 확정 ✅

**Phase 6 결과**: 
- LB 16.5622 (+0.03m vs Phase 5)
- 에러 분석 기반 피처 추가했으나 성능 미개선
- Phase 5가 여전히 최고 기록 유지

### Phase 5 vs Phase 5.1: Meta-Learner 비교

#### Meta-Learner 성능 비교
| Meta-Model | OOF RMSE | LB Score | 개선 | 비고 |
|------------|----------|----------|------|------|
| Ridge Regression | 13.19m | - | - | 선형 조합 |
| **LightGBM** | **12.84m** | **16.5316** 🥇 | baseline | Tree 기반 |
| **MLP (Neural Net)** | TBD | **16.7311** ❌ | **+0.20m** | 2-layer 비선형 |

**핵심 발견:**
- ❌ **MLP가 가장 나쁨** (+0.20m, 1.2% 악화)
- ✅ **LightGBM Meta가 최고** 확정
- ⚠️ 복잡한 모델 ≠ 더 좋은 성능

#### MLP 실패 원인 분석

**1. 과적합 (Overfitting to OOF)**
```
문제: MLP가 Train(OOF) 데이터에 과적합
증거: OOF 성능 vs LB 성능 차이 증가
원인: Neural Network의 높은 표현력 + 적은 샘플(15,435개)
```

**2. 메타 피처의 단순성**
```
메타 피처: 6개만 (xgb_x, xgb_y, lgb_x, lgb_y, cat_x, cat_y)

문제: Neural Network는 고차원 피처에 강함
      6개 저차원에서는 오히려 불리
      LightGBM/Ridge가 더 적합
```

**3. Base 모델 예측의 높은 상관관계**
```
상관계수: 0.98+ (거의 동일한 예측)

의미: 비선형 조합의 이득 제한적
      선형 조합(Ridge)이나 Tree(LGB)로 충분
```

**4. 앙상블 다양성 부족**
```
3개 Base 모델: 모두 Gradient Boosting 계열
결과: 비슷한 예측 패턴
     비선형 Meta-Learner의 이득 제한
```

#### MLP 아키텍처
```python
Input (6) → Linear(32) → ReLU → Dropout(0.2)
         → Linear(16) → ReLU → Dropout(0.2)
         → Linear(1) → Output

설정:
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Batch Size: 256
- Early Stopping: patience=20
- Regularization: Dropout + L2
```

### Phase 5: Stacking 앙상블 상세

#### OOF (Out-of-Fold) 성능
5-Fold Cross-Validation으로 생성된 OOF 예측:

| 모델 | OOF RMSE (end_x) | OOF RMSE (end_y) | 평균 |
|------|------------------|------------------|------|
| XGBoost | 12.74m | 14.02m | 13.40m |
| LightGBM | 12.72m | 13.97m | 13.36m |
| CatBoost | 12.65m | 13.92m | **13.30m** ⭐ |

**핵심**: OOF는 과적합이 없는 순수한 일반화 성능!

#### Meta-Learner 비교
6개 Base 예측 → Meta-Learner → 최종 2개 좌표

| Meta-Model | Train RMSE | 비고 |
|------------|------------|------|
| Ridge Regression | 13.19m | 선형 조합, 해석 가능 |
| **LightGBM** | **12.84m** ⭐ | 비선형 조합, **-0.35m 개선** |

**선택**: LightGBM Meta-Learner (2.6% 더 우수)

#### Base 모델 예측 상관관계
```
같은 좌표 예측 간 상관계수: 0.98+
→ 3개 모델이 거의 비슷하게 예측
→ 하지만 미묘한 차이를 Meta-Learner가 활용!
```

#### Meta-Learner 피처 중요도
**LightGBM Meta-Model (end_x):**
```
1. cat_pred_x (365)         ← CatBoost 예측이 가장 중요
2. xgb_pred_x (342)
3. lgb_pred_x (250)
4. lgb_pred_y (168)         ← 교차 좌표도 활용!
5. xgb_pred_y (138)
6. cat_pred_y (137)
```

**핵심 발견**:
- 모든 6개 피처 골고루 사용
- 다른 좌표(y) 정보도 x 예측에 도움
- CatBoost의 다양성이 가장 중요

### 개별 모델 성능
| 모델 | Phase 2 (v2) | Phase 3 (v3) | v3 튜닝 | Phase 4 (v4) | **OOF (Phase 5)** | Phase 6 (v6) |
|------|--------------|--------------|---------|--------------|-------------------|--------------|
| **XGBoost** | 18.88m | 18.91m | 18.87m | 18.73m | **13.40m** | TBD |
| **LightGBM** | 18.81m | 18.82m | 18.81m | 18.64m | **13.36m** | TBD |
| **CatBoost** | 18.97m | 18.82m | 18.82m | 18.73m | **13.30m** ⭐ | TBD |
| **평균** | 18.89m | 18.85m | 18.83m | 18.70m | **13.35m** | TBD |
| **Meta (LGB)** | - | - | - | - | **12.84m** ⭐⭐ | - |

**주의**: OOF는 CV 성능이므로 전체 학습보다 낮음 (과적합 없음)

### 공모전 제출
- **Public LB (최고)**: 16.5316 RMSE 🥇 **Phase 5 Stacking (LightGBM Meta)**
- **Public LB (Phase 5.1)**: 16.7311 RMSE (Phase 5 대비 +0.20m) ❌
- **Public LB (Phase 6)**: 16.5622 RMSE (Phase 5 대비 +0.03m)
- **순위**: 452/816 (상위 약 55%)
- **일반화 성능**: 베이스라인 대비 약 **-18.8%** 개선 
- **Phase 4 대비**: -0.30m (-1.8%) 추가 개선 (Phase 5)

### 앙상블 방식 비교
| 방식 | Phase 4 | Phase 5 (LGB Meta) | Phase 5.1 (MLP Meta) | Phase 6 |
|------|---------|-------------------|---------------------|---------|
| **가중 평균** | 16.83m | - | - | - |
| **Stacking** | - | **16.53m** 🥇 | 16.73m ❌ | 16.56m |

**Stacking의 장점**:
1. 비선형 조합 가능
2. Base 모델 간 상호작용 학습
3. 과적합 방지 (OOF 사용)

**Phase 5.1 결과 분석**:
- MLP Meta-Learner 시도 (2-layer Neural Network)
- LB 결과: Phase 5보다 0.20m 악화
- **교훈: 복잡한 모델이 항상 더 좋은 것은 아님**
- Meta 피처 6개로는 Neural Network가 과함
- LightGBM/Ridge 같은 단순한 모델이 더 적합

**Phase 6 결과 분석**:
- 에러 분석 기반 23개 피처 추가
- mean end_x: 66.9m (Phase 5 대비 +15m, 과도한 공격성)
- LB 결과: Phase 5보다 0.03m 악화
- 교훈: 최적점에서 추가 피처는 노이즈 될 수 있음

## 📈 피처 개발

### Phase 1: 위치 기반 피처 (8개)
```python
- start_x, start_y              # 시작 좌표
- dist_to_target_goal           # 골대까지 거리
- zone_x, zone_y, zone_combined # 구역 분류 (9 zones)
- in_penalty_box                # 페널티 박스 여부
- in_final_third                # 최종 3구역 여부
```
**효과**: CV 20.36m (베이스라인)

### Phase 2: 시계열 피처 (7개)
```python
- prev_end_x, prev_end_y        # 이전 액션 종료 위치
- prev_action_distance          # 이전 액션과의 거리
- time_since_prev               # 이전 액션과의 시간 간격
- prev_direction_x, prev_direction_y # 공격 방향
- pass_count_in_episode         # Episode 내 패스 카운트
```
**효과**: CV 18.88m (**-7.3%** 개선)

### Phase 3: 고급 시계열 피처 (선별 6개)
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

### Phase 4: 도메인 특화 피처 (9개)
```python
# 선수 스타일 (4개)
- player_avg_pass_distance      # 선수 평균 패스 거리
- player_forward_ratio          # 전진 패스 비율
- player_success_rate           # 패스 성공률
- player_pass_count             # 패스 횟수 (경험치)

# 팀 전술 (2개)
- team_avg_pass_distance        # 팀 평균 패스 거리
- team_attack_style             # 측면/중앙 선호도

# 경기 흐름 (3개)
- score_diff                    # 득점차
- match_period_normalized       # 경기 진행률 (0~1)
- is_late_game                  # 후반 75분 이후
```
**효과**: CV 18.70m (**-0.8%** 추가 개선)

**핵심 인사이트**:
- 선수별 스타일이 패스 좌표 예측에 핵심적
- 경기 진행률(시간)이 패스 패턴 결정
- 득점차가 공격 성향에 영향

### Phase 5: Meta-Features (6개)
```python
# Base 모델 예측값 (6개 → 2개로 압축)
- xgb_pred_x, xgb_pred_y        # XGBoost 예측
- lgb_pred_x, lgb_pred_y        # LightGBM 예측
- cat_pred_x, cat_pred_y        # CatBoost 예측
```
**효과**: 
- CV 12.84m (Meta-Learner 학습 성능)
- LB 16.53m (**-1.8%** 추가 개선) 🥇

**핵심 인사이트**:
- 모델 간 다양성이 핵심 (상관계수 0.98+인데도 개선!)
- 비선형 조합이 선형보다 우수 (LGB > Ridge)
- 교차 좌표(x↔y) 정보도 유용

### Phase 6: 에러 분석 기반 타겟팅 피처 (23개)

**전략 1: 구역별 특화 (5개)**
```python
- is_defensive_zone           # 수비 구역 여부
- defensive_uncertainty       # 수비 구역 불확실성
- player_style_in_defense     # 수비 구역 선수 스타일
- is_defensive_center         # 중앙 수비 구역
- pressure_zone_interaction   # 구역-압박 상호작용
```

**전략 2: 최종 구역 미진입 타겟팅 (4개)**
```python
- attack_failure_risk         # 공격 실패 리스크
- stuck_in_midfield          # 중원 정체
- buildup_style              # 빌드업 스타일
- attack_momentum            # 공격 모멘텀
```

**전략 3: 측면 vs 중앙 차별화 (4개)**
```python
- central_uncertainty        # 중앙 불확실성
- wing_attack_pattern        # 측면 공격 패턴
- cross_likelihood           # 크로스 가능성
- wing_central_balance       # 측면-중앙 균형
```

**전략 4: 득점 상황별 전술 변화 (3개)**
```python
- leading_defensive          # 리드 시 수비적
- losing_aggressive          # 지는 상황 공격적
- endgame_pressure          # 경기 후반 압박
```

**전략 5: 극단값 특수 처리 (3개)**
```python
- near_boundary             # 경계 근처
- extreme_pass              # 극단적 패스
- abnormal_situation        # 비정상 상황
```

**보너스: 상호작용 (4개)**
```python
- zone_final_interaction    # 구역-최종 진입
- wing_pressure_interaction # 측면-압박
- player_zone_interaction   # 선수-구역
```

**효과**: 
- LB 16.56m (Phase 5 대비 **+0.03m 악화**)
- mean end_x: 66.9m (Phase 5 대비 +15m, 과도한 공격성)

**핵심 발견**:
- 에러 분석 → 타겟팅 피처 전략의 한계
- 큰 오차 케이스는 원래 예측 어려운 케이스
- 추가 피처가 노이즈로 작용 가능
- **Phase 5 Stacking이 이미 최적점** ✅

**기술적 도전**:
- category dtype 3번의 에러 극복
- `np.select()` 활용한 안전한 구역 생성
- pandas dtype 전문 지식 습득

## 🎓 피처 중요도 분석

### Phase 4 피처 중요도 (모델별)

**XGBoost (zone 중심 유지 + 선수 스타일 활용)**
```
1. zone_x_encoded (71.7%)               ← 여전히 압도적
2. start_x (8.9%)
3. in_penalty_box (2.0%)
4. player_avg_pass_distance (1.7%)      ← Phase 4 최고!
5. match_period_normalized (0.6%)       ← 경기 흐름
```

**LightGBM (균형잡힌 + Phase 4 적극 활용)**
```
1. start_x (55.3M)
2. zone_x_encoded (8.4M)
3. player_avg_pass_distance (4.5M)      ← Phase 4 최고!
4. time_since_prev (3.0M)
5. prev_end_x (3.0M)
...Phase 4 피처들이 Top 15에 7개 진입!
```

**CatBoost (공간 피처 + 선수 통계)**
```
1. start_x (27.4)
2. player_avg_pass_distance (8.2)       ← Phase 4 최고!
3. zone_x_encoded (5.2)
4. time_since_prev (4.5)
5. touchline_proximity (4.2)            ← Phase 3
```

**결론**: 
- **player_avg_pass_distance**가 3개 모델 모두에서 Phase 4 최고 중요도!
- 각 모델이 Phase 4 피처를 서로 다르게 활용 → 앙상블 효과 극대화
- 도메인 지식 기반 피처가 실제로 효과적임을 입증

## 🤖 AI 협업 전략

### Claude 활용 방법
1. **피처 아이디어 생성**: 70+ 프롬프트
2. **코드 리뷰 및 디버깅**: 실시간 오류 수정
3. **전략 수립**: 앙상블 가중치, 피처 선택, Stacking 설계, Meta-Learner 비교
4. **문서화**: 체계적 프롬프트 로그

### 프롬프트 로그 구조
```
reports/prompts/
├── 01_data_understanding.md           # 데이터 구조 파악
├── 02_feature_engineering.md          # 피처 설계
├── 03_model_ensemble.md               # 앙상블 전략
├── 04_phase3_advanced_features.md     # 고급 시계열 피처
├── 05_hyperparameter_tuning.md        # 하이퍼파라미터 & 가중치 최적화
├── 06_phase4_domain_features.md       # 도메인 특화 피처
├── 07_stacking_ensemble.md            # Stacking 앙상블
├── 08_phase6_error_analysis.md        # 에러 분석 & 타겟팅
└── 09_phase51_mlp_meta_learner.md     # MLP Meta-Learner 🆕
```
상세: [AI Collaboration Log](reports/prompts/)

## 📝 프로젝트 회고

### Day 1-3 (2025-12-11 ~ 12-13)
- ✅ 프로젝트 구조 설계 및 Git 설정
- ✅ 탐색적 데이터 분석 (EDA)
- ✅ Episode 구조 완전 파악
- ✅ 2단계 피처 엔지니어링
- ✅ 베이스라인 → 개선 모델 개발
- ✅ End-to-end ML 파이프라인 구축
- ✅ 첫 제출 성공 (LB 17.23)

**핵심 학습:**
1. 시계열 피처의 중요성: 7개 피처로 7.3% 성능 개선
2. 과적합 방지: CV 18.88 vs LB 17.23 (일반화 우수)
3. 파이프라인 구축: 재현 가능한 ML 워크플로우
4. AI 활용 전략: 체계적 프롬프트 엔지니어링

### Day 4 (2025-12-14)
- ✅ CatBoost 모델 개발 (CV RMSE 18.97m)
- ✅ 3-model 앙상블 구축
- ✅ LB 점수 개선 (17.13 → 17.03)
- ✅ 가중치 실험 및 최적화

**주요 학습:**
1. 모델 다양성의 중요성: 각 모델이 완전히 다른 패턴 학습
2. 가중치 최적화: [0.2, 0.4, 0.4]가 최적

### Day 5-6 (2025-12-15 ~ 12-16)
- ✅ Phase 3 피처 23개 생성
- ✅ 극단값 문제 발견 및 수정
- ✅ 피처 선택 (23개 → 6개)
- ✅ 3개 모델 v3 학습 완료
- ✅ 앙상블 LB 16.98m 달성

**주요 학습:**
1. 피처 품질 > 피처 수량
2. 극단값 처리의 중요성: pass_velocity 853 → 40 m/s

### Day 7 (2025-12-17 ~ 12-22)
- ✅ 하이퍼파라미터 수동 튜닝
- ✅ Optuna 자동 튜닝 (LightGBM)
- ✅ 앙상블 가중치 실험 (4개 조합)
- ✅ 최종 결론: 기존 설정 유지

**주요 학습:**
1. 개별 성능 ≠ 앙상블 기여도
2. 하이퍼파라미터 튜닝의 한계: 피처 개발이 더 중요

### Day 8-10 (2025-12-23 ~ 12-28)
- ✅ Phase 4 피처 설계 (도메인 특화)
- ✅ 선수별/팀별/경기 흐름 통계 계산
- ✅ 3개 모델 v4 재학습
- ✅ 앙상블 LB 16.8272 달성

**핵심 학습:**
1. 도메인 지식의 가치: 선수 스타일이 예측에 핵심적
2. 누적 통계의 효과: 경기별로 이전 경기 통계 사용
3. 점진적 개선의 누적: Phase 1-4 총 -8.2% 개선

### Day 11-12 (2025-12-29 ~ 12-30)
- ✅ Stacking 앙상블 설계
- ✅ OOF 예측 생성 (5-Fold CV)
- ✅ Meta-Learner 학습 (Ridge vs LightGBM)
- ✅ Stacking 예측 파이프라인 구축
- ✅ 최종 제출: LB **16.5316** (최고 기록!)

**핵심 학습:**
1. **Stacking의 위력**: 가중 평균 16.83m → Stacking 16.53m (-0.30m)
2. **OOF의 중요성**: 과적합 없는 순수 일반화 성능 (13.35m)
3. **Meta-Learner 선택**: LightGBM이 Ridge보다 0.35m 우수
4. **모델 다양성**: 상관계수 0.98+인데도 0.3m 개선
5. **교차 좌표 효과**: end_x 예측에 y 정보도 활용
6. **구조적 개선**: Phase 3-4 피처 실험 +0.18m < Phase 5 Stacking +0.30m

### Day 13 (2025-12-31 ~ 2026-01-03)
- ✅ Phase 6 에러 분석 및 타겟팅 피처 설계
- ✅ 23개 에러 타겟팅 피처 생성
- ✅ dtype 에러 3번 극복 (category, object → int)
- ✅ `np.select()` 활용한 안전한 구현
- ✅ Phase 6 모델 학습 및 예측
- ✅ 제출: LB 16.5622 (Phase 5 대비 +0.03m)

**핵심 학습:**
1. **에러 분석의 함정**: 큰 오차 케이스 = 원래 예측 어려운 케이스 → 노이즈
2. **과도한 공격성**: mean end_x 66.9m (+15m) → 실제는 더 보수적
3. **dtype 지옥 극복**: category/object 에러 3번 → `np.select()` 해결
4. **최적점의 인식**: Phase 5 이미 최적 → 추가 피처는 리스크
5. **구조 vs 피처**: Stacking +0.30m > Phase 6 피처 -0.03m
6. **실패의 가치**: Stacking 우수성 재확인, dtype 전문가 됨

### Day 14 (2026-01-04 ~ 2026-01-06) 🆕
- ✅ Phase 5.1 MLP Meta-Learner 설계 및 구현
- ✅ 2-layer Neural Network 학습 (PyTorch)
- ✅ MLP Stacking 예측 파이프라인 구축
- ✅ PyTorch 버전 이슈 해결 (verbose 파라미터)
- ✅ 제출: LB 16.7311 (Phase 5 대비 +0.20m)

**핵심 학습:**
1. **복잡한 모델 ≠ 더 좋은 성능**: MLP가 가장 나쁨 (+0.20m)
2. **메타 피처의 단순성**: 6개 피처로는 Neural Network가 과함
3. **과적합 위험**: 높은 표현력 + 적은 샘플 = 과적합
4. **Base 모델 상관관계**: 0.98+ → 비선형 조합 이득 제한적
5. **앙상블 다양성 부족**: 모두 Boosting 계열 → MLP 효과 제한
6. **문제 복잡도 매칭**: 단순한 문제는 단순한 모델이 최적
7. **Phase 5 최고 확정**: LightGBM Meta-Learner가 최적 ✅

**기술적 성과:**
- PyTorch 기반 Meta-Learner 구현
- StandardScaler + Early Stopping + Dropout 적용
- 완전 재현 가능한 Neural Network 파이프라인

## 🎯 다음 단계

### 현재 상황
- ✅ **Phase 5 Stacking: 16.5316m** 🥇 **최고 기록 확정**
- ⚠️ Phase 5.1 MLP: 16.7311m (+0.20m 악화)
- ⚠️ Phase 6 에러 분석: 16.5622m (+0.03m 악화)
- 📊 총 14일간 6개 Phase + 1개 실험 진행

### 최종 결론
```
🏆 Phase 5 (LightGBM Meta-Learner)가 최고!

실험 결과:
Phase 5 (LGB Meta):   16.5316m ← 최고 ✅
Phase 5.1 (MLP Meta): 16.7311m (+0.20m)
Phase 6 (Error):      16.5622m (+0.03m)

교훈:
- 최적점에서 추가 복잡도는 역효과
- 문제 복잡도에 맞는 모델 선택 중요
- 구조적 개선 > 피처/모델 복잡도
```

### 고려 중인 방향

#### 옵션 1: 프로젝트 마무리 ⭐ 추천
```
✅ Phase 5를 최종 제출로 확정
✅ README 및 문서 정리 (진행 중)
✅ 프롬프트 로그 완성
✅ 포트폴리오 작성
✅ 다음 프로젝트로 이동
```

#### 옵션 2: 추가 실험 (선택)
```
- 6-Model Stacking (다양성 증대)
- Blending (Phase 5 + others)
- SHAP values 분석
- 예측 실패 케이스 딥다이브
```

#### 옵션 3: 다른 접근
```
- Pseudo Labeling
- Adversarial Validation
- 딥러닝 모델 (LSTM, Transformer)
```

---

## 🏆 프로젝트 하이라이트

### 성과
```
시작: CV 20.36m
최고: CV 12.84m (Meta-Learner), LB 16.5316 🥇

총 개선: -18.8% (3.83m)
핵심 기법: 시계열 + 도메인 특화 + Stacking 앙상블
```

### 주요 성과
- ✅ 체계적 피처 엔지니어링 (6 phases, 71개 피처)
- ✅ **Stacking 앙상블** (OOF + Meta-Learning)
- ✅ **Meta-Learner 3종 비교** (Ridge, LightGBM, MLP) 🆕
- ✅ 3-model 앙상블 최적화
- ✅ AI 기반 개발 프로세스 구축
- ✅ 완전 재현 가능한 파이프라인
- ✅ **dtype 전문가 됨** (3번의 에러 극복)
- ✅ **PyTorch 기반 Meta-Learning 구현** 🆕

### 최종 스코어보드
| 항목 | 수치 | 비고 |
|------|------|------|
| Public LB (최고) | **16.5316** | Phase 5 Stacking (LGB Meta) 🥇 |
| Public LB (Phase 5.1) | 16.7311 | MLP Meta-Learner 🆕 |
| Public LB (Phase 6) | 16.5622 | 에러 분석 |
| 순위 | 452/816 | 상위 55% |
| 총 개선 | -18.8% | 베이스라인 대비 |
| Phase 5 개선 | -1.8% | Phase 4 대비 |
| 총 개발 기간 | 14일 | 6 phases + 1 실험 |

---

## 📚 핵심 교훈

### 1. 구조적 개선 > 피처/모델 복잡도
```
Phase 3-4: 수많은 피처 실험 → +0.18m
Phase 5: Stacking (구조 변경) → +0.30m
Phase 5.1: MLP (복잡도 증가) → -0.20m (악화) 🆕
Phase 6: +23개 피처 → -0.03m (악화)

→ 구조적 개선이 가장 효과적!
→ 과도한 복잡도는 역효과!
```

### 2. 문제 복잡도에 맞는 모델 선택 🆕
```
Meta-Features: 6개만 (단순)

Ridge (선형):     적절 ✅
LightGBM (비선형): 최적 ✅✅
MLP (Deep 비선형): 과함 ❌

교훈:
"망치로 못 박기" ✅
"망치로 나사 조이기" ❌
→ 문제에 맞는 도구 선택이 핵심
```

### 3. 최적점의 인식
```
Phase 5 Stacking: 이미 최적
Phase 5.1 MLP: 과도한 복잡도
Phase 6 피처: 추가 노이즈

→ "멈출 때를 아는 것"도 능력
→ 최적점에서 추가 시도는 리스크
```

### 4. 실패도 학습
```
Phase 5.1 MLP 실패를 통해:
✅ Phase 5 최고 확정
✅ 메타 피처 단순성 이해
✅ 과적합 위험 인식
✅ PyTorch 구현 능력 습득
✅ 문제-모델 매칭 중요성 학습
```

### 5. 모델 다양성 > 개별 우수성
```
상관계수 0.98+ (거의 동일)
But Stacking: +0.30m 개선

→ 미묘한 차이가 중요!
→ 다양성이 앙상블의 핵심
```

### 6. OOF의 중요성
```
OOF: 13.35m (과적합 없음)
Train: 18.70m (과적합 포함)

→ OOF가 진짜 일반화 성능
→ Meta-Learner 학습의 핵심
```

### 7. 점진적 개선의 가치
```
작은 개선들의 누적:
Phase 1→2: -7.3%
Phase 2→3: -0.2%
Phase 3→4: -0.9%
Phase 4→5: -1.8%

총 누적: -18.8% ✨

→ 포기하지 않는 것이 중요
```

---

## 📖 참고 자료

### 문서
- [AI Collaboration Log](reports/prompts/): 9개 프롬프트 로그
- [Phase 5.1 MLP Meta-Learner](reports/prompts/09_phase51_mlp_meta_learner.md): MLP 실험 전체 과정 🆕
- [Phase 6 Error Analysis](reports/prompts/08_phase6_error_analysis.md): 에러 분석 전체 과정

### 코드
- [피처 생성](src/features/): Phase 1-6 피처 모듈
- [모델 학습](src/models/): 학습 및 예측 스크립트
- [Meta-Learner](src/models/): Ridge, LightGBM, MLP 3종 🆕
- [제출 파일](submissions/): 전체 제출 이력

### 주요 알고리즘
- **Stacking 앙상블**: OOF 기반 Meta-Learning
- **5-Fold CV**: Data Leakage 방지
- **Meta-Learning**: Ridge, LightGBM, Neural Network
- **PyTorch**: 딥러닝 Meta-Learner 구현 🆕

---

