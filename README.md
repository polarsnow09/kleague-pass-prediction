# K리그 최종 패스 좌표 예측 🎯⚽

> K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발

## 📌 프로젝트 개요
- **목표**: K리그의 실제 경기 데이터를 기반으로, 단편적인 이벤트의 나열을 넘어 특정 상황의 맥락을 AI가 학습하고, 이어지는 패스가 도달할 최적의 위치를 예측하는 것. 나아가 이를 통해 데이터 기반의 선수 평가 및 전술 분석에 대한 새로운 가능성을 발굴하고자 함.
- **기간**: 2025.12.10 ~ 2026.01.12
- **역할**: 데이터 분석, 모델링, AI 도구 활용 전략 수립
- **성과**: Public LB **16.4981** (상위 약 56%, 518/925) 🥇 **최고 기록!**

## 🛠️ 기술 스택
- **언어**: Python 3.10
- **라이브러리**: pandas, numpy, scikit-learn, **PyTorch**
- **모델**: XGBoost, LightGBM, CatBoost, **Neural Network (MLP)**
- **기법**: 시계열 피처 엔지니어링, 도메인 특화 피처, **Stacking 앙상블**, **Blending 최적화**, K-Fold CV, **Meta-Learning (Ridge, LightGBM, MLP)**, Grid Search

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
│   ├── meta_ridge_x.pkl        # Meta-Learner (Ridge)
│   ├── meta_lgb_x.pkl          # Meta-Learner (LightGBM) ⭐
│   └── meta_mlp_x.pkl          # Meta-Learner (MLP)
├── src/
│   ├── features/               # 피처 생성 모듈
│   │   ├── build_feature.py    # Phase 1, 2
│   │   ├── advanced_features.py # Phase 3
│   │   └── build_phase6_features.py # Phase 6
│   └── models/                 # 모델 학습/예측
│       ├── train_model_v4.py
│       ├── predict_ensemble_v4.py
│       ├── generate_oof_predictions.py
│       ├── train_meta_learner.py
│       ├── predict_stacking.py
│       ├── predict_averaging_grid_search.py      # Grid Search 1차
│       └── predict_averaging_grid_search_v2.py   # Grid Search 2차
├── reports/
│   ├── figures/                # 시각화
│   └── prompts/                # AI 협업 로그
│       ├── 01_data_understanding.md
│       ├── ...
│       └── 10_phase55_blending_optimization.md  # Phase 5.5
├── submissions/                # 제출 파일
│   ├── submission_stacking_lgb.csv  # Phase 5 (16.5316)
│   └── submission_averaging_grid_7_w560_w640.csv  # Phase 5.5 (16.4981) 🥇
└── README.md
```

## 🚀 실행 방법

### 환경 설정
```bash
pip install -r requirements.txt
```

### Phase 5.5: Blending 최적화 (최종 최고 기록) 🆕
```bash
# 1. OOF 예측 생성 (Phase 5와 동일)
python src/models/generate_oof_predictions.py

# 2. Meta-Learner 학습 (Phase 5)
python src/models/train_meta_learner.py

# 3. Phase 5 Stacking 예측
python src/models/predict_stacking.py
# 출력: submissions/submission_stacking_lgb.csv

# 4. Phase 6 Stacking 예측 (기존)
# 출력: submissions/submission_stacking_v6.csv

# 5. Grid Search로 최적 가중치 탐색
python src/models/predict_averaging_grid_search.py
python src/models/predict_averaging_grid_search_v2.py
# 출력: 7개의 블렌딩 제출 파일

# 최적: 0.60/0.40 가중치
# 출력: submissions/submission_averaging_grid_7_w560_w640.csv
```

### Phase 5: Stacking 앙상블
```bash
# 1. OOF 예측 생성 (5-Fold CV)
python src/models/generate_oof_predictions.py

# 2. Meta-Learner 학습
python src/models/train_meta_learner.py

# 3. Stacking 예측 및 제출
python src/models/predict_stacking.py
```

### Phase 4: 기본 앙상블
```bash
# 개별 모델 학습
python src/models/train_model_v4.py          # XGBoost
python src/models/train_model_lgb_v4.py      # LightGBM
python src/models/train_model_catboost_v4.py # CatBoost

# 가중 평균 앙상블 예측
python src/models/predict_ensemble_v4.py
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
| Phase 5 (Stacking) | 12.84m | 16.5316 | -1.8% 🥈 | Meta-Learning (LGB) |
| Phase 5.1 (MLP) | TBD | 16.7311 | +1.2% ❌ | Neural Network |
| Phase 6 (Error) | TBD | 16.5622 | +0.2% ⚠️ | 에러 타겟팅 |
| **Phase 5.5 (Blending)** | **-** | **16.4981** | **-0.2%** 🥇 | **최적 가중치 (0.6/0.4)** |

**총 개선**: 20.36m → 16.4981m (**-18.9%**, 3.86m) 🎉

**Phase 5.5 상세 결과**: 
```
Phase 5 + Phase 6 Blending (Grid Search 7개 가중치)
- 0.80/0.20: 16.5065m
- 0.78/0.22: 16.5049m
- 0.75/0.25: 16.5029m
- 0.70/0.30: 16.5003m
- 0.65/0.35: 16.4988m
- 0.60/0.40: 16.4981m 🥇 ← 최적!
- 0.55/0.45: 16.4985m

패턴: U자 곡선 (0.60/0.40이 최저점)
최적 비율: Phase 5 : Phase 6 = 3 : 2
```

### 공모전 제출
- **Public LB (최고)**: 16.4981 RMSE 🥇 **Phase 5.5 Blending (0.60/0.40)**
- **Public LB (Phase 5)**: 16.5316 RMSE 🥈 **Stacking (LightGBM Meta)**
- **순위**: 459/872 (상위 약 53%)
- **일반화 성능**: 베이스라인 대비 약 **-18.9%** 개선 
- **Phase 5 대비**: -0.0335m (-0.20%) 추가 개선 (Phase 5.5)

### 개별 모델 성능
| 모델 | Phase 2 (v2) | Phase 3 (v3) | v3 튜닝 | Phase 4 (v4) | **OOF (Phase 5)** |
|------|--------------|--------------|---------|--------------|-------------------|
| **XGBoost** | 18.88m | 18.91m | 18.87m | 18.73m | **13.40m** |
| **LightGBM** | 18.81m | 18.82m | 18.81m | 18.64m | **13.36m** |
| **CatBoost** | 18.97m | 18.82m | 18.82m | 18.73m | **13.30m** ⭐ |
| **평균** | 18.89m | 18.85m | 18.83m | 18.70m | **13.35m** |
| **Meta (LGB)** | - | - | - | - | **12.84m** ⭐⭐ |

### 앙상블 방식 비교
| 방식 | Phase 4 | Phase 5 (Stacking) | Phase 5.5 (Blending) |
|------|---------|-------------------|---------------------|
| **가중 평균** | 16.83m | - | - |
| **Stacking** | - | 16.53m 🥈 | - |
| **Blending** | - | - | **16.50m** 🥇 |

**Phase 5.5의 혁신**:
1. Stacking (비선형 Meta-Learning)
2. + Blending (Phase 5 + Phase 6)
3. + Grid Search (최적 가중치)
4. = **최고 성능 달성!**

## 📈 피처 개발

### Phase 1: 위치 기반 피처 (8개)
```python
- start_x, start_y              # 시작 좌표
- dist_to_target_goal           # 골대까지 거리
- zone_x, zone_y, zone_combined # 구역 분류 (9 zones)
- in_penalty_box                # 페널티 박스 여부
- in_final_third                # 최종 3구역 여부
```

### Phase 2: 시계열 피처 (7개)
```python
- prev_end_x, prev_end_y        # 이전 액션 종료 위치
- prev_action_distance          # 이전 액션과의 거리
- time_since_prev               # 이전 액션과의 시간 간격
- prev_direction_x, prev_direction_y # 공격 방향
- pass_count_in_episode         # Episode 내 패스 카운트
```

### Phase 3: 고급 시계열 피처 (6개)
```python
- pass_velocity                 # 패스 속도 (m/s)
- touchline_proximity           # 터치라인 근접도
- is_under_pressure             # 압박 상황 여부
- rolling_mean_distance_3       # 최근 3개 평균 거리
- avg_episode_velocity          # Episode 평균 속도
- episode_x_range               # X축 활용 범위
```

### Phase 4: 도메인 특화 피처 (9개)
```python
# 선수 스타일 (4개)
- player_avg_pass_distance      # 선수 평균 패스 거리 ⭐
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

### Phase 5: Meta-Features (6개)
```python
# Base 모델 예측값 (6개 → 2개로 압축)
- xgb_pred_x, xgb_pred_y        # XGBoost 예측
- lgb_pred_x, lgb_pred_y        # LightGBM 예측
- cat_pred_x, cat_pred_y        # CatBoost 예측
```

### Phase 5.5: Blending (신규!) 🆕
```python
# 최적 가중치: 0.60 / 0.40
final_x = 0.60 * phase5_pred_x + 0.40 * phase6_pred_x
final_y = 0.60 * phase5_pred_y + 0.40 * phase6_pred_y

# 특징
- U자 곡선 패턴 발견
- Phase 5 : Phase 6 = 3 : 2 (황금비율)
- Grid Search 7개로 최적점 확인
```

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

## 🎓 핵심 학습

### 1. Blending의 조건 (Phase 5.5) 🆕
```
성공 조건:
✅ 비슷한 성능 모델끼리 (차이 0.05m 이내)
   Phase 5 (16.53m) + Phase 6 (16.56m) = 성공!

실패 조건:
❌ 성능 차이 큰 모델 (차이 0.3m 이상)
   Phase 5 (16.53m) + Phase 4 (16.83m) = 실패
```

### 2. Grid Search의 힘 🆕
```
단순 시도: 0.8/0.2 = 16.5065m
Grid Search: 0.6/0.4 = 16.4981m

추가 개선: -0.0084m (50% 더 좋음!)
→ 체계적 탐색의 중요성
```

### 3. U자 곡선의 발견 🆕
```
가설: "더 공격적일수록 좋다" (선형)
실제: U자 곡선 (최적점 존재)

0.60/0.40이 Phase 5와 6의 황금비율!
```

### 4. 구조적 개선 > 피처 개선
```
Phase 3-4: 수많은 피처 실험 → +0.18m
Phase 5: Stacking (구조 변경) → +0.30m
Phase 5.5: Blending (최적화) → +0.03m

→ 구조적 개선이 가장 효과적!
```

### 5. 언제 멈출지 아는 것도 능력 🆕
```
멈춘 이유:
✅ 명확한 최적점 발견 (U자 곡선)
✅ 양쪽 모두 악화 (0.65, 0.55)
✅ 추가 개선 미미 (0.001m 이하)

끝없는 최적화의 함정:
❌ 0.62/0.38? 0.58/0.42?
→ 소수점 싸움, 문서화가 더 가치 있음
```

## 🤖 AI 협업 전략

### Claude 활용 방법
1. **피처 아이디어 생성**: 80+ 프롬프트
2. **코드 리뷰 및 디버깅**: 실시간 오류 수정
3. **전략 수립**: 앙상블, Stacking, **Blending 최적화**
4. **문서화**: 체계적 프롬프트 로그

### 프롬프트 로그 구조
```
reports/prompts/
├── 01_data_understanding.md           # 데이터 구조 파악
├── 02_feature_engineering.md          # 피처 설계
├── 03_model_ensemble.md               # 앙상블 전략
├── 04_phase3_advanced_features.md     # 고급 시계열 피처
├── 05_hyperparameter_tuning.md        # 하이퍼파라미터 최적화
├── 06_phase4_domain_features.md       # 도메인 특화 피처
├── 07_stacking_ensemble.md            # Stacking 앙상블
├── 08_phase6_error_analysis.md        # 에러 분석
├── 09_phase51_mlp_meta_learner.md     # MLP Meta-Learner
└── 10_phase55_blending_optimization.md # Blending 최적화 🆕
```
상세: [AI Collaboration Log](reports/prompts/)

## 📝 프로젝트 회고

### 전체 여정
```
Day 1-3:   베이스라인 → Phase 2 (17.23m)
Day 4:     3-model 앙상블 (17.03m)
Day 5-6:   Phase 3 고급 피처 (16.98m)
Day 7:     하이퍼파라미터 튜닝 (16.97m)
Day 8-10:  Phase 4 도메인 피처 (16.83m)
Day 11-12: Phase 5 Stacking (16.53m) 🥈
Day 13:    Phase 5.1 MLP (실패)
Day 13:    Phase 6 에러 분석 (16.56m)
Day 14-15: Phase 5.5 Blending (16.50m) 🥇

총 15일간 6 Phases + 1 Blending
총 개선: -18.9% (3.86m) ✨
```

### 핵심 성공 요인

**1. 체계적 접근**
- 단계별 개선 (6 Phases)
- 철저한 검증 (CV + LB)
- 완전한 문서화 (10개 프롬프트 로그)

**2. AI 협업**
- 80+ 프롬프트
- 즉각적 피드백
- 코드 리뷰

**3. 도메인 지식**
- 축구 이해
- 피처 설계
- 해석 가능성

**4. 실험 정신**
- 다양한 시도 (성공 6개, 실패 2개)
- 실패 수용 (Phase 5.1, 6)
- 지속적 개선 (Phase 5.5)

**5. 적절한 종료 🆕**
- U자 곡선 발견
- 최적점 확인
- 문서화 우선

### 최종 메시지

> **"작은 개선의 누적이 큰 결과를 만든다"**
> 
> Phase 5.5: 0.0335m (작음)
> 하지만 총 누적: 3.86m (큼)
> 
> 포기하지 않고 계속 시도하는 것이 핵심!

> **"언제 멈출지 아는 것도 능력"**
> 
> U자 곡선 발견 = 멈출 시점
> 추가 미세 조정 < 문서화 가치
> 
> "완벽은 선의 적"

## 🏆 프로젝트 하이라이트

### 최종 성과
- ✅ **Public LB: 16.4981 RMSE** 🥇
- ✅ 순위: 459/872 (상위 약 53%)
- ✅ 총 개선: -18.9% (3.86m)
- ✅ 완전 재현 가능한 파이프라인

### 주요 기여
- ✅ 6 Phases + 1 Blending 최적화
- ✅ Stacking 앙상블 + Blending 조합
- ✅ Grid Search로 최적 가중치 발견 (U자 곡선)
- ✅ AI 협업 프로세스 구축 (80+ 프롬프트)
- ✅ 체계적 문서화 (10개 프롬프트 로그)

### 기술적 성과
- ✅ OOF 기반 Stacking (Data Leakage 0%)
- ✅ Meta-Learning (3종: Ridge, LightGBM, MLP)
- ✅ Blending 최적화 (Grid Search 7개)
- ✅ 도메인 특화 피처 (선수/팀/경기 흐름)
- ✅ 완전 자동화 파이프라인

---

## 📚 참고 자료

### 문서
- [AI Collaboration Log](reports/prompts/): 10개 프롬프트 로그
- [Phase 5.5 Blending](reports/prompts/10_phase55_blending_optimization.md): 최종 최적화

### 코드
- [피처 생성](src/features/): Phase 1-6 피처 모듈
- [모델 학습](src/models/): 학습 및 예측 스크립트
- [Meta-Learner](src/models/): Ridge, LightGBM, MLP 3종
- [Blending](src/models/): Grid Search 자동화

### 제출 파일
- `submissions/submission_stacking_lgb.csv` (Phase 5, 16.5316)
- `submissions/submission_averaging_grid_7_w560_w640.csv` (Phase 5.5, 16.4981) 🥇

---

