# K리그 최종 패스 좌표 예측 🎯⚽

> K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발

## 📌 프로젝트 개요
- **목표**: K리그의 실제 경기 데이터를 기반으로, 단편적인 이벤트의 나열을 넘어 특정 상황의 맥락을 AI가 학습하고, 이어지는 패스가 도달할 최적의 위치를 예측하는 것. 나아가 이를 통해 데이터 기반의 선수 평가 및 전술 분석에 대한 새로운 가능성을 발굴하고자 함.
- **기간**: 2025.12.10 ~ 2026.01.12
- **역할**: 데이터 분석, 모델링, AI 도구 활용 전략 수립
- **성과**: Public LB **16.8272** (상위 약 54%, 374/687)

## 🛠️ 기술 스택
- **언어**: Python 3.10
- **라이브러리**: pandas, numpy, scikit-learn
- **모델**: XGBoost, LightGBM, CatBoost
- **기법**: 시계열 피처 엔지니어링, 도메인 특화 피처, 앙상블 모델링, K-Fold CV

## 📂 프로젝트 구조
```
kleague-pass-prediction/
├── data/
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 데이터
│       ├── train_final_passes_v2.csv  # Phase 2
│       ├── train_final_passes_v3.csv  # Phase 3
│       └── train_final_passes_v4.csv  # Phase 4
├── models/                     # 학습된 모델 (.pkl)
│   ├── baseline_model_v4.pkl
│   ├── lgb_model_v4.pkl
│   └── catboost_model_v4.pkl
├── src/
│   ├── features/               # 피처 생성 모듈
│   │   ├── build_feature.py    # Phase 1, 2
│   │   └── advanced_features.py # Phase 3
│   └── models/                 # 모델 학습/예측
│       ├── train_model_v4.py
│       ├── train_model_lgb_v4.py
│       ├── train_model_catboost_v4.py
│       └── predict_ensemble_v4.py
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
# 개별 모델 학습 (Phase 4)
python src/models/train_model_v4.py          # XGBoost
python src/models/train_model_lgb_v4.py      # LightGBM
python src/models/train_model_catboost_v4.py # CatBoost
```

### 앙상블 예측
```bash
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
| **Phase 4 (Domain)** | **18.70m** | **16.8272** | **-0.9%** ✨ | **도메인 특화** |

### 개별 모델 성능
| 모델 | Phase 2 (v2) | Phase 3 (v3) | v3 튜닝 | **Phase 4 (v4)** | v3 대비 개선 |
|------|--------------|--------------|---------|------------------|--------------|
| **XGBoost** | 18.88m | 18.91m | 18.87m | **18.73m** | -0.18m ✅ |
| **LightGBM** | 18.81m | 18.82m | 18.81m | **18.64m** | -0.18m ✅ |
| **CatBoost** | 18.97m | 18.82m | 18.82m | **18.73m** | -0.09m ✅ |
| **평균** | 18.89m | 18.85m | 18.83m | **18.70m** | **-0.15m** |

**핵심 발견**: 도메인 특화 피처가 모든 모델에서 일관되게 성능 향상 ✨

### 공모전 제출
- **Public LB**: 16.8272 RMSE
- **순위**: 374/687 (상위 약 54%)
- **일반화 성능**: 베이스라인 대비 약 -17.4% 개선 

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

#### Phase 4: 도메인 특화 피처 (9개) ⭐ NEW!
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
**효과**: CV 18.70m (**-0.8%** 추가 개선) ✨

**핵심 인사이트**:
- 선수별 스타일이 패스 좌표 예측에 핵심적
- 경기 진행률(시간)이 패스 패턴 결정
- 득점차가 공격 성향에 영향

## 📈 피처 중요도 분석

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
- 도메인 지식 기반 피처가 실제로 효과적임을 입증 ✨

## 🤖 AI 협업 전략
### Claude 활용 방법
1. **피처 아이디어 생성**: 40+ 프롬프트
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
├── 05_hyperparameter_tuning.md # 하이퍼파라미터 & 가중치 최적화
└── 06_phase4_domain_features.md # 도메인 특화 피처 ⭐ NEW!
```
상세: [AI Collaboration Log](reports/prompts/)

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
1. **시계열 피처의 중요성**: 7개 피처로 7.3% 성능 개선
2. **과적합 방지**: CV 18.88 vs LB 17.23 (일반화 우수)
3. **파이프라인 구축**: 재현 가능한 ML 워크플로우
4. **AI 활용 전략**: 체계적 프롬프트 엔지니어링

### Day 4 (2025-12-14)
- ✅ CatBoost 모델 개발 (CV RMSE 18.97m)
- ✅ 3-model 앙상블 구축
- ✅ LB 점수 개선 (17.13 → 17.03)
- ✅ 가중치 실험 및 최적화

#### 주요 학습
1. **모델 다양성의 중요성**
   - 각 모델이 완전히 다른 패턴 학습
   - 앙상블로 장점 결합

2. **가중치 최적화**
   - [0.2, 0.4, 0.4]가 최적
   - XGBoost 감소가 효과적

### Day 5-6 (2025-12-15 ~ 12-16)
- ✅ Phase 3 피처 23개 생성
- ✅ 극단값 문제 발견 및 수정
- ✅ 피처 선택 (23개 → 6개)
- ✅ 3개 모델 v3 학습 완료
- ✅ 앙상블 LB 16.98m 달성

#### 주요 학습
1. **피처 품질 > 피처 수량**
   - 23개 전체: 악화
   - 6개 선별: 회복

2. **극단값 처리의 중요성**
   - pass_velocity: 853 → 40 m/s
   - 데이터 품질 = 모델 품질

### Day 7 (2025-12-17 ~ 12-22)
- ✅ 하이퍼파라미터 수동 튜닝
- ✅ Optuna 자동 튜닝 (LightGBM)
- ✅ 앙상블 가중치 실험 (4개 조합)
- ✅ 최종 결론: 기존 설정 유지

#### 주요 학습
1. **개별 성능 ≠ 앙상블 기여도**
   - Optuna LGB(18.76m)가 개별 최고
   - 앙상블에서는 기존 LGB와 동일
   - 다양성 > 개별 우수성

2. **하이퍼파라미터 튜닝의 한계**
   - 수동: -0.02m 개선
   - Optuna: -0.05m 개선 (개별)
   - 앙상블: ±0.00m
   - → 피처 개발이 더 중요

### Day 8-10 (2025-12-23 ~ 12-28) ⭐ NEW!
- ✅ Phase 4 피처 설계 (도메인 특화)
- ✅ 선수별 통계 계산 (경기별 누적)
- ✅ 팀별 통계 계산
- ✅ 경기 흐름 피처 생성
- ✅ 3개 모델 v4 재학습
- ✅ 앙상블 LB **16.8272** 달성 (최고 기록!)

#### 핵심 학습
1. **도메인 지식의 가치**
   - 선수 스타일이 예측에 핵심적
   - 경기 흐름(시간, 득점차)이 중요
   - CV -0.15m, LB -0.14m 개선

2. **누적 통계의 효과**
   - 경기별로 이전 경기 통계 사용
   - 시간에 따라 누적 → 현실적
   - 신규 선수는 전체 평균 활용

3. **Phase 4 피처 효과**
   - player_avg_pass_distance: 3개 모델 모두 Top 5
   - match_period_normalized: 경기 흐름 반영
   - team_attack_style: 팀 전술 특성

4. **점진적 개선의 누적**
   ```
   Phase 1: 20.36m (베이스라인)
   Phase 2: 18.88m (-7.3%)
   Phase 3: 18.85m (-0.2%)
   Phase 4: 18.70m (-0.8%) ← 누적 -8.2%!
   ```

5. **실용적 데이터 처리**
   - is_late_game: 후반 75분 이후 데이터 없음 (정상)
   - 최종 패스는 대부분 경기 초중반 발생
   - 이론과 현실의 차이 이해

상세 내용: [reports/prompts/06_phase4_domain_features.md](reports/prompts/06_phase4_domain_features.md)

## 📝 종료 후 회고
(프로젝트 종료 후 작성)

## 🎯 다음 단계 (고려중)

### 단기
- [ ] Phase 5 피처 실험
  - 상대 팀 압박 강도
  - 경기 상황별 가중치
  - 포지션별 특성

### 중기
- [ ] Stacking 앙상블
- [ ] SHAP values 분석
- [ ] 예측 실패 케이스 분석

### 장기
- [ ] 딥러닝 모델 (LSTM, Transformer)
- [ ] 실시간 예측 시스템
- [ ] 도메인 전문가 피드백

---

**⭐ 프로젝트 하이라이트**
```
시작: CV 20.36m
현재: CV 18.70m, LB 16.8272

총 개선: -17.4% (3.53m)
핵심 기법: 시계열 + 도메인 특화 피처 + 앙상블
```

**🏆 주요 성과**
- 체계적 피처 엔지니어링 (4 phases, 30개 피처)
- 3-model 앙상블 최적화
- AI 기반 개발 프로세스 구축
- 완전 재현 가능한 파이프라인