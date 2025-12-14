# K리그 최종 패스 좌표 예측 🎯⚽

> K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발

## 📌 프로젝트 개요
- **목표**: K리그의 실제 경기 데이터를 기반으로, 단편적인 이벤트의 나열을 넘어 특정 상황의 맥락을 AI가 학습하고, 이어지는 패스가 도달할 최적의 위치를 예측하는 것. 나아가 이를 통해 데이터 기반의 선수 평가 및 전술 분석에 대한 새로운 가능성을 발굴하고자 함.
- **기간**: 2025.12.10 ~ 2026.01.12
- **역할**: 데이터 분석, 모델링, AI 도구 활용 전략 수립
- **성과**: (추후 작성)

## 🛠️ 기술 스택
- Python 3.10
- pandas, numpy, scikit-learn
- (추가 예정)

## 📂 프로젝트 구조
```
kleague-pass-prediction/
├── data/                   # 데이터 폴더
├── notebooks/              # 분석 노트북
├── src/                    # 소스 코드
├── models/                 # 학습된 모델
├── reports/                # 분석 보고서
├── README.md
└── requirements.txt
```

## 🚀 실행 방법

### 모델 학습
```bash
python src/models/train_model.py
python src/models/train_model_lgb.py
```

### Test 예측
```bash
python src/models/predict_ensemble.py
```

## 📊 주요 결과

### 모델 성능
- **XGBoost**:
  - **v1 (Baseline)**: CV RMSE 20.36m
  - **v2 (Temporal)**: CV RMSE 18.88m (↓7.3%)
- **LightGBM**:  CV 18.81m 
- **CatBoost**:  CV 18.97m 
- **ensemble(XGB+LGB)** : 17.13392 
- **ensemble(XGB+LGB+CatBoost)** : 17.0292987272

### 공모전 제출
- **Public LB**: 17.0292987272 RMSE
- **순위**: 285/500 (상위 57%)
- **일반화 성능**: ensemble(XGB+LGB) 대비 ↓ 0.20 (1.2%)

### 피처 개발
- **Phase 1** (8개): 위치 기반 피처
  - start_x, start_y, dist_to_target_goal
  - zone encoding (9 zones)
  - penalty_box, final_third
  
- **Phase 2** (7개): 시계열 피처
  - prev_end_x/y (이전 액션 위치)
  - prev_action_distance, time_since_prev
  - prev_direction_x/y (공격 방향)
  - pass_count_in_episode

## 🤖 AI 협업 전략

이 프로젝트는 Claude를 활용하여:
- 30+ 프롬프트로 피처 아이디어 도출
- 체계적 프롬프트 로그 작성 (reports/prompts/)
- 코드 리뷰 및 디버깅
- 실무 수준의 프로젝트 구조 설계

상세: [AI Collaboration Log](reports/prompts/02_feature_engineering.md)

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

### 다음 단계
- [ ] 고급 시계열 피처
- [ ] 하이퍼파라미터 튜닝

상세 내용: [reports/prompts/README.md](reports/prompts/README.md)

## 📝 종료 후 회고
(프로젝트 종료 후 작성)