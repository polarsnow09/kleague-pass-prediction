# 프롬프트 로그 #07: Phase 5 - Stacking 앙상블

## 날짜
2025-12-29 ~ 2025-12-30

## 목적
Phase 4 이후 추가 성능 향상을 위한 **Stacking 앙상블** 구현

---

## 배경

### Phase 4 현황
```
XGBoost v4:  18.73m
LightGBM v4: 18.64m
CatBoost v4: 18.73m
앙상블 v4:   16.8272m (LB)
가중치:      [0.2, 0.4, 0.4]
```

### 목표
- Stacking 앙상블로 비선형 조합 구현
- LB < 16.7m 목표
- Meta-Learning 기법 적용

---

## 프롬프트 1: 다음 방향성 제안

### 입력
```
Phase 4까지 완료했습니다:
- Phase 1: 위치 피처
- Phase 2: 시계열 피처
- Phase 3: 고급 시계열
- Phase 4: 도메인 특화 피처
- 현재 LB: 16.8272m

앙상블 예측까지 진행하여 예측 모델의 성능을 한단계 상승시킨 
이후 아직 다음 방향성을 결정하지 않았습니다.

이 이후의 여정에 함께 해줄래?
```

### Claude 응답 요약
```
다음 방향성 제안 (4가지):

1. Phase 5 피처 개발 (점진적 개선)
   - 상대 팀 압박 피처
   - 선수 포지션 정보
   - 예상 개선: 0.1-0.2m

2. Stacking 앙상블 (구조적 개선) ⭐ 추천
   - Meta-learner 추가
   - OOF 예측 활용
   - 예상 개선: 0.2-0.3m

3. 딥러닝 모델 추가 (새로운 접근)
   - LSTM/Transformer
   - 예상 개선: 0.1-0.4m (불확실성 높음)

4. 예측 실패 케이스 분석 (디버깅)
   - 오차 큰 케이스 분석
   - 예상 개선: 0.1-0.2m

추천 이유:
- Stacking이 가장 확실함
- 기존 모델 활용 (리스크 낮음)
- 검증된 방법
```

---

## 프롬프트 2: Stacking 구현 계획

### 입력
```
Stacking 앙상블을 진행한다는 말씀이지요?
그 첫 시작으로 이미 학습된 3개의 모델을 가지고, 
예측된 결과를 결합하여 최종 예측을 만드는 과정을 
거치게 되는거고요?

좋습니다. 추천해주신 방향으로 진행해보죠.
```

### Claude 응답 요약
```
Stacking 구현 계획 (3단계):

Phase 1: Out-of-Fold (OOF) 예측 생성
- 5-Fold CV로 OOF 예측
- Data Leakage 완전 방지
- 전체 Train에 대한 예측값

Phase 2: Meta-Learner 학습
- Input: 3개 모델의 OOF 예측 (6개 피처)
- Output: 실제 타겟 (end_x, end_y)
- Meta-Model 후보:
  1. Ridge Regression (선형, 안전)
  2. LightGBM (비선형, 강력)

Phase 3: Test 예측
- Base 모델로 Test 예측
- Meta-Model로 최종 조합

예상 시나리오:
- 낙관적: 16.65m (-0.18m)
- 현실적: 16.75m (-0.08m)
- 비관적: 16.85m (+0.02m)
```

---

## 프롬프트 3: OOF 생성 스크립트

### 입력
```
학습 데이터 경로: data/processed/train_final_passes_v4.csv
Phase 4까지의 데이터가 모두 포함되어 있습니다.

통합 스크립트로 진행하는 것이 좋다고 생각합니다.
바로 진행해주세요.
```

### Claude 응답 (핵심 코드)
```python
# generate_oof_predictions.py

# 5-Fold CV
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 각 Fold별로
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # Train/Val 분리
    X_train, X_val = X[train_idx], X[val_idx]
    
    # 모델 학습
    model_x.fit(X_train, y_train_x)
    
    # Val 예측 (OOF)
    oof_predictions['xgb']['x'][val_idx] = model_x.predict(X_val)

# 결과: 전체 Train에 대한 예측 (과적합 없음)
```

### 실행 결과
```
OOF 성능:
- XGBoost : 13.40m
- LightGBM: 13.36m
- CatBoost: 13.30m ⭐ (최고)
- 평균    : 13.35m

저장: data/processed/oof_predictions.csv
```

**핵심 발견:**
- OOF가 전체 학습(18.7m)보다 훨씬 낮음
- 이유: OOF는 과적합 없는 순수 일반화 성능
- CV와 LB 차이의 핵심 원인 파악!

---

## 프롬프트 4: Meta-Learner 학습

### 입력
```
OOF 생성이 성공적으로 완료되었습니다.
결과에 문제가 없다면 바로 Meta-Learner 학습을 진행해주세요.
```

### Claude 응답 (핵심 구조)
```python
# train_meta_learner.py

# Meta-Features: 6개 (3 models × 2 coordinates)
X_meta = oof_df[['xgb_pred_x', 'xgb_pred_y',
                 'lgb_pred_x', 'lgb_pred_y',
                 'cat_pred_x', 'cat_pred_y']].values

# 1. Ridge Regression
ridge_x = Ridge(alpha=best_alpha)
ridge_x.fit(X_meta, y_true_x)

# 2. LightGBM (비교용)
lgb_x = LGBMRegressor(num_leaves=15, n_estimators=100)
lgb_x.fit(X_meta, y_true_x)
```

### 실행 결과
```
Meta-Learner 비교:
- Ridge    : 13.19m
- LightGBM : 12.84m (-0.35m, 2.6% 개선) ⭐

선택: LightGBM Meta-Learner

피처 중요도 (end_x):
1. cat_pred_x (365)  ← CatBoost가 가장 중요
2. xgb_pred_x (342)
3. lgb_pred_x (250)
4. lgb_pred_y (168)  ← 교차 좌표도 활용!

상관관계:
- 같은 좌표 간: 0.98+ (매우 높음)
- 하지만 Meta-Learner가 미묘한 차이 활용!
```

---

## 프롬프트 5: Stacking 예측 파이프라인

### 입력
```
Meta-Learner 학습이 완료되었습니다.
이제 최종 Stacking 예측 파이프라인을 만들어주세요.
```

### Claude 응답 (전체 흐름)
```python
# predict_stacking.py

# Step 1: Base 모델 전체 학습
xgb_model_x.fit(X_train, y_train_x)
lgb_model_x.fit(X_train, y_train_x)
cat_model_x.fit(X_train, y_train_x)

# Step 2: Meta-Learner 로드
meta_model_x = pickle.load('meta_lgb_x.pkl')
meta_model_y = pickle.load('meta_lgb_y.pkl')

# Step 3: Test Episode별 예측
for episode in test:
    # Phase 1-4 피처 생성
    episode_df = build_baseline_features(episode_df)
    episode_df = add_previous_action_features(episode_df)
    episode_df = build_phase3_features(episode_df)
    # Phase 4 통계 추가...
    
    # Base 모델 예측
    xgb_pred_x = xgb_model_x.predict(X_test)
    lgb_pred_x = lgb_model_x.predict(X_test)
    cat_pred_x = cat_model_x.predict(X_test)
    
    # Meta-Features 구성
    meta_features = [xgb_pred_x, xgb_pred_y,
                     lgb_pred_x, lgb_pred_y,
                     cat_pred_x, cat_pred_y]
    
    # Meta-Learner 최종 예측
    final_x = meta_model_x.predict(meta_features)
    final_y = meta_model_y.predict(meta_features)
```

### 구현 중 오류 및 해결

**오류 1: 범주형 피처**
```
ValueError: could not convert string to float: 'attacking'
```
**해결:**
```python
# LabelEncoder로 범주형 인코딩
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
```

**오류 2: LightGBM 파라미터**
```
TypeError: train() got an unexpected keyword argument 'verbose_eval'
```
**해결:**
```python
# verbose_eval 제거 (verbosity=-1로 대체)
lgb.train(params, train_data, num_boost_round=800)
```

**오류 3: Phase 3 피처 누락**
```
KeyError: ['rolling_mean_distance_3', ...] not in index
```
**해결:**
```python
# advanced_features 모듈 임포트
from features.advanced_features import build_phase3_features
episode_df = build_phase3_features(episode_df)
```

### 최종 실행 결과
```
✅ Base 모델 학습 완료 (약 5-7분)
✅ Test 예측 완료 (약 2-3분)
✅ 제출 파일 생성: submission_stacking_lgb.csv

예측 통계:
- end_x: 평균 52.3, std 28.4
- end_y: 평균 34.1, std 18.7
```

---

## 실험 결과

### Stacking 성능

**Public LB:**
```
Phase 4 앙상블: 16.8272m
Phase 5 Stacking: 16.5316m ⭐

개선: -0.30m (-1.8%) ✨
```

**순위:**
```
452/816 (상위 55.4%)
```

### 성능 분석

**CV vs LB 비교:**
```
                CV      LB      차이
Phase 4:     18.70m  16.83m  -1.87m
Phase 5:     12.84m  16.53m  -3.69m

→ Phase 5의 CV-LB 차이가 더 큼
→ OOF가 더 보수적 예측 (과적합 없음)
```

**앙상블 방식 비교:**
```
가중 평균 (Phase 4):
- 단순 선형 조합: 0.2×XGB + 0.4×LGB + 0.4×Cat
- 효과: 18.70m → 16.83m (-10%)

Stacking (Phase 5):
- 비선형 조합: Meta-Learner가 학습
- 효과: 12.84m (OOF) → 16.53m (LB)
- 추가 개선: -0.30m (-1.8%)
```

---

## 핵심 학습

### 1. OOF의 중요성

```
OOF RMSE: 13.35m
전체 학습: 18.70m

차이: 5.35m (28.6%!)

의미:
- OOF가 진짜 일반화 성능
- 전체 학습은 과적합 포함
- CV와 LB 차이의 원인 파악
```

**교훈:**
> "CV 성능은 OOF로 측정해야 정확하다"

### 2. Meta-Learning의 효과

```
Ridge (선형):     13.19m
LightGBM (비선형): 12.84m

개선: -0.35m (2.6%)

의미:
- Base 모델 간 비선형 상호작용 존재
- 단순 가중치로는 포착 불가
- Meta-Learner가 복잡한 패턴 학습
```

**교훈:**
> "비선형 조합이 선형보다 항상 우수하진 않지만, 
> 충분한 데이터가 있다면 시도할 가치 있음"

### 3. 모델 다양성의 재발견

```
Base 모델 상관관계: 0.98+
→ 거의 동일한 예측

하지만 Stacking: +0.30m 개선

의미:
- 미묘한 차이가 중요
- 각 모델의 강점이 다름
- 상황별로 최적 모델 다름
```

**예시:**
```
특정 상황:
- CatBoost가 정확한 경우
- LightGBM이 정확한 경우
- XGBoost가 정확한 경우

Meta-Learner가 이를 학습!
```

### 4. 교차 좌표의 중요성

```
LightGBM Meta-Model (end_x 예측):
4. lgb_pred_y (168) ← y 좌표 정보 사용!

의미:
- end_x 예측에 end_y 정보 유용
- 좌표 간 상관관계 존재
- 독립적이지 않음
```

**축구 도메인 해석:**
```
Y좌표(측면/중앙) → X좌표(전진 거리) 영향

예시:
- 측면 공격 → 크로스 → 중앙 골문 앞
- 중앙 빌드업 → 전진 패스 → 페널티 박스
```

### 5. 구조적 개선 vs 피처 개선

```
Phase 3-4: 수많은 피처 실험
- 고급 시계열 (23개 → 6개)
- 도메인 특화 (9개)
- 하이퍼파라미터 튜닝
총 개선: +0.18m

Phase 5: Stacking 앙상블
- 구조적 변경
- 코드 3개 (OOF, Meta, Predict)
개선: +0.30m ⭐

결론: 구조적 개선이 더 효과적!
```

**교훈:**
> "때로는 더 많은 피처보다 
> 더 좋은 구조가 중요하다"

### 6. 점진적 개선의 누적

```
Phase 1 → 2 → 3 → 4 → 5
20.36  18.88  18.85  18.70  16.53

각 단계: 작은 개선
누적:   -18.8% (3.83m) ✨

의미:
- 작은 개선도 누적되면 큼
- 포기하지 않는 것이 중요
- 다양한 접근 시도
```

---

## 기술적 성과

### 1. 완전 재현 가능한 파이프라인

```python
# 전체 프로세스
1. generate_oof_predictions.py  # OOF 생성
2. train_meta_learner.py        # Meta 학습
3. predict_stacking.py          # 최종 예측

→ 3개 스크립트로 완전 재현 가능!
```

### 2. Data Leakage 완전 방지

```
OOF 생성:
- Fold 1: Fold 2-5로 학습 → Fold 1 예측
- Fold 2: Fold 1,3-5로 학습 → Fold 2 예측
...

→ 각 샘플이 학습에 사용되지 않은 모델로 예측
→ Data Leakage 0%
```

### 3. Phase 4 통계 완벽 통합

```python
# Test 예측 시
- 선수 통계: 전체 Train 기반
- 팀 통계: 전체 Train 기반
- 경기 흐름: match_info 활용

→ Phase 4 피처 모두 사용
```

### 4. 범주형 피처 일관성

```python
# Train 시
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le  # 저장

# Test 시
for col in categorical_cols:
    le = label_encoders[col]  # 재사용
    test[col] = le.transform(test[col])
```

---

## 실패와 교훈

### 실패 1: 첫 실행 시 범주형 오류

**문제:**
```
ValueError: could not convert string to float: 'attacking'
```

**원인:**
- 범주형 피처(zone_x, zone_y 등)를 숫자로 변환 안함

**해결:**
- LabelEncoder 추가

**교훈:**
> "OOF 생성과 최종 예측 파이프라인의 
> 데이터 전처리가 동일해야 함"

### 실패 2: Phase 3 피처 누락

**문제:**
```
KeyError: 23개 Phase 3 피처 없음
```

**원인:**
- Test 예측 시 Phase 3 피처 생성 함수 호출 누락

**해결:**
```python
from features.advanced_features import build_phase3_features
episode_df = build_phase3_features(episode_df)
```

**교훈:**
> "피처 생성 파이프라인의 
> 일관성이 가장 중요하다"

---

## 다음 단계 (미구현)

### 단기
- [ ] 다른 Meta-Learner 실험
  - Neural Network (MLP)
  - XGBoost Meta-Model
  - 2-Level Stacking

- [ ] 앙상블 다양성 증대
  - 다른 하이퍼파라미터 조합
  - Feature Selection 다양화
  - Bagging Base Models

### 중기
- [ ] SHAP values 분석
  - Base 모델 해석
  - Meta-Learner 해석
  - 피처 중요도 변화 추적

- [ ] Blending 실험
  - OOF + Holdout 조합
  - 다양한 가중치 전략

### 장기
- [ ] AutoML 도입
  - H2O AutoML
  - TPOT
  - AutoGluon

- [ ] 딥러닝 Base Model
  - TabNet
  - LSTM + Tabular
  - Transformer 기반

---

## 최종 정리

### Phase 5 성과

**성능:**
```
Phase 4: 16.83m
Phase 5: 16.53m (-0.30m, -1.8%) ⭐

총 개선: 20.36m → 16.53m (-18.8%)
```

**순위:**
```
452/816 (상위 55.4%)
```

**기술적 달성:**
- ✅ OOF 예측 생성 (5-Fold CV)
- ✅ Meta-Learner 학습 (Ridge vs LightGBM)
- ✅ Stacking 파이프라인 구축
- ✅ Data Leakage 완전 방지
- ✅ 완전 재현 가능한 코드

---

## 회고

### 잘한 점

1. ✅ **체계적 접근**
   - OOF → Meta → Test 3단계
   - 각 단계 검증

2. ✅ **빠른 이터레이션**
   - 2일 내 완성
   - 즉각적 오류 수정

3. ✅ **데이터 기반 의사결정**
   - Ridge vs LightGBM 비교
   - 피처 중요도 분석

4. ✅ **완전한 문서화**
   - 프롬프트 로그
   - 코드 주석
   - README 업데이트

### 아쉬운 점

1. ⚠️ **더 많은 Meta-Model 실험 부족**
   - Neural Network 미시도
   - 2-Level Stacking 미시도

2. ⚠️ **앙상블 다양성 실험 부족**
   - 다른 하이퍼파라미터 조합
   - Feature Selection 다양화

3. ⚠️ **시간 제약**
   - 더 많은 실험 가능했을 것

### 배운 점

> **"구조적 개선이 피처 개선보다 효과적일 수 있다"**
> 
> Phase 3-4: 수많은 피처 → +0.18m
> Phase 5: Stacking → +0.30m
> 
> 때로는 더 좋은 방법론이 
> 더 많은 데이터보다 중요하다.

> **"OOF는 진짜 일반화 성능"**
> 
> OOF: 13.35m (과적합 없음)
> 전체: 18.70m (과적합 포함)
> 
> CV 평가는 OOF로 해야 정확하다.

> **"미묘한 차이가 중요하다"**
> 
> 상관계수 0.98+ (거의 동일)
> 하지만 +0.30m 개선
> 
> 다양성은 양이 아니라 질이다.

---

## 참고 자료

### 코드 파일
- `src/models/generate_oof_predictions.py`
- `src/models/train_meta_learner.py`
- `src/models/predict_stacking.py`

### 데이터 파일
- `data/processed/oof_predictions.csv`
- `models/meta_ridge_x.pkl`, `models/meta_ridge_y.pkl`
- `models/meta_lgb_x.pkl`, `models/meta_lgb_y.pkl`

### 제출 파일
- `submissions/submission_stacking_lgb.csv` (LB 16.5316)

---

## 프롬프트 통계

- 총 프롬프트 수: 10+
- Phase 5 관련: 5개 (핵심)
- 평균 응답 시간: 30초
- 채택률: 100%
- 오류 수정: 3회 (즉시 해결)

---

**🎊 Phase 5 완료!**

```
Phase 1 → 2 → 3 → 4 → 5
기본   시계열  고급  도메인  Stacking

20.36m → 18.88m → 18.85m → 18.70m → 16.53m

총 개선: -18.8% (3.83m) ✨
```

**최종 순위: 452/816 (상위 55%)**

---

## 프로젝트 종합 회고

### 전체 여정
```
Day 1-3:   베이스라인 → Phase 2 (17.23m)
Day 4:     3-model 앙상블 (17.03m)
Day 5-6:   Phase 3 고급 피처 (16.98m)
Day 7:     하이퍼파라미터 튜닝 (16.97m)
Day 8-10:  Phase 4 도메인 피처 (16.83m)
Day 11-12: Phase 5 Stacking (16.53m) ⭐

총 12일: -18.8% 개선
```

### 핵심 성공 요인

1. **체계적 접근**
   - 단계별 개선
   - 철저한 검증
   - 완전한 문서화

2. **AI 협업**
   - 50+ 프롬프트
   - 즉각적 피드백
   - 코드 리뷰

3. **도메인 지식**
   - 축구 이해
   - 피처 설계
   - 해석 가능성

4. **실험 정신**
   - 다양한 시도
   - 실패 수용
   - 지속적 개선

### 최종 메시지

> **"ML 프로젝트는 마라톤이다"**
> 
> 작은 개선의 누적이 큰 결과를 만든다.
> 포기하지 않고 계속 시도하는 것이 핵심이다.
> 
> 20.36m → 16.53m
> 단 12일 만에 -18.8% 개선
> 
> 여러분도 할 수 있습니다! 💪