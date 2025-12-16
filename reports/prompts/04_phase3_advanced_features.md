# 프롬프트 로그 #04: Phase 3 고급 피처 개발

## 날짜
2025-12-16

## 목적
Phase 2 대비 추가 성능 향상을 위한 고급 시계열 피처 개발

---

## 배경

### Phase 2 현황
```
XGBoost v2: 18.88m
LightGBM v2: 18.81m
CatBoost v2: 18.97m
앙상블 v2: 17.01m (LB)
```

### 목표
- 고급 시계열 피처로 CV < 18.5m 달성
- 앙상블 LB < 16.8m 목표

---

## 프롬프트 1: Phase 3 피처 설계

### 입력
```
현재 Phase 2까지 완료했습니다:
- Phase 1: 위치 기반 (8개)
- Phase 2: 시계열 (7개)
- CV: 18.88m

추가로 어떤 고급 시계열 피처를 만들면 좋을까요?
Episode 구조를 활용한 피처를 제안해주세요.
```

### Claude 응답 요약
```python
제안된 피처 (5개 카테고리, 23개):

1. 롤링 통계 (8개)
   - rolling_mean_distance_3/5
   - rolling_std_distance_3/5
   - rolling_mean_direction_x/y_3/5

2. 누적 통계 (4개)
   - cumulative_distance
   - cumulative_forward/lateral
   - forward_lateral_ratio

3. 속도 관련 (4개)
   - pass_velocity
   - avg_episode_velocity
   - velocity_change
   - recent_3_avg_velocity

4. 공간 활용 (4개)
   - episode_x_range/y_range
   - touchline_proximity
   - avg_touchline_proximity

5. 패턴 인식 (3개)
   - is_buildup (느린 빌드업)
   - is_counter (빠른 역습)
   - is_under_pressure (압박)
```

---

## 프롬프트 2: 피처 구현 및 검증

### 입력
```
Phase 3 피처 23개를 생성했습니다.
하지만 다음 문제가 발생했습니다:

1. pass_velocity: max 853 m/s (음속의 2.5배!)
2. forward_lateral_ratio: max 3,242,505
3. 극단값으로 인한 학습 불안정

어떻게 수정해야 할까요?
```

### Claude 응답 요약
```python
문제 1: 속도 계산
원인: time_since_prev가 0.1초 미만
해결: 최소 0.5초 보장 + 40 m/s 상한선

수정 전:
velocity = distance / (time + 0.1)

수정 후:
velocity = np.where(
    time >= 0.5,
    distance / time,
    distance / 0.5
)
velocity = np.clip(velocity, 0, 40)

문제 2: 비율 계산
원인: 분모(lateral)가 거의 0
해결: 최소 5m 보장 + ±20 제한

수정 전:
ratio = forward / (lateral + 1e-6)

수정 후:
ratio = np.where(
    lateral > 5.0,
    forward / lateral,
    np.sign(forward) * 5.0
)
ratio = np.clip(ratio, -20, 20)
```

### 검증 결과
```
수정 전:
- pass_velocity max: 853 m/s
- forward_lateral_ratio max: 3,242,505

수정 후:
- pass_velocity max: 40.0 m/s ✅
- forward_lateral_ratio max: 17.9 ✅
- 결측치: 0개
- Inf: 0개
```

---

## 프롬프트 3: 성능 평가 및 피처 선택

### 입력
```
Phase 3 피처(23개) 추가 후 학습 결과:

XGBoost v3: 18.99m (v2 대비 +0.11m 악화)
피처 중요도:
1. zone_x_encoded (75%)
2. start_x (9%)
3-10. Phase 3 피처 중 3개만 상위 10위

문제: 너무 많은 피처로 노이즈 증가?
해결 방법을 제안해주세요.
```

### Claude 응답 요약
```
원인 분석:
1. 피처 간 상관관계 (multicollinearity)
2. 신호 대 잡음 비율 감소
3. 샘플 수(15,435개) 대비 피처 과다

해결 방안: 피처 선택
- 피처 중요도 > 0.5%
- 상관관계 < 0.8
- 도메인 지식 활용

선별 결과 (6개):
1. pass_velocity (0.8%)
2. touchline_proximity (0.9%)
3. is_under_pressure (0.6%)
4. avg_episode_velocity
5. episode_x_range
6. rolling_mean_distance_3

기대 효과: 18.99m → 18.5-18.7m
```

### 실제 결과
```
XGBoost v3 (선별): 18.91m (-0.08m 개선)
→ v2 대비: +0.03m (거의 동일)
```

---

## 프롬프트 4: 3개 모델 종합 분석

### 입력
```
3개 모델 학습 완료:
- XGBoost v3: 18.91m (v2: 18.88m, +0.03m)
- LightGBM v3: 18.82m (v2: 18.81m, +0.01m)
- CatBoost v3: 18.82m (v2: 18.97m, -0.15m)

v3 채택 여부를 결정해야 합니다.
어떻게 판단해야 할까요?
```

### Claude 응답 요약
```
분석:
1. CatBoost가 Phase 3 피처를 가장 잘 활용
   - touchline_proximity 중요도 12.0
   - -0.15m 유일한 개선

2. XGBoost, LightGBM은 거의 동일
   - 통계적으로 유의미한 차이 아님

3. 평균 성능
   - v2: 18.89m
   - v3: 18.85m (-0.04m)

결정 기준:
- CatBoost 개선 명확 → v3 채택 권장
- 앙상블에서 효과 증폭 예상
- 최악의 경우도 v2와 비슷

추천: v3 전체 채택
```

### 최종 결정
```
✅ v3 채택
- 3개 모델 모두 v3 사용
- 가중치 [0.2, 0.4, 0.4] 유지
```

---

## 프롬프트 5: 앙상블 예측 파이프라인

### 입력
```
기존 predict_ensemble.py가 있습니다.
v3 모델을 사용하려면 어떻게 수정해야 하나요?
Phase 3 피처 생성 로직을 추가해야 합니다.
```

### Claude 응답 요약
```
수정 필요 사항 (3곳):

1. import 추가:
   from src.features.advanced_features import build_phase3_features

2. preprocess_episode 수정:
   def preprocess_episode(self, df):
       df = build_baseline_features(df)
       df = add_previous_action_features(df)
       df = build_phase3_features(df)  # 추가!
       return df

3. 모델 경로 변경:
   MODEL_PATHS = [
       'baseline_model_v3.pkl',
       'lgb_model_v3.pkl',
       'catboost_model_v3.pkl'
   ]

예상 결과:
- v2: 17.01m
- v3: 16.90-16.98m
```

### 실제 결과
```
✅ Public LB: 16.98155m
→ 예상 범위 내 달성!
→ v2 대비 -0.03m (0.18%) 개선
```

---

## 최종 성과

### CV 성능
```
모델별:
- XGBoost:  18.91m (+0.03m)
- LightGBM: 18.82m (+0.01m)
- CatBoost: 18.82m (-0.15m) ⭐
- 평균:     18.85m (-0.04m)
```

### 앙상블 효과
```
v2 앙상블: 18.89m → 17.01m (-1.88m, 9.9%)
v3 앙상블: 18.85m → 16.98m (-1.87m, 10.5%)

→ 앙상블 효과 유지!
```

### Public LB
```
v2: 17.01m
v3: 16.98m (-0.03m, 0.18%)

총 개선: 20.36m → 16.98m (-3.38m, 16.6%)
```

---

## 핵심 학습

### 1. 피처 품질 > 피처 수량
```
23개 피처 → 성능 악화 (노이즈)
6개 선별 → 성능 회복 (신호)

교훈: 무분별한 피처 추가는 역효과
```

### 2. 극단값 처리의 중요성
```
수정 전: max 853 m/s (비현실적)
수정 후: max 40 m/s (합리적)

→ 데이터 품질이 모델 성능 결정
```

### 3. 모델별 피처 활용도 차이
```
CatBoost: Phase 3 피처 효과적 활용
XGBoost:  zone 피처 압도적 의존
LightGBM: 균형잡힌 활용

→ 앙상블로 보완 효과
```

### 4. 점진적 개선의 가치
```
Phase 3 단독: -0.04m (미미)
앙상블 결합: -0.03m (LB)

→ 작은 개선도 누적되면 의미 있음
```

---

## 다음 단계 (미구현)

### 단기
- [ ] 하이퍼파라미터 튜닝 (Optuna)
- [ ] 추가 피처 실험
  - 경기 상황별 가중치
  - 선수 스타일 피처

### 중기
- [ ] Stacking 앙상블
- [ ] SHAP values 분석
- [ ] 예측 실패 케이스 분석

### 장기
- [ ] 딥러닝 모델 (LSTM, Transformer)
- [ ] 실시간 예측 시스템

---

## 참고 코드

### advanced_features.py 핵심 로직
```python
def add_velocity_features(df):
    # 속도 계산 (극단값 처리)
    df['pass_velocity'] = np.where(
        df['time_since_prev'] >= 0.5,
        df['prev_action_distance'] / df['time_since_prev'],
        df['prev_action_distance'] / 0.5
    )
    df['pass_velocity'] = np.clip(df['pass_velocity'], 0, 40)
    return df

def add_cumulative_features(df):
    # 비율 계산 (0 나누기 방지)
    df['forward_lateral_ratio'] = np.where(
        df['cumulative_lateral'] > 5.0,
        df['cumulative_forward'] / df['cumulative_lateral'],
        np.sign(df['cumulative_forward']) * 5.0
    )
    df['forward_lateral_ratio'] = np.clip(df['forward_lateral_ratio'], -20, 20)
    return df
```

---

## 회고

### 잘한 점
1. ✅ 극단값 문제 즉시 감지 및 수정
2. ✅ 피처 선택으로 과적합 방지
3. ✅ 3개 모델 종합 분석 후 신중한 결정
4. ✅ 예상 범위 내 성과 달성

### 아쉬운 점
1. ⚠️ Phase 3 단독 효과는 미미 (0.04m)
2. ⚠️ 초기 23개 피처 생성 시 검증 부족
3. ⚠️ 하이퍼파라미터 튜닝 미실시

### 배운 점
> "더 많은 피처가 항상 더 좋은 것은 아니다. 
> 품질 높은 피처 선별이 핵심이다."

---

## 프롬프트 통계

- 총 프롬프트 수: 40+
- Phase 3 관련: 10+
- 평균 응답 시간: 30초
- 채택률: 85%
