# 프롬프트 로그 #02: 피처 엔지니어링 전략

## 날짜
2025-12-11

## 목적
좌표 예측 모델을 위한 효과적인 피처 설계

## 현재까지 파악한 정보

### 데이터 구조
- 356,721개 패스 이벤트
- 좌표: 105m × 68m 표준 경기장
- Episode 기반 시퀀스 구조

### 주요 발견
1. **공간적 패턴**
   - **Start X 분포**: 정규분포 형태, 중앙(50m) 피크, 양 골문 근처 적음
   - **Start Y 분포**: 3개 봉우리 (좌측면 0-10m, 중앙 30-38m, 우측면 58-68m)
   - **End X 분포**: Start X와 유사하나 약간 오른쪽(공격 방향) 이동
   - **End Y 분포**: Start Y와 거의 동일 (Y축은 공격 방향 아님)
   - **패스 방향성**: 전방향 우세, 대각선 패스 빈번, 중앙선 부근 활발

2. **액션 타입** (action_type_distribution.png 결과)
   - 가장 많은 type_name: **Pass (약 190,000회, 53%)**
   - 두 번째: **Carry (약 80,000회, 22%)**
   - 세 번째: **Recovery (약 30,000회, 8%)**
   - 나머지: Interception, Duel, Tackle, Throw-In 등 수비/세트피스 액션

3. **Episode 구조 추정**
   - 단순히 Pass만 있는 게 아니라 전체 플레이 시퀀스 포함
   - "최종 패스"는 특정 조건(episode 마지막, 또는 결과적 패스)을 만족하는 서브셋으로 추정

## 사용한 프롬프트

### 프롬프트 2-1: 기본 피처 설계
```
K리그 패스 좌표 예측을 위한 피처 엔지니어링을 설계하고 있어.

**확인된 데이터 특성:**
- 좌표계: (0,0) 왼쪽 하단 코너, (105, 68) 오른쪽 상단
- 중앙선: x=52.5, 페널티 박스: x<16.5 또는 x>88.5
- 공격 방향: X축 증가 방향 (0→105)

**Episode 구조 (중요!):**
- 평균 49개 액션/에피소드
- Pass(53%) + Carry(22%) + Recovery(8%) 등 혼합
- Pass-Carry 교차 패턴: Carry의 end = 다음 Pass의 start
- Unsuccessful Pass 후 → Interception → 소유권 전환
- "최종 패스" = Episode 마지막 Pass (예측 타겟)

**시각화 결과 인사이트:**
- Start X: 정규분포, 중앙(50m) 피크
- Start Y: 3개 봉우리 (좌측/중앙/우측면)
- End X: Start X보다 약간 오른쪽(공격 방향)
- 패스 방향: 전방향 우세, 대각선 패턴

**사용 가능한 원본 피처:**
- start_x, start_y (출발 좌표)
- time_seconds, period_id (시간)
- team_id, player_id, is_home (주체)
- type_name, result_name (액션 정보)
- episode_id, game_episode (시퀀스 ID)

**목표:** end_x, end_y 예측 (Episode 내 최종 Pass의 도착 좌표)

**질문:**

1. **좌표 기반 피처** - 즉시 만들 수 있는 것들
   - 거리, 각도, 골대까지 거리 등
   - 경기장 구역 분류 (수비/중원/공격)
   - Python 구현 코드 포함해서 알려줘

2. **Episode 시계열 피처** - 이전 액션과의 관계
   - 직전 N개 Pass의 방향 벡터
   - 패스 속도 (거리/시간 간격)
   - Episode 진행률 (시작 후 몇 번째 액션)
   - pandas groupby 활용 구현 방법

3. **전술적 피처** - 축구 도메인 지식 활용
   - 페널티 박스 진입 여부
   - 측면/중앙 공격 분류
   - 압박 상황 추정 (좁은 공간 vs 여유 공간)
   
4. **우선순위 제안**
   - 베이스라인 모델용 필수 피처 5개
   - 그 다음 단계 피처 5개
   - 구현 순서와 이유

각 피처에 대해:
- 왜 효과적일지 전술적 근거
- Python 코드 예시
- 예상되는 이슈나 주의사항
```

### Claude 응답
```
1. 좌표 기반 피처 (Baseline 필수, 기본 지리적 피처) 
A. 전술적 근거:
    - dist_to_target_goal: 골 찬스와 직접 연관, 짧을수록 최종 패스 가능성 ↑
    - in_penalty_box: 가장 위험한 구역, 슈팅 직전 패스 발생
    - zone_combined: 구역별 패스 패턴 학습 (예: 좌측 공격 시 크로스 경향)
    - dist_to_touchline: 압박 회피 가능성 (사이드라인 가까우면 선택지 제한)
B. 주의사항:
    - is_home 반드시 확인 (공격 방향 반대)
    - 구역 경계값은 도메인 지식으로 조정 가능

2. Episode 시계열 피처(이전 액션과의 관계)
A. 전술적 근거:
    - episode_progress: 초반(빌드업) vs 후반(마무리) 패턴 차이
    - time_since_prev_action: 빠른 역습(짧음) vs 느린 빌드업(김)
    - recent_3_forward_dist: 공격 모멘텀 측정
    - cumulative_pass_count: 소유권 안정성 (많을수록 조직적 공격)
B. 주의사항:
    - 미래 정보 누수: episode_total_actions는 Train에서만 사용
    - Test에서는 현재까지의 액션만 알 수 있음
    - 롤링 통계는 min_periods=1 설정 필수

3. 전술적 피처(축구 도메인 지식 활용)
A. 전술적 근거:
    - goal_angle: 슈팅 확률과 직결, 각도 클수록 득점 가능성 ↑
    - is_wide_attack + in_crossing_zone: 측면 공격 → 크로스 → 중앙 패스 패턴
    - in_tight_space: 압박 상황 → 짧고 안전한 패스 경향
    - is_switch_play: 사이드 체인지 → 반대편 공간 활용

4. 우선 순위 제안
A. 베이스라인 필수 피처 (Top 5) : 예상 성능: RMSE ~8-10m (경기장 대비 약 10%)
    - start_x, start_y: 위치가 목적지의 90% 결정 (짧은 패스 경향)
    - dist_to_target_goal: 공격 완성도 직접 지표
    - zone_combined: 구역별 전술 패턴 학습 (9개 클러스터)
    - in_final_third: 최종 패스 vs 중간 패스 구분
    - episode_progress: 초반 빌드업 vs 후반 마무리
B. 다음 단계 피처 (Top 5) : 예상 성능 향상: RMSE ~6-8m (추가 20% 개선)
    - prev_end_x/y: Pass-Carry 패턴 활용 (직접 연결)
    - time_since_prev_action: 빠른 역습 vs 느린 공격 구분
    - cumulative_pass_count: 조직적 공격 vs 단순 공격
    - goal_angle: 슈팅 기회 정량화
    - in_shooting_zone: 최종 패스 가능성 ↑↑
```

### 추가 검증 결과 (2025-12-11)

**Episode 구조 확정:**
- ✅ Episode 마지막 액션은 **100% Pass**
- ✅ 최종 패스 성공률: 56% (실패 44%)
- ✅ 전체 Pass 중 최종 패스 비율: 8.6%

**Test 데이터 구조:**
- 2,414개 episode (예측 대상)
- 각 episode의 시퀀스 데이터는 `path` 컬럼의 CSV 파일
- 목표: 각 episode의 마지막 Pass의 end_x, end_y 예측

**모델링 전략:**
- 최종 패스는 일반 패스보다 실패율 높음 (위험한 시도)
- Episode 내 맥락(이전 액션들)이 중요
- Test 시에는 실시간으로 시퀀스 데이터 읽어야 함

### 채택한 피처 리스트

#### Phase 1: 베이스라인 모델 (2025-12-11 구현)
- [x] start_x, start_y (원본 좌표)
- [x] dist_to_target_goal (골대까지 거리)
- [x] zone_combined (9구역 분류)
- [x] in_final_third (최종 3구역 여부)
- [x] episode_progress (에피소드 내 위치)
**목표 성능:** RMSE < 10m

#### Phase 2: 시계열 피처 추가 (2025-12-12 예정)
- [ ] prev_end_x, prev_end_y
- [ ] time_since_prev_action
- [ ] cumulative_pass_count
- [ ] goal_angle
- [ ] in_shooting_zone
**목표 성능:** RMSE < 8m

### 구현 계획
**2025-12-11:**
1. ✅ 데이터 구조 파악 완료
2. ✅ 프롬프트 설계 완료
3. ✅ Phase 1 피처 구현 시작

**2025-12-12:**
1. Phase 1 피처 완성 + 베이스라인 모델
2. Phase 2 피처 설계
3. 첫 제출 (Public LB)

**Day 4:**
1. 오차 분석
2. 피처 개선
3. 모델 튜닝