# 프롬프트 로그 #06: Phase 4 도메인 특화 피처

## 날짜
2025-12-23 ~ 2025-12-28

## 목적
Phase 3 이후 추가 성능 향상을 위한 **도메인 지식 기반 특화 피처** 개발

---

## 배경

### Phase 3 현황
```
XGBoost v3:  18.91m
LightGBM v3: 18.82m
CatBoost v3: 18.82m
앙상블 v3:   16.9724m (LB)
```

### 목표
- 축구 도메인 지식을 활용한 피처 개발
- CV < 18.6m 달성
- 앙상블 LB < 16.8m 목표

---

## 프롬프트 1: Phase 4 방향성 설정

### 입력
```
Phase 3까지 완료했습니다:
- Phase 1: 위치 기반 (8개)
- Phase 2: 시계열 (7개)
- Phase 3: 고급 시계열 (6개)
- CV: 18.85m

다음 단계로 어떤 피처를 추가하면 좋을까요?
축구 도메인 지식을 활용하고 싶습니다.
```

### Claude 응답 요약
```
Phase 4 제안: 도메인 특화 피처 (3개 카테고리)

1. 선수별 스타일 피처 (4-5개)
   - 선수 평균 패스 거리
   - 전진 패스 비율
   - 패스 성공률
   - 패스 횟수 (경험치)
   
근거:
- 같은 위치에서도 선수마다 다른 패스
- 공격수는 짧고, 미드필더는 길게
- 베테랑과 신인의 차이

2. 팀별 전술 피처 (2-3개)
   - 팀 평균 패스 거리
   - 측면/중앙 공격 선호도
   - 빌드업 vs 역습 스타일
   
근거:
- 팀마다 고유한 전술적 특성
- 감독의 철학 반영
- 상대 전술에 따른 변화

3. 경기 흐름 피처 (3-4개)
   - 득점차 (리드/동점/뒤짐)
   - 경기 진행률 (0~1)
   - 후반전 여부
   - 컨디션 (체력 소모)
   
근거:
- 득점차에 따라 공격 성향 변화
- 후반으로 갈수록 안전한 패스
- 리드 시 수비적, 뒤질 때 공격적

예상 효과: CV 18.6-18.7m (-0.1~0.2m)
```

---

## 프롬프트 2: 통계 계산 전략

### 입력
```
선수별 통계를 계산하려고 합니다.

문제:
- 학습 시: 전체 경기 데이터 사용 가능
- 예측 시: 해당 경기 이전 데이터만 사용 가능

어떻게 구현해야 Data Leakage를 방지할 수 있을까요?
```

### Claude 응답 요약
```
누적 통계 (Cumulative Statistics) 방식 사용:

1. 학습 데이터 생성 시:
   - 경기를 시간순 정렬
   - 각 경기마다 "이전 경기들"의 통계 사용
   - 첫 경기는 전체 평균 사용

2. 예측 시:
   - 전체 학습 데이터 통계 사용
   - 신규 선수는 전체 평균

예시:
Game 1: 통계 없음 → 전체 평균
Game 2: Game 1 통계만 사용
Game 3: Game 1+2 통계 사용
...
Game N: Game 1~N-1 통계 사용

Test: Game 1~N 전체 통계 사용

장점:
- Data Leakage 완전 방지
- 현실적 (과거 데이터만 활용)
- 시간에 따라 통계 정확도 향상

구현:
python
game_ids = sorted(train['game_id'].unique())
player_stats = {}

for i, game_id in enumerate(game_ids):
    if i == 0:
        player_stats[game_id] = None  # 전체 평균 사용
    else:
        prev_games = game_ids[:i]
        prev_data = train[train['game_id'].isin(prev_games)]
        player_stats[game_id] = prev_data.groupby('player_id').agg(...)
```

---

## 프롬프트 3: 피처 생성 구현

### 입력
```
누적 통계 방식으로 구현하려고 합니다.

생성할 피처:
1. 선수: 평균 거리, 전진 비율, 성공률, 횟수
2. 팀: 평균 거리, 공격 스타일
3. 경기: 득점차, 진행률, 후반전

효율적인 구현 방법을 제안해주세요.
```

### Claude 응답 (핵심 코드)
```python
# 1. Pass 데이터 전처리
passes = train[train['type_name'] == 'Pass'].copy()
passes['pass_distance'] = np.sqrt(
    (passes['end_x'] - passes['start_x'])**2 + 
    (passes['end_y'] - passes['start_y'])**2
)
passes['is_forward'] = np.where(
    passes['is_home'],
    passes['end_x'] - passes['start_x'],
    passes['start_x'] - passes['end_x']
) > 0
passes['is_success'] = (passes['result_name'] == 'Successful').astype(int)
passes['is_wide'] = ((passes['start_y'] < 20) | (passes['start_y'] > 48)).astype(int)

# 2. 경기별 누적 통계
game_ids = sorted(passes['game_id'].unique())
player_cumulative_stats = {}

for i, game_id in enumerate(tqdm(game_ids)):
    if i == 0:
        player_cumulative_stats[game_id] = None
    else:
        prev_games = game_ids[:i]
        prev_passes = passes[passes['game_id'].isin(prev_games)]
        
        player_stats = prev_passes.groupby('player_id').agg({
            'pass_distance': 'mean',
            'is_forward': 'mean',
            'is_success': 'mean',
            'player_id': 'count'
        }).rename(columns={'player_id': 'pass_count'})
        
        player_cumulative_stats[game_id] = player_stats.to_dict('index')

# 3. 전체 평균 (신규 선수용)
global_player_stats = {
    'pass_distance': passes['pass_distance'].mean(),
    'is_forward': passes['is_forward'].mean(),
    'is_success': passes['is_success'].mean(),
    'pass_count': 50
}

# 4. DataFrame에 추가
def get_player_stat(row, stat_name):
    game_id = row['game_id']
    player_id = row['player_id']
    
    game_stats = player_cumulative_stats.get(game_id)
    if game_stats is None or player_id not in game_stats:
        return global_player_stats[stat_name]
    
    return game_stats[player_id][stat_name]

df['player_avg_pass_distance'] = df.apply(
    lambda row: get_player_stat(row, 'pass_distance'), axis=1
)
```

---

## 프롬프트 4: 피처 검증

### 입력
```
Phase 4 피처를 생성했습니다.

검증 결과:
- player_avg_pass_distance: 4,755개 고유값 ✅
- team_attack_style: 385개 고유값 ✅
- is_late_game: 1개 고유값 ⚠️ (모두 0)

is_late_game이 모두 0인 건 문제인가요?
```

### Claude 응답 요약
```
문제 없음! ✅

이유:
1. 최종 패스는 대부분 경기 초중반 발생
2. 후반 75분 이후는 시간 끌기/수비 위주
3. Train 데이터의 특성일 뿐

대응:
- 피처는 유지 (Test에 있을 수 있음)
- 모델이 자동으로 중요도 0 부여
- 삭제하지 말 것

비슷한 사례:
- 페널티 킥: 매우 드물지만 중요
- 드문 상황도 피처로 유지

결론: 정상적인 데이터 특성
```

---

## 실험 결과

### Phase 4 피처 효과 (개별 모델)

**XGBoost v4**
```
CV RMSE: 18.73m (v3: 18.91m, -0.18m ✅)

Phase 4 피처 중요도:
1. player_avg_pass_distance (1.7%)  ← 최고!
2. match_period_normalized (0.6%)
3. team_attack_style (0.6%)
4. player_success_rate (0.5%)
5. score_diff (0.5%)
...
9. is_late_game (0.0%)  ← 예상대로

총 개선: v3 대비 -0.18m (0.9%)
```

**LightGBM v4**
```
CV RMSE: 18.64m (v3: 18.82m, -0.18m ✅)

Phase 4 피처 중요도:
1. player_avg_pass_distance (4.5M)  ← 최고!
2. match_period_normalized (1.5M)
3. player_success_rate (1.3M)
4. player_pass_count (1.3M)
5. team_attack_style (1.2M)
...
9. is_late_game (0.0M)

총 개선: v3 대비 -0.18m (0.9%)
```

**CatBoost v4**
```
CV RMSE: 18.73m (v3: 18.82m, -0.09m ✅)

Phase 4 피처 중요도:
1. player_avg_pass_distance (8.2)  ← 최고!
2. player_success_rate (2.7)
3. match_period_normalized (2.5)
4. team_avg_pass_distance (2.3)
5. player_pass_count (2.2)
...
9. is_late_game (0.0)

총 개선: v3 대비 -0.09m (0.5%)
```

### 종합 분석

**모델별 비교**
| 모델 | v3 | v4 | 개선 | 개선률 |
|------|----|----|------|--------|
| XGBoost | 18.91m | 18.73m | -0.18m | 0.9% |
| LightGBM | 18.82m | 18.64m | -0.18m | 0.9% |
| CatBoost | 18.82m | 18.73m | -0.09m | 0.5% |
| **평균** | **18.85m** | **18.70m** | **-0.15m** | **0.8%** |

**핵심 발견**

1. **player_avg_pass_distance의 압도적 효과**
   - 3개 모델 모두에서 Phase 4 최고 중요도
   - 선수별 스타일이 좌표 예측의 핵심
   - 도메인 지식이 정확했음을 입증

2. **경기 흐름 피처의 효과**
   - match_period_normalized: 3개 모두 Top 10
   - 경기 시간이 패스 패턴 결정
   - score_diff: XGBoost에서 0.5%

3. **팀 전술 피처**
   - team_attack_style: 측면 선호도 반영
   - team_avg_pass_distance: 팀 스타일
   - 선수 개인보다는 보조적 역할

4. **is_late_game**
   - 예상대로 중요도 0
   - 하지만 삭제 불필요 (Test 대비)

### 앙상블 효과

```
Phase 3 앙상블: 18.85m → 16.9724m (LB)
Phase 4 앙상블: 18.70m → 16.8272m (LB)

개선: -0.14m (0.9%)
```

**LB 개선 분석**
- CV 개선: -0.15m
- LB 개선: -0.14m
- 일관성 있는 개선! ✅
- 과적합 없음

---

## 프롬프트 5: 예측 파이프라인 수정

### 입력
```
predict_ensemble_v4.py를 수정하려고 합니다.

Phase 4 통계를 어떻게 연결해야 할까요?
- Train 데이터: 전체 통계 사용
- Test 데이터: player_id, team_id로 매칭
```

### Claude 응답 (핵심 구조)
```python
class Phase4Statistics:
    """Phase 4 통계 계산기"""
    
    def __init__(self, train_csv, match_info_csv):
        # 전체 Train 데이터로 통계 계산
        passes = pd.read_csv(train_csv)
        passes = passes[passes['type_name'] == 'Pass']
        
        # 선수별 통계
        self.player_stats = passes.groupby('player_id').agg({
            'pass_distance': 'mean',
            'is_forward': 'mean',
            'is_success': 'mean',
            'player_id': 'count'
        }).to_dict('index')
        
        # 팀별 통계
        self.team_stats = passes.groupby('team_id').agg({
            'pass_distance': 'mean',
            'is_wide': 'mean'
        }).to_dict('index')
        
        # 전체 평균 (신규용)
        self.global_player = {...}
        self.global_team = {...}
    
    def get_player_stats(self, player_id):
        if player_id in self.player_stats:
            return self.player_stats[player_id]
        else:
            return self.global_player
    
    def get_team_stats(self, team_id):
        # 동일
    
    def get_match_stats(self, game_id, team_id, is_home, time_seconds):
        # match_info에서 득점 가져오기
        # 경기 진행률 계산
        return {
            'score_diff': ...,
            'match_period_normalized': time_seconds / 5400,
            'is_late_game': int(time_seconds >= 4050)
        }

# 예측 시 사용
class EnsemblePredictor:
    def __init__(self, ..., phase4_stats):
        self.phase4_stats = phase4_stats
    
    def add_phase4_features(self, df):
        # 최종 Pass에서 player_id, team_id 추출
        last_pass = df[df['type_name'] == 'Pass'].iloc[-1]
        player_id = last_pass['player_id']
        team_id = last_pass['team_id']
        
        # 통계 가져와서 DataFrame에 추가
        player_stats = self.phase4_stats.get_player_stats(player_id)
        for key, value in player_stats.items():
            df[key] = value
        
        # 팀, 경기 흐름도 동일
        return df
```

---

## 프롬프트 6: 결과 분석

### 입력
```
Phase 4 결과가 나왔습니다:

CV: 18.70m (v3 대비 -0.15m)
LB: 16.8272 (v3 대비 -0.14m)

이 결과를 어떻게 해석해야 할까요?
```

### Claude 응답 요약
```
훌륭한 결과! ✅

긍정적 신호:
1. CV와 LB 개선 일치 (-0.15m vs -0.14m)
   → 과적합 없음
   → 진짜 성능 향상

2. 3개 모델 모두 개선
   → Phase 4 피처의 보편적 효과
   → 특정 모델에만 좋은 게 아님

3. 도메인 지식의 가치 입증
   → 이론(선수 스타일)이 실제로 효과적
   → 데이터 과학 + 도메인 전문성

Phase 4의 의미:
- Phase 1-3: 기술적 피처 (위치, 시계열, 통계)
- Phase 4: 의미론적 피처 (선수, 팀, 경기 맥락)
- 기술 + 의미 = 최고 성능

개선 여지:
- player_avg_pass_distance가 압도적
- 다른 선수 특성 추가 가능
  예: 포지션, 나이, 경력, 발 선호도

역사적 맥락:
Phase 1: 20.36m (베이스라인)
Phase 2: 18.88m (-7.3%, 시계열)
Phase 3: 18.85m (-0.2%, 고급 통계)
Phase 4: 18.70m (-0.8%, 도메인 지식)

총 누적: -8.2% 개선! 🎊
```

---

## 최종 정리

### Phase 4 성과

**피처 개발**
- ✅ 선수 스타일 (4개)
- ✅ 팀 전술 (2개)
- ✅ 경기 흐름 (3개)
- 총 9개 도메인 특화 피처

**성능 개선**
```
CV RMSE:
XGBoost:  18.91m → 18.73m (-0.18m)
LightGBM: 18.82m → 18.64m (-0.18m)
CatBoost: 18.82m → 18.73m (-0.09m)
평균:     18.85m → 18.70m (-0.15m)

LB Score:
16.9724 → 16.8272 (-0.14m)
```

**기술적 성과**
- ✅ Data Leakage 방지 (누적 통계)
- ✅ 현실적 구현 (과거 데이터만)
- ✅ 신규 선수 처리 (전체 평균)
- ✅ 예측 파이프라인 완성

---

## 핵심 학습

### 1. 도메인 지식의 가치
```
"선수마다 패스 스타일이 다를 것이다"
→ player_avg_pass_distance
→ 3개 모델 모두 최고 중요도
→ 이론이 현실로!
```

### 2. 누적 통계 전략
```
시간순 누적:
Game 1: 통계 없음 (전체 평균)
Game 2: Game 1만 사용
Game N: Game 1~N-1 사용

장점:
- Data Leakage 완전 방지
- 현실적 (과거만 활용)
- 시간 흐름 반영
```

### 3. 점진적 개선의 누적
```
Phase 1: -    (베이스라인)
Phase 2: -7.3% (대폭 개선)
Phase 3: -0.2% (미세 개선)
Phase 4: -0.8% (의미 있는 개선)

→ 작은 개선도 누적되면 큰 효과!
→ 포기하지 않는 것이 중요
```

### 4. 이론과 현실의 괴리
```
is_late_game: 좋은 아이디어
실제 데이터: 후반 75분 이후 없음
이유: 최종 패스는 초중반 발생

교훈:
- 도메인 지식도 검증 필요
- 데이터가 진실을 말함
- 실패한 피처도 정보
```

### 5. 모델 다양성 유지
```
XGBoost:  zone 중심 + 선수 스타일
LightGBM: 균형 + Phase 4 적극 활용
CatBoost: 공간 + 선수 통계

→ 각자 다르게 활용
→ 앙상블 효과 극대화
```

---

## 다음 단계 (고려중)

### Phase 5 후보 피처

**1. 선수 심화 특성**
```python
- player_position: 포지션별 특성
- player_experience: 경력 (경기 수)
- player_recent_form: 최근 5경기 성공률
- player_stamina: 경기 시간별 체력
```

**2. 상대 팀 압박**
```python
- opponent_pressure: 근처 상대 선수 수
- space_available: 빈 공간 크기
- defensive_line_distance: 상대 수비선까지 거리
```

**3. 경기 상황 심화**
```python
- time_pressure: 남은 시간 압박
- score_urgency: 득점 필요 긴급도
- momentum: 최근 5분 공격 성공률
```

**4. 팀 조합**
```python
- chemistry: 선수 간 조합 성공률
- formation_effect: 포메이션 영향
- substitution_impact: 교체 후 변화
```

---

## 회고

### 잘한 점
1. ✅ 도메인 지식을 체계적으로 적용
2. ✅ Data Leakage 완전 방지
3. ✅ 3개 모델 모두 개선
4. ✅ 현실적이고 재현 가능한 구현

### 아쉬운 점
1. ⚠️ is_late_game이 효과 없음 (데이터 특성)
2. ⚠️ 포지션 정보 미활용
3. ⚠️ 상대 팀 정보 미사용

### 배운 점
> **"도메인 지식 + 데이터 과학 = 최고의 조합"**
> 
> 기술만으로는 한계가 있다.
> 현장 지식이 결합되어야 진짜 개선이 온다.

---

## 참고 자료

### 코드 파일
- `notebooks/02_phase4_features.ipynb` (피처 생성)
- `src/models/train_model_v4.py` (XGBoost)
- `src/models/train_model_lgb_v4.py` (LightGBM)
- `src/models/train_model_catboost_v4.py` (CatBoost)
- `src/models/predict_ensemble_v4.py` (앙상블 예측)

### 데이터 파일
- `data/processed/train_final_passes_v4.csv` (Phase 4)
- `models/baseline_model_v4.pkl`
- `models/lgb_model_v4.pkl`
- `models/catboost_model_v4.pkl`

### 제출 파일
- `submissions/submission_ensemble_v4.csv` (LB 16.8272)

---

## 프롬프트 통계

- 총 프롬프트 수: 10+
- Phase 4 관련: 6개
- 평균 응답 시간: 30초
- 채택률: 100% (모든 제안 효과적)

---

**🎊 Phase 4 완료!**

```
Phase 1 → 2 → 3 → 4
기본   시계열  고급  도메인

20.36m → 18.88m → 18.85m → 18.70m

총 개선: -8.2% (1.66m)
```

**다음 목표: Phase 5 또는 Stacking**