# K리그 최종 패스 좌표 예측 🎯⚽

> K리그 경기 내 최종 패스 좌표 예측 AI 모델 개발

[![Public LB](https://img.shields.io/badge/Public%20LB-16.4981-blue)](https://github.com/polarsnow09/kleague-pass-prediction)
[![Rank](https://img.shields.io/badge/Rank-552%2F937%20(59%25)-green)](https://github.com/polarsnow09/kleague-pass-prediction)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)

## 🏆 주요 성과

- **Public LB**: 16.4981 RMSE (최고 기록 🥇)
- **순위**: 459/872 (상위 53%)
- **총 개선**: 20.36m → 16.4981m (**-18.9%**, 3.86m)
- **기간**: 2025.12.10 ~ 2026.01.12 (약 1개월)

## 📊 핵심 결과

| Phase | LB Score | 개선 | 주요 기법 |
|-------|----------|------|-----------|
| Phase 1 | 20.36m | - | 위치 기반 피처 |
| Phase 2 | 17.23m | -15.4% | 시계열 피처 |
| Phase 4 | 16.83m | -2.3% | 도메인 특화 피처 |
| Phase 5 | 16.53m 🥈 | -1.8% | **Stacking 앙상블** |
| **Phase 5.5** | **16.50m** 🥇 | **-0.2%** | **Blending 최적화** |

### Phase 5.5: Blending 최적화 (최종)

Grid Search 7개 가중치 탐색으로 **U자 곡선** 발견:

```
Phase 5 : Phase 6 = 0.60 : 0.40 (최적 비율)
```

**결과**: 16.5316m → 16.4981m (-0.2% 추가 개선)

## 🛠️ 기술 스택

**언어/라이브러리**
- Python 3.10 | pandas, numpy, scikit-learn, PyTorch

**모델**
- XGBoost, LightGBM, CatBoost
- Meta-Learning (Ridge, LightGBM, MLP)

**핵심 기법**
- 시계열 피처 엔지니어링
- 도메인 특화 피처 (선수/팀/경기 흐름)
- Stacking 앙상블 (OOF + Meta-Learner)
- Blending 최적화 (Grid Search)

## 🚀 빠른 시작

### 환경 설정
```bash
pip install -r requirements.txt
```

### 최종 모델 실행 (Phase 5.5)
```bash
# 1. OOF 예측 생성
python src/models/generate_oof_predictions.py

# 2. Meta-Learner 학습
python src/models/train_meta_learner.py

# 3. Stacking 예측
python src/models/predict_stacking.py

# 4. Grid Search로 최적 가중치 탐색
python src/models/predict_averaging_grid_search.py
python src/models/predict_averaging_grid_search_v2.py
```

## 📂 프로젝트 구조

```
kleague-pass-prediction/
├── data/                   # 데이터
│   ├── raw/               # 원본
│   └── processed/         # 전처리 (Phase 2-6)
├── models/                # 학습된 모델 (.pkl)
├── src/
│   ├── features/          # 피처 생성 모듈
│   └── models/            # 학습/예측 스크립트
├── reports/
│   ├── figures/           # 시각화
│   └── prompts/           # AI 협업 로그 (10개)
└── submissions/           # 제출 파일
```

## 🎓 핵심 학습

### 1. 구조적 개선 > 피처 개선
```
Phase 3-4 피처 실험: +0.18m
Phase 5 Stacking:    +0.30m ⭐
```

### 2. 도메인 지식의 가치
```
player_avg_pass_distance (선수별 스타일)
→ 3개 모델 모두 Phase 4 최고 중요도!
```

### 3. Blending의 조건
```
성공: 비슷한 성능 모델끼리 (차이 0.05m 이내)
실패: 성능 차이 큰 모델 (0.3m 이상)
```

### 4. 언제 멈출지 아는 것도 능력
```
Grid Search 7개 → U자 곡선 발견
→ 최적점 확인 후 문서화 우선
```

## 🤖 AI 협업 프로세스

- **80+ 프롬프트**로 개발 속도 12배 향상
- **10개 프롬프트 로그**로 체계적 문서화
- 즉각적 피드백으로 빠른 이터레이션

📖 [AI Collaboration Log](reports/prompts/)

## 📝 상세 문서

프로젝트의 전체 과정, 기술적 구현, 성과 분석은 아래 Notion 페이지에서 확인하세요:

🔗 **[프로젝트 포트폴리오 (Notion)](https://sky-sunstone-c93.notion.site/K-2ff5874538bb809eb5dcfa43dd4a3479)** 

- 📊 Phase별 상세 분석
- 🛠️ 기술적 구현 세부사항
- 📈 피처 중요도 및 인사이트
- 💡 회고 및 핵심 학습

## 📞 연락처

- **이메일**: polarsnow09@gmail.com
- **GitHub**: [polarsnow09](https://github.com/polarsnow09)
- **LinkedIn**

---

**"작은 개선의 누적이 큰 결과를 만든다"**

Phase 1 → 2 → 3 → 4 → 5 → 5.5  
20.36m → 16.50m (-18.9%)

포기하지 않고 계속 시도하는 것이 핵심! 💪
