# 🏎️ EV 주행거리 예측 대시보드 (F1 / 속도계 테마)

두 노트북(`데이터_병합_초본.ipynb`, `실전_프로젝트_머신러닝의_사본.ipynb`)의
전처리·모델링 파이프라인을 Streamlit 대시보드로 옮긴 결과물입니다.

- **타깃**: `Distance` (주행거리, km) — Trip 1건 = 1행
- **모델**: CatBoost (Optuna 튜닝 best params, 노트북 기준 R²≈0.84 / MAE≈3.7km)
- **입력**: 상위 20개 피처 (평균·최고속도, 주행시간, SOC 소모, 배터리 온도·출력, 가속도 등)
- **테마**: F1 피트월 콘셉트 — 속도계(게이지), 레이싱 폰트(Orbitron), 카본/레드 컬러

---

## 실행 방법

```bash
# 1) 가상환경(선택)
python -m venv venv && source venv/bin/activate   # 윈도우: venv\Scripts\activate

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 실행
streamlit run app.py
```

브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

---

## 실데이터로 돌리기

앱은 기본적으로 **물리식 기반 합성(데모) 데이터**로 동작합니다.
실제 데이터로 학습시키려면 둘 중 하나만 하면 됩니다.

1. `df_final_vif.csv` 를 `app.py` 와 같은 폴더에 두기 → 실행 시 자동 인식
2. 앱 좌측 사이드바의 **"실데이터 업로드"** 버튼으로 CSV 직접 업로드

> CSV에 노트북과 동일한 컬럼명(`Distance`, `Duration`, `SOC_Consumed`,
> `Velocity_mean` ...)이 있어야 합니다. 데이터 소스가 바뀌면 모델이 자동 재학습됩니다.

---

## 화면 구성

| 영역 | 내용 |
|---|---|
| 🎛️ 주행 셋업 | 평균/최고속도, 주행시간, SOC 소모, 가속도 등 슬라이더 (고급 변수 펼치기 포함) |
| 🏁 예측 주행거리 | 예측값(km) 대형 표시 + 평균속도 속도계 게이지(흰 눈금=최고속도) |
| KPI 카드 | 물리식 거리(v×t), 주행시간, SOC 소모, SOC 효율(km/%) |
| 📟 모델 성능 | R² / MAE / RMSE + 모델 vs 물리식 비교 막대 |
| 🧠 변수 중요도 | CatBoost feature importance Top 12 |

---

## 참고

- CatBoost 설치가 어려운 환경에서는 자동으로 scikit-learn `GradientBoosting`
  으로 폴백됩니다(사이드바에 표시).
- 데이터 출처: *BMW i3 — Battery and Heating Data in Real Driving Cycles*
  (FTM, TU München).
