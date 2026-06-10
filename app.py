# -*- coding: utf-8 -*-
"""
BMW i3 EV 주행거리 예측 대시보드
실행: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_CATBOOST = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

LEAKAGE_COLS = {'Duration', 'SOC_Consumed', 'Battery_State_of_Charge_End'}

# SHAP 기반 최종 선택 피쳐 (13개)
FEATURES = [
    'SoC_lag1_std', 'Elevation_MA3_std', 'Battery_Current_max',
    'Velocity_mean', 'Throttle_lag1_std', 'Battery_Temperature_diff_mean',
    'Route_Area_Munich_East', 'Weather_rainy', 'AirCon_Power_lag1_mean',
    'Throttle_lag1_mean', 'Route_Area_FTMRoute_2x',
    'Route_Area_Highway', 'Route_Area_Munich_North_Fast_Charging',
]

CAT_FEATURES = {
    'Route_Area_Munich_East', 'Weather_rainy', 'Route_Area_FTMRoute_2x',
    'Route_Area_Highway', 'Route_Area_Munich_North_Fast_Charging',
}

TARGET = 'Distance'

BEST_PARAMS = {
    'iterations': 1862,
    'depth': 3,
    'learning_rate': 0.021653374253691442,
    'l2_leaf_reg': 12.76728103986453,
    'random_strength': 4.854975752752124,
    'bagging_temperature': 4.561826584647859,
}

META = {
    'SoC_lag1_std':                  ('SoC 변화 편차 (lag1)',        '%',    0.0,  20.0,  0.1),
    'Elevation_MA3_std':             ('고도 이동평균 편차 (MA3)',     'm',    0.0, 100.0,  0.5),
    'Battery_Current_max':           ('배터리 최대 전류',             'A',    0.0, 200.0,  1.0),
    'Velocity_mean':                 ('평균 속도',                   'km/h', 0.0, 110.0,  1.0),
    'Throttle_lag1_std':             ('스로틀 변화 편차 (lag1)',      '%',    0.0,  40.0,  0.5),
    'Battery_Temperature_diff_mean': ('배터리 온도 변화 평균',        '°C', -2.0,   5.0,  0.05),
    'AirCon_Power_lag1_mean':        ('에어컨 출력 평균 (lag1)',      'kW',   0.0,   5.0,  0.1),
    'Throttle_lag1_mean':            ('스로틀 평균 (lag1)',           '%',    0.0,  80.0,  1.0),
}

PRIMARY = ['Velocity_mean', 'Battery_Current_max', 'SoC_lag1_std',
           'Throttle_lag1_mean', 'AirCon_Power_lag1_mean',
           'Battery_Temperature_diff_mean', 'Elevation_MA3_std', 'Throttle_lag1_std']

BMW_BLUE  = '#1C69D4'
BMW_DARK  = '#0A3D91'
BMW_LIGHT = '#5B9BD5'
DARK      = '#0B1929'
PANEL     = '#0F2440'
PANEL2    = '#162d50'
LINE      = '#1E3A5F'
TXT       = '#FFFFFF'
SUB       = '#A8C4E0'
GREEN     = '#00C896'
AMBER     = '#FFB000'
RED       = '#E84040'


@st.cache_data(show_spinner=False)
def make_synthetic(n=500, seed=42):
    rng = np.random.default_rng(seed)
    vmean        = rng.uniform(18, 95, n)
    batt_curr_max= rng.uniform(30, 200, n)
    thr_mean     = rng.uniform(5, 60, n)
    thr_std      = rng.uniform(2, 30, n)
    soc_diff_std = rng.uniform(0.1, 15, n)
    elev_ma3_std = rng.uniform(0, 80, n)
    bt_diff_mean = rng.uniform(-0.5, 2.0, n)
    aircon_mean  = rng.uniform(0, 4.0, n)

    highway  = (rng.random(n) < 0.20).astype(float)
    rainy    = (rng.random(n) < 0.15).astype(float)
    ftm_2x   = (rng.random(n) < 0.15).astype(float)
    muc_east = (rng.random(n) < 0.30).astype(float)
    muc_n_fc = (rng.random(n) < 0.10).astype(float)

    dist = (
        vmean * 0.32 + batt_curr_max * 0.06 + thr_mean * 0.12 +
        highway * 18.0 + rainy * (-5.0) + aircon_mean * (-1.5)
    ) * rng.normal(1, 0.10, n)
    dist = np.clip(dist, 1.5, None)
    dur = dist / np.clip(vmean, 10, None) * 60 * rng.normal(1, 0.05, n)
    soc = np.clip(dist / 120 + rng.normal(0, 0.02, n), 0.01, 0.6)

    return pd.DataFrame({
        'Distance': dist, 'Duration': dur, 'SOC_Consumed': soc,
        'Velocity_mean':                 vmean,
        'Battery_Current_max':           batt_curr_max,
        'Throttle_lag1_mean':            thr_mean,
        'Throttle_lag1_std':             thr_std,
        'SoC_lag1_std':                  soc_diff_std,
        'Elevation_MA3_std':             elev_ma3_std,
        'Battery_Temperature_diff_mean': bt_diff_mean,
        'AirCon_Power_lag1_mean':        aircon_mean,
        'Route_Area_Highway':            highway,
        'Weather_rainy':                 rainy,
        'Route_Area_FTMRoute_2x':        ftm_2x,
        'Route_Area_Munich_East':        muc_east,
        'Route_Area_Munich_North_Fast_Charging': muc_n_fc,
    })


@st.cache_data(show_spinner=False)
def load_data():
    candidates = ['df_final_vif.csv', 'data/df_final_vif.csv', './df_final_vif.csv']
    df_main = None
    for p in candidates:
        if os.path.exists(p):
            df_main = pd.read_csv(p)
            break

    if df_main is None:
        return make_synthetic(), 'synthetic'

    # AirCon_Power_mean → AirCon_Power_lag1_mean (proxy)
    if 'AirCon_Power_mean' in df_main.columns and 'AirCon_Power_lag1_mean' not in df_main.columns:
        df_main['AirCon_Power_lag1_mean'] = df_main['AirCon_Power_mean']

    # Elevation, Throttle, SoC lag/MA features from raw trip CSVs (순서 매칭)
    data_dir = 'data'
    trip_files = (sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.csv')])
                  if os.path.exists(data_dir) else [])

    if len(trip_files) == len(df_main):
        soc_lag1, elev_ma3, thr_std, thr_mean = [], [], [], []
        for fname in trip_files:
            try:
                df_raw = None
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df_raw = pd.read_csv(os.path.join(data_dir, fname), sep=';', encoding=enc)
                        if df_raw.shape[1] >= 5:
                            break
                    except Exception:
                        continue
                if df_raw is None or df_raw.empty:
                    raise ValueError

                soc_col = next((c for c in df_raw.columns
                                if 'soc [%]' in c.lower()
                                and not any(x in c.lower() for x in ('max', 'min', 'displayed'))), None)
                soc_lag1.append(float(pd.to_numeric(df_raw[soc_col], errors='coerce').diff().std())
                                if soc_col else np.nan)

                elev_col = next((c for c in df_raw.columns if 'elevation' in c.lower()), None)
                if elev_col:
                    e = pd.to_numeric(df_raw[elev_col], errors='coerce')
                    elev_ma3.append(float(e.rolling(3, min_periods=1).mean().std()))
                else:
                    elev_ma3.append(np.nan)

                thr_col = next((c for c in df_raw.columns if 'throttle' in c.lower()), None)
                if thr_col:
                    thr = pd.to_numeric(df_raw[thr_col], errors='coerce').shift(1)
                    thr_std.append(float(thr.std()))
                    thr_mean.append(float(thr.mean()))
                else:
                    thr_std.append(np.nan); thr_mean.append(np.nan)
            except Exception:
                soc_lag1.append(np.nan); elev_ma3.append(np.nan)
                thr_std.append(np.nan);  thr_mean.append(np.nan)

        df_main['SoC_lag1_std']       = soc_lag1
        df_main['Elevation_MA3_std']  = elev_ma3
        df_main['Throttle_lag1_std']  = thr_std
        df_main['Throttle_lag1_mean'] = thr_mean

    return df_main, 'real'


@st.cache_resource(show_spinner=False)
def train_model(df_key, df):
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET].copy()
    # NaN이 있는 행 제거 (일부 lag/MA 피쳐가 없는 경우 대비)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    if HAS_CATBOOST:
        cat_idx = [i for i, c in enumerate(cols) if c in CAT_FEATURES]
        for _c in CAT_FEATURES:
            if _c in Xtr.columns:
                Xtr[_c] = Xtr[_c].astype(int)
                Xte[_c] = Xte[_c].astype(int)
        model = CatBoostRegressor(**BEST_PARAMS, loss_function='RMSE',
                                  random_seed=42, verbose=False)
        model.fit(Xtr, ytr, cat_features=cat_idx if cat_idx else None)
    else:
        model = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                          learning_rate=0.05, random_state=42)
        model.fit(Xtr, ytr)
    pred_te = model.predict(Xte)
    metrics = {
        'R2': r2_score(yte, pred_te),
        'MAE': mean_absolute_error(yte, pred_te),
        'RMSE': mean_squared_error(yte, pred_te) ** 0.5,
        'n': len(df),
        'yte': yte.values,
        'pred_te': pred_te,
    }
    imp = (model.get_feature_importance() if HAS_CATBOOST
           else model.feature_importances_)
    importance = (pd.DataFrame({'Feature': cols, 'Importance': imp})
                  .sort_values('Importance', ascending=False)
                  .reset_index(drop=True))
    return model, metrics, importance, cols, df[cols].median()


@st.cache_data(show_spinner=False)
def load_trip_list():
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return []
    files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.csv')])
    return files


@st.cache_data(show_spinner=False)
def load_trip_raw(filename):
    path = os.path.join('data', filename)
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(path, sep=';', encoding=enc)
            if df.shape[1] >= 10:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def _get_col(df, keywords, exclude=None):
    exclude = [e.lower() for e in (exclude or [])]
    for c in df.columns:
        cl = c.lower()
        if any(ex in cl for ex in exclude):
            continue
        if any(kw.lower() in cl for kw in keywords):
            return pd.to_numeric(df[c], errors='coerce')
    return None


def compute_trip_features(df, route_area=None, weather_rainy=0):
    def safe(s, default=0.0):
        return s if s is not None else pd.Series([default] * len(df))

    t    = safe(_get_col(df, ['time [s]', 'time']), 0.0)
    v    = safe(_get_col(df, ['velocity']), 0.0)
    bt   = safe(_get_col(df, ['battery temperature'],
                         exclude=['max', 'min', 'coolant', 'exchanger', 'heater', 'cabin', 'inlet']), 20.0)
    soc  = safe(_get_col(df, ['soc [%]'], exclude=['max', 'min', 'displayed']), 50.0)
    bc   = safe(_get_col(df, ['battery current']), 0.0)
    ac   = safe(_get_col(df, ['aircon power', 'air con']), 0.0)
    thr  = safe(_get_col(df, ['throttle']), 0.0)
    elev = safe(_get_col(df, ['elevation']), 0.0)

    dur       = float((t.max() - t.min()) / 60)
    vel_mean  = float(v.mean())
    soc_clean = soc.dropna()
    soc_start = float(soc_clean.iloc[0])  if len(soc_clean) > 0 else 50.0
    soc_end_val = float(soc_clean.iloc[-1]) if len(soc_clean) > 0 else 40.0
    soc_consumed = max(0.0, (soc_start - soc_end_val) / 100)

    batt_curr_max        = float(bc.max())
    soc_lag1_std         = float(soc.diff().std())
    elev_ma3_std         = float(elev.rolling(3, min_periods=1).mean().std())
    thr_lag1_std         = float(thr.shift(1).std())
    thr_lag1_mean        = float(thr.shift(1).mean())
    bt_diff_mean         = float(bt.diff().mean())
    aircon_lag1_mean     = float(ac.shift(1).mean())

    dt = t.diff().fillna(0.0)
    actual_dist = float((v * dt / 3600).sum())

    ra = route_area or {}
    feats = {
        'Duration': dur, 'SOC_Consumed': soc_consumed,
        'Velocity_mean':                 vel_mean,
        'Battery_Current_max':           batt_curr_max,
        'SoC_lag1_std':                  soc_lag1_std,
        'Elevation_MA3_std':             elev_ma3_std,
        'Throttle_lag1_std':             thr_lag1_std,
        'Battery_Temperature_diff_mean': bt_diff_mean,
        'AirCon_Power_lag1_mean':        aircon_lag1_mean,
        'Throttle_lag1_mean':            thr_lag1_mean,
        'Route_Area_Munich_East':             float(ra.get('Munich_East', 0)),
        'Route_Area_FTMRoute_2x':             float(ra.get('FTMRoute_2x', 0)),
        'Route_Area_Highway':                 float(ra.get('Highway', 0)),
        'Route_Area_Munich_North_Fast_Charging': float(ra.get('Munich_North_Fast_Charging', 0)),
        'Weather_rainy':                 float(weather_rainy),
    }
    return feats, actual_dist, soc_start, soc_end_val


@st.cache_data(show_spinner=False)
def load_averaged_timeseries(trip_files_tuple):
    N_POINTS = 300
    pct_grid = np.linspace(0, 100, N_POINTS)
    ch = {'v': [], 'ac': [], 'sc': [], 'bt': []}
    for fname in trip_files_tuple:
        df_raw = load_trip_raw(fname)
        if df_raw.empty:
            continue
        t_col = next((c for c in df_raw.columns if 'time' in c.lower()), None)
        if t_col is None:
            continue
        t_raw = pd.to_numeric(df_raw[t_col], errors='coerce')
        t_min, t_max = t_raw.min(), t_raw.max()
        if pd.isna(t_min) or pd.isna(t_max) or t_max == t_min:
            continue
        t_pct = (t_raw - t_min) / (t_max - t_min) * 100
        v_col  = next((c for c in df_raw.columns if 'velocity' in c.lower()), None)
        ac_col = next((c for c in df_raw.columns if 'longitudinal acceleration' in c.lower()), None)
        sc_col = next((c for c in df_raw.columns
                       if 'soc [%]' in c.lower() and 'max' not in c.lower()
                       and 'min' not in c.lower() and 'displayed' not in c.lower()), None)
        bt_col = next((c for c in df_raw.columns
                       if 'battery temperature' in c.lower() and 'max' not in c.lower()), None)
        for col_name, key in [(v_col, 'v'), (ac_col, 'ac'), (sc_col, 'sc'), (bt_col, 'bt')]:
            if col_name is None or col_name not in df_raw.columns:
                continue
            y = pd.to_numeric(df_raw[col_name], errors='coerce')
            mask = ~(t_pct.isna() | y.isna())
            if mask.sum() < 5:
                continue
            tv = t_pct[mask].values
            yv = y[mask].values
            sidx = np.argsort(tv)
            ch[key].append(np.interp(pct_grid, tv[sidx], yv[sidx]))
    avg = {k: np.mean(v, axis=0) if v else None for k, v in ch.items()}
    return pct_grid, avg


@st.cache_data(show_spinner=False)
def load_model_comparison():
    base_path = os.path.join('model comparison', 'baseline_model_comparison.csv')
    opt_path  = os.path.join('model comparison', 'optuna_gridsearch_comparison.csv')
    df_base = pd.read_csv(base_path, encoding='utf-8') if os.path.exists(base_path) else pd.DataFrame()
    df_opt  = pd.read_csv(opt_path,  encoding='utf-8') if os.path.exists(opt_path)  else pd.DataFrame()
    return df_base, df_opt


# ── 페이지 설정 + CSS ────────────────────────────────────────────
st.set_page_config(page_title='BMW i3 · EV Dashboard', page_icon='🚗',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown(f"""
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

 .stApp {{
   background: linear-gradient(160deg, #0B1929 0%, #0D1F36 50%, #091525 100%) fixed;
   color:{TXT}; font-family:'Inter',sans-serif;
 }}
 #MainMenu, footer, header {{ visibility:hidden; }}
 [data-testid="collapsedControl"],
 [data-testid="collapsedControl"]:hover,
 [data-testid="collapsedControl"]:focus,
 button[kind="header"] {{ display:none !important; }}
 .block-container {{ padding-top:1.5rem; max-width:1440px; }}

 /* 사이드바 배경 */
 section[data-testid="stSidebar"] {{
   background: linear-gradient(180deg, #0A1E35 0%, #0D2441 100%);
   border-right: 1px solid {LINE};
 }}
 /* 사이드바 텍스트 — 모든 라벨/단락/span 강제 흰색 */
 section[data-testid="stSidebar"] label,
 section[data-testid="stSidebar"] label p,
 section[data-testid="stSidebar"] label span,
 section[data-testid="stSidebar"] .stRadio label,
 section[data-testid="stSidebar"] .stRadio label p,
 section[data-testid="stSidebar"] div[role="radiogroup"] label {{
   color: {TXT} !important;
   font-size: .95rem !important;
   font-weight: 500 !important;
 }}
 section[data-testid="stSidebar"] .stCaption,
 section[data-testid="stSidebar"] small {{
   color: {SUB} !important;
 }}

 .bmw-title {{
   font-weight:700; font-size:2rem; letter-spacing:.5px; color:{TXT};
   border-left:5px solid {BMW_BLUE}; padding-left:16px; line-height:1.2;
 }}
 .bmw-sub {{ color:{SUB}; font-size:.9rem; padding-left:21px; margin-top:2px; }}

 .card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; padding:18px 20px; }}

 .kpi {{ text-align:center; background:{PANEL}; border:1px solid {LINE};
   border-radius:10px; padding:14px 10px; }}
 .kpi .v {{ font-size:1.7rem; font-weight:700; color:{TXT}; }}
 .kpi .l {{ color:{SUB}; font-size:.75rem; letter-spacing:1.5px;
   text-transform:uppercase; margin-top:2px; }}

 .pred-box {{
   background: linear-gradient(135deg, #0d2d52 0%, #0A3D91 100%);
   border:2px solid {BMW_BLUE}; border-radius:16px; padding:28px; text-align:center;
 }}
 .pred-num {{ font-size:4.2rem; font-weight:700; color:#fff; line-height:1; }}
 .pred-unit {{ font-size:1.3rem; color:{SUB}; vertical-align:super; }}
 .pred-label {{ color:{BMW_LIGHT}; font-size:.85rem; letter-spacing:2px;
   text-transform:uppercase; margin-top:8px; }}

 .divider {{ height:1px; background: linear-gradient(90deg,
   transparent, {LINE} 20%, {LINE} 80%, transparent); margin:20px 0; }}

 .stSlider label {{ color:{TXT} !important; font-weight:500; }}
 .stButton>button {{ background:{BMW_BLUE}; color:#fff; border:0; border-radius:8px;
   font-weight:600; padding:.55rem 1.4rem; }}
 .stButton>button:hover {{ background:{BMW_DARK}; }}

 .badge {{ display:inline-block; padding:3px 10px; border-radius:20px;
   font-size:.75rem; font-weight:600; letter-spacing:.5px; }}

 .sec-head {{ font-size:1.1rem; font-weight:600; color:{TXT}; margin-bottom:12px;
   padding-bottom:8px; border-bottom:1px solid {LINE}; }}

 .insight {{
   background: linear-gradient(135deg, #0d2448 0%, {PANEL} 100%);
   border-left:3px solid {BMW_BLUE}; border-radius:0 8px 8px 0;
   padding:10px 14px; margin:8px 0; font-size:.88rem; color:{SUB};
 }}
 .insight strong {{ color:{TXT}; }}
</style>
""", unsafe_allow_html=True)


# ── 데이터 & 모델 ────────────────────────────────────────────────
df, source = load_data()
model, metrics, importance, cols, medians = train_model(source + str(len(df)), df)
trip_list = load_trip_list()

BMW_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="32" height="32" style="vertical-align:middle;margin-right:8px">
  <circle cx="50" cy="50" r="49" fill="#1a1a1a"/>
  <circle cx="50" cy="50" r="36" fill="#fff"/>
  <path d="M50 14 A36 36 0 0 1 86 50 L50 50 Z" fill="#1C69D4"/>
  <path d="M50 86 A36 36 0 0 1 14 50 L50 50 Z" fill="#1C69D4"/>
  <circle cx="50" cy="50" r="36" fill="none" stroke="#1a1a1a" stroke-width="2"/>
  <line x1="50" y1="14" x2="50" y2="86" stroke="#1a1a1a" stroke-width="2"/>
  <line x1="14" y1="50" x2="86" y2="50" stroke="#1a1a1a" stroke-width="2"/>
  <circle cx="50" cy="50" r="49" fill="none" stroke="#1a1a1a" stroke-width="2"/>
</svg>
"""

# ── 사이드바 ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px 0">
      <div style="font-size:1.7rem;font-weight:700;color:{TXT};display:flex;align-items:center">
        {BMW_LOGO_SVG} BMW i3
      </div>
      <div style="color:{SUB};font-size:.95rem;margin-top:4px">EV 주행거리 예측 대시보드</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "페이지 선택",
        ["데이터 현황", "주행거리 예측", "트립별 예측 검증", "모델 분석", "변수 분석"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    selected_trip = None
    selected_data_trip = "전체"
    uploaded_pred = None

    if page == "트립별 예측 검증":
        if trip_list:
            st.markdown(
                f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
                f'letter-spacing:.5px;margin-bottom:6px">트립 선택</div>',
                unsafe_allow_html=True)
            selected_trip = st.selectbox("trip", trip_list, label_visibility='collapsed')
        else:
            st.warning('data/ 폴더에 트립 파일이 없습니다.')

    elif page == "데이터 현황":
        if trip_list:
            st.markdown(
                f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
                f'letter-spacing:.5px;margin-bottom:6px">트립 선택</div>',
                unsafe_allow_html=True)
            data_trip_opts = ["전체"] + trip_list
            selected_data_trip = st.selectbox(
                "data_trip", data_trip_opts,
                format_func=lambda x: x.replace('.csv', '') if x != '전체' else '전체',
                key='sidebar_data_trip',
                label_visibility='collapsed'
            )

    elif page == "주행거리 예측":
        st.markdown(
            f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
            f'letter-spacing:.5px;margin-bottom:6px">트립 CSV 업로드</div>',
            unsafe_allow_html=True)
        uploaded_pred = st.file_uploader(
            '트립 CSV 업로드',
            type=['csv'], key='pred_csv_upload', label_visibility='collapsed',
        )
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        if st.button('슬라이더 초기화', key='pred_reset'):
            for _f in cols:
                st.session_state.pop(f'pred_{_f}', None)
            st.rerun()

    badge_html = (f'<span class="badge" style="background:{GREEN};color:#000">● 실데이터</span>'
                  if source == 'real' else
                  f'<span class="badge" style="background:{AMBER};color:#000">● 데모(합성)</span>')
    st.markdown(badge_html, unsafe_allow_html=True)
    eng = 'CatBoost' if HAS_CATBOOST else 'GradientBoosting'
    st.caption(f'엔진: {eng}  ·  샘플: {len(df)}건')
    if trip_list:
        st.caption(f'트립 파일: {len(trip_list)}개')


# ════════════════════════════════════════════════════════════════
# PAGE 1 : 데이터 현황
# ════════════════════════════════════════════════════════════════
if page == "데이터 현황":
    # ── 선택 트립 데이터 로드 ─────────────────────────────────
    _sel_feats = None
    _sel_dist  = None
    _sel_soc_s = None
    _sel_soc_e = None
    if selected_data_trip != "전체" and trip_list:
        _td_raw = load_trip_raw(selected_data_trip)
        if not _td_raw.empty:
            _sel_feats, _sel_dist, _sel_soc_s, _sel_soc_e = compute_trip_features(_td_raw)

    trip_label_d = selected_data_trip.replace('.csv', '') if selected_data_trip != "전체" else None

    st.markdown('<div class="bmw-title">BMW i3 · 주행 데이터 현황</div>', unsafe_allow_html=True)
    if trip_label_d:
        st.markdown(f'<div class="bmw-sub">개별 트립 분석 · <span style="color:{BMW_BLUE};font-weight:600">{trip_label_d}</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="bmw-sub">트립 집계 데이터 탐색 · 변수 분포 · 상관관계</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 데이터 요약 KPI ───────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    if _sel_feats is not None:
        for c_, val, lab in [
            (k1, f'{_sel_feats["Duration"]:.1f} min',       '주행 시간'),
            (k2, f'{_sel_dist:.1f} km',                      '실제 주행거리'),
            (k3, f'{_sel_feats["Velocity_mean"]:.1f} km/h', '평균 속도'),
            (k4, f'{_sel_feats["SOC_Consumed"]*100:.1f} %', 'SOC 소모'),
            (k5, f'{_sel_soc_s:.0f}% → {_sel_soc_e:.0f}%', 'SoC 범위'),
        ]:
            c_.markdown(f'<div class="kpi"><div class="v">{val}</div>'
                       f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)
    else:
        dist_mean = df['Distance'].mean()         if 'Distance'      in df.columns else 0
        dist_std  = df['Distance'].std()          if 'Distance'      in df.columns else 0
        vel_mean  = df['Velocity_mean'].mean()    if 'Velocity_mean' in df.columns else 0
        soc_mean  = df['SOC_Consumed'].mean()*100 if 'SOC_Consumed'  in df.columns else 0
        for c_, val, lab in [
            (k1, f'{len(df)}건',         '총 트립 수'),
            (k2, f'{dist_mean:.1f} km',  '평균 주행거리'),
            (k3, f'±{dist_std:.1f} km',  '주행거리 편차'),
            (k4, f'{vel_mean:.1f} km/h', '평균 속도'),
            (k5, f'{soc_mean:.1f} %',    '평균 SOC 소모'),
        ]:
            c_.markdown(f'<div class="kpi"><div class="v">{val}</div>'
                       f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    # ── 인터랙티브 분포 탭 ────────────────────────────────────
    DIST_COLORS = [BMW_BLUE, BMW_LIGHT, GREEN, AMBER, '#9C6EDD', '#FF6B8A']

    tab1, tab2, tab3 = st.tabs(["주요 변수 분포", "변화량 분포", "트립 시계열"])

    # ── Tab 1 : 주요 변수 분포 ─────────────────────────────────
    with tab1:
        dist_vars_def = [
            ('Distance',           '주행거리',        'km'),
            ('Velocity_mean',      '평균 속도',       'km/h'),
            ('Battery_Current_max','배터리 최대 전류', 'A'),
        ]
        dist_vars = [(c, l, u) for c, l, u in dist_vars_def if c in df.columns]
        if dist_vars:
            t1_cols = st.columns(3)
            for idx, (col_name, label, unit) in enumerate(dist_vars):
                vals = df[col_name].dropna()
                if col_name == 'SOC_Consumed':
                    vals = vals * 100
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=vals, nbinsx=25,
                    marker_color=DIST_COLORS[idx % len(DIST_COLORS)],
                    opacity=0.82, showlegend=False,
                ))
                fig_h.add_vline(
                    x=float(vals.mean()), line_color=AMBER,
                    line_width=1.5, line_dash='dash',
                    annotation_text=f'μ={vals.mean():.1f}',
                    annotation_font_color=AMBER, annotation_font_size=9,
                )
                if _sel_feats is not None:
                    if col_name == 'Distance':
                        trip_val = _sel_dist
                    elif col_name == 'SOC_Consumed':
                        trip_val = _sel_feats.get('SOC_Consumed', 0) * 100
                    else:
                        trip_val = _sel_feats.get(col_name)
                    if trip_val is not None:
                        fig_h.add_vline(
                            x=float(trip_val), line_color=RED, line_width=2.5,
                            annotation_text=f'선택 {trip_val:.1f}',
                            annotation_font_color=RED, annotation_font_size=9,
                            annotation_position='top right',
                        )
                fig_h.update_layout(
                    title={'text': label, 'font': {'color': TXT, 'size': 12}, 'x': 0.03},
                    height=260, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=50, r=10, t=35, b=50),
                    xaxis={
                        'gridcolor': LINE, 'color': SUB, 'tickfont': {'size': 9},
                        'title': {'text': f'{label} ({unit})',
                                  'font': {'size': 10, 'color': SUB}},
                    },
                    yaxis={
                        'gridcolor': LINE, 'color': SUB, 'tickfont': {'size': 9},
                        'title': {'text': '빈도 (건수)',
                                  'font': {'size': 10, 'color': SUB}},
                    },
                )
                t1_cols[idx].plotly_chart(fig_h, use_container_width=True,
                                          config={'displayModeBar': False})
        if _sel_feats is not None:
            st.markdown(
                f'<div class="insight">주황 점선 = 전체 평균 · <span style="color:{RED};font-weight:600">빨간 실선</span> = <strong>{trip_label_d}</strong> 트립 위치</div>',
                unsafe_allow_html=True)

    # ── Tab 2 : 변화량 분포 ────────────────────────────────────
    with tab2:
        var_vars_def = [
            ('SoC_lag1_std',                  'SoC 변화 편차 (lag1)', '%'),
            ('Elevation_MA3_std',             '고도 MA3 편차',        'm'),
            ('Battery_Temperature_diff_mean', '배터리 온도 변화 평균', '°C'),
        ]
        var_vars = [(c, l, u) for c, l, u in var_vars_def if c in df.columns]
        if var_vars:
            t2_cols = st.columns(3)
            for idx, (col_name, label, unit) in enumerate(var_vars):
                vals = df[col_name].dropna()
                fig_v = go.Figure()
                fig_v.add_trace(go.Histogram(
                    x=vals, nbinsx=25,
                    marker_color=DIST_COLORS[idx % len(DIST_COLORS)],
                    opacity=0.82, showlegend=False,
                ))
                fig_v.add_vline(
                    x=float(vals.mean()), line_color=AMBER,
                    line_width=1.5, line_dash='dash',
                    annotation_text=f'μ={vals.mean():.2f}',
                    annotation_font_color=AMBER, annotation_font_size=9,
                )
                if _sel_feats is not None:
                    trip_val2 = _sel_feats.get(col_name)
                    if trip_val2 is not None:
                        fig_v.add_vline(
                            x=float(trip_val2), line_color=RED, line_width=2.5,
                            annotation_text=f'선택 {trip_val2:.2f}',
                            annotation_font_color=RED, annotation_font_size=9,
                            annotation_position='top right',
                        )
                fig_v.update_layout(
                    title={'text': label, 'font': {'color': TXT, 'size': 12}, 'x': 0.03},
                    height=260, paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=50, r=10, t=35, b=50),
                    xaxis={
                        'gridcolor': LINE, 'color': SUB, 'tickfont': {'size': 9},
                        'title': {'text': f'{label} ({unit})',
                                  'font': {'size': 10, 'color': SUB}},
                    },
                    yaxis={
                        'gridcolor': LINE, 'color': SUB, 'tickfont': {'size': 9},
                        'title': {'text': '빈도 (건수)',
                                  'font': {'size': 10, 'color': SUB}},
                    },
                )
                t2_cols[idx].plotly_chart(fig_v, use_container_width=True,
                                          config={'displayModeBar': False})
            if _sel_feats is not None:
                st.markdown(
                    f'<div class="insight">주황 점선 = 전체 평균 · <span style="color:{RED};font-weight:600">빨간 실선</span> = <strong>{trip_label_d}</strong> 트립 위치 · 값이 클수록 주행 조건이 불안정합니다</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="insight">값이 클수록 해당 트립의 주행 조건이 불안정했음을 나타냅니다. 주황 점선은 전체 트립 평균입니다.</div>',
                    unsafe_allow_html=True)

    # ── Tab 3 : 트립 시계열 ────────────────────────────────────
    with tab3:
        if not trip_list:
            st.info('data/ 폴더에 트립 CSV 파일이 없습니다.')
        else:
            def _draw_ts_subplots(x_vals, panels_data, x_label, caption_text):
                n3 = len(panels_data)
                t3_fig = make_subplots(rows=n3, cols=1, shared_xaxes=True,
                                       row_heights=[1] * n3, vertical_spacing=0.04)
                for ri, (y3, lbl, unit, clr, fc, do_fill) in enumerate(panels_data, 1):
                    t3_fig.add_trace(go.Scatter(
                        x=x_vals, y=y3, mode='lines', name=lbl,
                        line={'color': clr, 'width': 1.5},
                        fill='tozeroy' if do_fill else 'none',
                        fillcolor=fc if do_fill else None, showlegend=True,
                    ), row=ri, col=1)
                    if 'ac' in lbl.lower() or '가속' in lbl:
                        t3_fig.add_hline(y=0, line_color=SUB, line_dash='dot',
                                         line_width=0.8, row=ri, col=1)
                    t3_fig.update_yaxes(
                        title_text=f'{lbl}<br>({unit})',
                        title_font={'size': 9, 'color': SUB},
                        gridcolor=LINE, color=SUB, tickfont={'size': 8},
                        row=ri, col=1)
                for r in range(1, n3 + 1):
                    t3_fig.update_xaxes(
                        gridcolor=LINE, color=SUB, tickfont={'size': 9},
                        showticklabels=(r == n3),
                        title_text=x_label if r == n3 else '',
                        title_font={'size': 10, 'color': SUB},
                        row=r, col=1)
                t3_fig.update_layout(
                    height=80 + n3 * 130,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=70, r=20, t=16, b=40),
                    legend={'font': {'color': SUB, 'size': 10},
                            'bgcolor': 'rgba(0,0,0,0)',
                            'orientation': 'h', 'x': 0, 'y': 1.02})
                st.plotly_chart(t3_fig, use_container_width=True, config={'displayModeBar': False})
                st.caption(caption_text)

            if selected_data_trip == "전체":
                pct_grid, avg = load_averaged_timeseries(tuple(trip_list))
                panels_avg_def = [
                    ('v',  '속도',         'km/h', BMW_BLUE, 'rgba(28,105,212,0.15)', True),
                    ('ac', '종방향 가속도', 'm/s²', AMBER,   'rgba(255,176,0,0.10)',  False),
                    ('sc', 'SoC',           '%',    GREEN,   'rgba(0,200,150,0.12)',  True),
                    ('bt', '배터리 온도',   '°C',   RED,     'rgba(232,64,64,0.10)',  False),
                ]
                panels_avg = [(avg[k], lbl, unit, clr, fc, fi)
                              for k, lbl, unit, clr, fc, fi in panels_avg_def
                              if avg.get(k) is not None]
                if panels_avg:
                    _draw_ts_subplots(pct_grid, panels_avg,
                                      '주행 진행률 (%)',
                                      f'전체 {len(trip_list)}개 트립의 평균 시계열 · 가로축은 주행 완료 비율입니다')
                else:
                    st.warning('평균 계산에 사용할 수 있는 트립 데이터가 없습니다.')
            else:
                ts_df = load_trip_raw(selected_data_trip)
                if ts_df.empty:
                    st.warning('데이터를 불러올 수 없습니다.')
                else:
                    ts_t  = next((c for c in ts_df.columns if 'time' in c.lower()), None)
                    ts_v  = next((c for c in ts_df.columns if 'velocity' in c.lower()), None)
                    ts_ac = next((c for c in ts_df.columns
                                  if 'longitudinal acceleration' in c.lower()), None)
                    ts_sc = next((c for c in ts_df.columns
                                  if 'soc [%]' in c.lower()
                                  and 'max' not in c.lower()
                                  and 'min' not in c.lower()
                                  and 'displayed' not in c.lower()), None)
                    ts_bt = next((c for c in ts_df.columns
                                  if 'battery temperature' in c.lower()
                                  and 'max' not in c.lower()), None)
                    if ts_t:
                        t3_t_ser = pd.to_numeric(ts_df[ts_t], errors='coerce') / 60
                        panels_def3 = [
                            (ts_v,  '속도',         'km/h', BMW_BLUE, 'rgba(28,105,212,0.15)', True),
                            (ts_ac, '종방향 가속도', 'm/s²', AMBER,   'rgba(255,176,0,0.10)',  False),
                            (ts_sc, 'SoC',           '%',    GREEN,   'rgba(0,200,150,0.12)',  True),
                            (ts_bt, '배터리 온도',   '°C',   RED,     'rgba(232,64,64,0.10)',  False),
                        ]
                        t3_panels = [(pd.to_numeric(ts_df[c], errors='coerce'),
                                      l, u, cl, fc, fi)
                                     for c, l, u, cl, fc, fi in panels_def3
                                     if c and c in ts_df.columns]
                        if t3_panels:
                            _draw_ts_subplots(t3_t_ser, t3_panels,
                                              '시간 (분)',
                                              selected_data_trip.replace('.csv', '') + ' 트립 원본 시계열')

    # ── 상관관계 히트맵 ───────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">주요 변수 상관관계</div>', unsafe_allow_html=True)
    heat_cols = ['Distance', 'Velocity_mean', 'Battery_Current_max',
                 'SoC_lag1_std', 'Elevation_MA3_std', 'Throttle_lag1_mean',
                 'AirCon_Power_lag1_mean', 'Battery_Temperature_diff_mean']
    heat_cols = [c for c in heat_cols if c in df.columns]
    corr = df[heat_cols].corr().round(2)
    labels_h = [META[c][0] if c in META else c.replace('_', ' ') for c in corr.columns]

    h_left, h_right = st.columns([2, 1])

    with h_left:
        heat = go.Figure(go.Heatmap(
            z=corr.values, x=labels_h, y=labels_h,
            colorscale=[[0, '#0a1929'], [0.5, '#1C69D4'], [1, '#ffffff']],
            zmin=-1, zmax=1, text=corr.values, texttemplate='%{text}',
            textfont={'size': 10}, hoverongaps=False,
        ))
        heat.update_layout(
            height=380, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'color': SUB, 'tickangle': -30},
            yaxis={'color': SUB},
        )
        st.plotly_chart(heat, use_container_width=True, config={'displayModeBar': False})

    with h_right:
        # 주행거리와의 상관계수 순위
        if 'Distance' in corr.columns:
            dist_corr = (corr['Distance']
                         .drop('Distance', errors='ignore')
                         .abs()
                         .sort_values(ascending=False))
            raw_corr  = corr['Distance'].drop('Distance', errors='ignore')

            st.markdown(
                '<div style="color:#A8C4E0;font-size:0.8rem;font-weight:600;'
                'margin-bottom:8px;">주행거리와의 상관관계 순위</div>',
                unsafe_allow_html=True,
            )
            for rank, (col_key, abs_val) in enumerate(dist_corr.items(), 1):
                raw_val  = raw_corr[col_key]
                bar_color = BMW_BLUE if raw_val >= 0 else RED
                bar_pct   = int(abs_val * 100)
                col_label = META[col_key][0] if col_key in META else col_key.replace('_', ' ')
                sign_char  = '+' if raw_val >= 0 else '−'
                st.markdown(f"""
                <div style="margin-bottom:6px;">
                  <div style="display:flex;justify-content:space-between;
                              font-size:0.78rem;color:#FFFFFF;margin-bottom:2px;">
                    <span>#{rank} {col_label}</span>
                    <span style="color:{bar_color};font-weight:700;">
                      {sign_char}{abs_val:.2f}
                    </span>
                  </div>
                  <div style="background:#1E3A5F;border-radius:3px;height:6px;">
                    <div style="width:{bar_pct}%;background:{bar_color};
                                border-radius:3px;height:6px;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 : 트립 예측 검증
# ════════════════════════════════════════════════════════════════
elif page == "트립별 예측 검증":
    st.markdown('<div class="bmw-title">트립별 예측 검증</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">실제 트립의 예측값 vs 실제값 비교 · 오차 분석 · 사이드바에서 트립을 선택하세요</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not trip_list:
        st.warning('data/ 폴더에 트립 CSV 파일이 없습니다.')
        st.stop()
    if selected_trip is None:
        selected_trip = trip_list[0]

    trip_df = load_trip_raw(selected_trip)
    if trip_df.empty:
        st.error(f'트립 데이터를 불러올 수 없습니다: {selected_trip}')
        st.stop()

    feats, actual_dist, soc_start_pct, soc_end_pct = compute_trip_features(trip_df)
    X_trip = pd.DataFrame([feats])[cols]
    for _c in CAT_FEATURES:
        if _c in X_trip.columns:
            X_trip[_c] = X_trip[_c].astype(int)
    pred_dist = float(max(model.predict(X_trip)[0], 0.0))
    trip_name = selected_trip.replace('.csv', '')
    series_label = ('Series A' if 'TripA' in selected_trip else
                    'Series B' if 'TripB' in selected_trip else '—')

    k1, k2, k3, k4 = st.columns(4)
    for c_, val, lab in [
        (k1, f'{feats["Duration"]:.1f} min',       '주행 시간'),
        (k2, f'{feats["Velocity_mean"]:.1f} km/h', '평균 속도'),
        (k3, f'{feats["SOC_Consumed"]*100:.1f} %', 'SOC 소모'),
        (k4, f'{soc_start_pct:.0f}% → {soc_end_pct:.0f}%', '초기 → 종료 SoC'),
    ]:
        c_.markdown(f'<div class="kpi"><div class="v">{val}</div><div class="l">{lab}</div></div>',
                   unsafe_allow_html=True)

    # 시계열 컬럼 탐색 (레이아웃 전체에서 공유)
    t_col_name   = next((c for c in trip_df.columns if 'time' in c.lower()), None)
    v_col_name   = next((c for c in trip_df.columns if 'velocity' in c.lower()), None)
    soc_col_name = next(
        (c for c in trip_df.columns
         if 'soc [%]' in c.lower() and 'max' not in c.lower()
         and 'min' not in c.lower() and 'displayed' not in c.lower()), None)

    err_pct = abs(pred_dist - actual_dist) / actual_dist * 100 if actual_dist > 1 else 0

    # ── 상단: 예측 박스 | 예측 vs 실제 비교 ──────────────────
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    top_l, top_r = st.columns([1, 1.5])

    with top_l:
        st.markdown(
            f'<div class="pred-box">'
            f'<div style="color:{BMW_LIGHT};font-size:.8rem;letter-spacing:2px;'
            f'text-transform:uppercase;margin-bottom:8px">예측 주행거리</div>'
            f'<div class="pred-num">{pred_dist:,.1f}<span class="pred-unit"> km</span></div>'
            f'<div class="pred-label">{trip_name} · {series_label}</div>'
            f'</div>', unsafe_allow_html=True)

    with top_r:
        cmp = go.Figure()
        cmp.add_trace(go.Bar(
            x=['예측 주행거리', '실제 주행거리'],
            y=[pred_dist, actual_dist],
            marker_color=[BMW_BLUE, GREEN],
            text=[f'{pred_dist:.1f} km', f'{actual_dist:.1f} km'],
            textposition='outside', textfont={'color': TXT, 'size': 13}, width=0.4,
        ))
        cmp.update_layout(
            height=220, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'km',
                   'range': [0, max(pred_dist, actual_dist, 1) * 1.3]},
            xaxis={'color': TXT},
        )
        st.plotly_chart(cmp, use_container_width=True, config={'displayModeBar': False})

    # ── 인사이트 ─────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    err_color = GREEN if err_pct < 10 else AMBER if err_pct < 20 else RED
    for _col, icon, label, value in [
        (i1, '📍', '예측 오차율',
         f'<span style="color:{err_color};font-size:1.3rem;font-weight:700">{err_pct:.1f}%</span>'
         f'<br><span style="font-size:.85rem;color:{SUB}">예측 {pred_dist:.1f} km · 실제 {actual_dist:.1f} km</span>'),
        (i2, '🚗', '주행 특성',
         f'<span style="font-size:1.0rem;font-weight:600;color:{TXT}">{feats["Velocity_mean"]:.1f} km/h</span>'
         f'<br><span style="font-size:.85rem;color:{SUB}">평균속도 · SoC 편차 {feats["SoC_lag1_std"]:.2f} %</span>'),
        (i3, '🔋', '배터리',
         f'<span style="font-size:1.0rem;font-weight:600;color:{TXT}">{feats["Battery_Current_max"]:.1f} A</span>'
         f'<br><span style="font-size:.85rem;color:{SUB}">최대 전류 · 에어컨 {feats["AirCon_Power_lag1_mean"]:.2f} kW</span>'),
    ]:
        _col.markdown(
            f'<div style="background:{PANEL};border-radius:10px;padding:14px 16px;'
            f'border-left:3px solid {BMW_BLUE};min-height:72px">'
            f'<div style="font-size:.75rem;color:{SUB};margin-bottom:4px">{label}</div>'
            f'<div>{value}</div></div>',
            unsafe_allow_html=True)

    # ── 샘플링 단위별 속도 비교 (단일 다중 라인) ─────────────
    if t_col_name and v_col_name:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">샘플링 단위별 속도 프로파일 비교</div>',
                    unsafe_allow_html=True)

        t_raw  = pd.to_numeric(trip_df[t_col_name], errors='coerce').reset_index(drop=True)
        v_raw  = pd.to_numeric(trip_df[v_col_name], errors='coerce').reset_index(drop=True)
        dt_med = float(t_raw.diff().median())

        intervals_s = [None, 1.0, 3.0, 5.0]
        ts_labels   = ['원본', '1초', '3초', '5초']
        ts_colors   = [BMW_BLUE, GREEN, AMBER, RED]
        ts_widths   = [1.0, 1.4, 1.8, 2.2]

        ms_fig = go.Figure()
        for interval, lbl, color, lw in zip(intervals_s, ts_labels, ts_colors, ts_widths):
            if interval is None:
                t_d, v_d = t_raw / 60, v_raw
            elif dt_med > 0 and dt_med < interval - 0.01:
                skip = max(1, round(interval / dt_med))
                t_d = t_raw.iloc[::skip] / 60
                v_d = v_raw.iloc[::skip]
            else:
                t_d, v_d = t_raw / 60, v_raw

            ms_fig.add_trace(go.Scatter(
                x=t_d, y=v_d, mode='lines', name=lbl,
                line={'color': color, 'width': lw},
            ))

        ms_fig.update_layout(
            height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=50, r=20, t=20, b=40),
            xaxis={'gridcolor': LINE, 'color': SUB, 'tickfont': {'size': 9},
                   'title': {'text': '시간 (분)', 'font': {'size': 10, 'color': SUB}}},
            yaxis={'gridcolor': LINE, 'color': SUB, 'tickfont': {'size': 9},
                   'title': {'text': '속도 (km/h)', 'font': {'size': 10, 'color': SUB}}},
            legend={'font': {'color': SUB, 'size': 10}, 'bgcolor': 'rgba(0,0,0,0)',
                    'orientation': 'h', 'x': 0.01, 'y': 1.02},
        )
        st.plotly_chart(ms_fig, use_container_width=True, config={'displayModeBar': False})
        st.caption('샘플링 간격이 클수록 속도 세부 변화가 평탄화됩니다.')


# ════════════════════════════════════════════════════════════════
# PAGE 3 : 주행거리 예측 (슬라이더)
# ════════════════════════════════════════════════════════════════
elif page == "주행거리 예측":
    st.markdown('<div class="bmw-title">주행거리 예측</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">슬라이더로 주행 조건을 설정하거나 사이드바에서 트립 CSV를 업로드하면 AI가 주행거리를 실시간 예측합니다</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    _upload_summary = None
    if uploaded_pred is not None:
        _df_up = None
        for _enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                uploaded_pred.seek(0)
                _df_try = pd.read_csv(uploaded_pred, sep=';', encoding=_enc)
                if _df_try.shape[1] >= 5:
                    _df_up = _df_try
                    break
            except Exception:
                continue
        if _df_up is not None and not _df_up.empty:
            _feats_up, _dist_up, _soc_s_up, _soc_e_up = compute_trip_features(_df_up)
            _upload_summary = (_feats_up, _dist_up, _soc_s_up, _soc_e_up)

            for feat in cols:
                if feat in CAT_FEATURES:
                    st.session_state[f'pred_{feat}'] = int(_feats_up.get(feat, 0))
                else:
                    _val = float(_feats_up.get(feat, float(medians.get(feat, 0))))
                    if feat in PRIMARY and feat in META:
                        _, _, _lo, _hi, _ = META[feat]
                        _val = float(np.clip(_val, _lo, _hi))
                    else:
                        _med = float(medians.get(feat, 0))
                        _hi2 = float(max(_med * 2.5, _med + 1))
                        _val = float(np.clip(_val, 0.0, _hi2))
                    st.session_state[f'pred_{feat}'] = _val
        else:
            st.error('CSV 파싱 실패 — 세미콜론(;) 구분 파일인지 확인하세요.')

    if _upload_summary:
        _feats_up, _dist_up, _soc_s_up, _soc_e_up = _upload_summary
        us1, us2, us3, us4, us5 = st.columns(5)
        for _uc, _uv, _ul in [
            (us1, f'{_dist_up:.1f} km',                                  '실제 주행거리'),
            (us2, f'{_feats_up["Velocity_mean"]:.1f} km/h',              '평균 속도'),
            (us3, f'{_feats_up["Battery_Current_max"]:.1f} A',           '배터리 최대 전류'),
            (us4, f'{_feats_up["SoC_lag1_std"]:.2f} %',                  'SoC 변화 편차'),
            (us5, f'{_feats_up["AirCon_Power_lag1_mean"]:.2f} kW',       '에어컨 출력'),
        ]:
            _uc.markdown(
                f'<div class="kpi" style="border-color:{BMW_BLUE}">'
                f'<div class="v" style="font-size:1.2rem">{_uv}</div>'
                f'<div class="l">{_ul}</div></div>',
                unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="sec-head">주행 조건 설정</div>', unsafe_allow_html=True)
        inputs = {}
        c1, c2 = st.columns(2)
        for i, feat in enumerate(PRIMARY):
            if feat not in cols:
                continue
            label, unit, lo, hi, step = META[feat]
            default = float(medians.get(feat, (lo + hi) / 2))
            if feat == 'SOC_Consumed':
                default = default * 100
            default = float(np.clip(default, lo, hi))
            tgt = c1 if i % 2 == 0 else c2
            inputs[feat] = tgt.slider(f'{label} ({unit})', lo, hi, default, step,
                                       key=f'pred_{feat}')

        with st.expander('경로 · 날씨 조건'):
            _cat_labels = {
                'Route_Area_Munich_East':                '📍 뮌헨 동부',
                'Route_Area_FTMRoute_2x':                '🔁 FTM 경로 (2배)',
                'Route_Area_Highway':                    '🛣 고속도로',
                'Route_Area_Munich_North_Fast_Charging': '⚡ 뮌헨 북부 급속충전',
                'Weather_rainy':                         '🌧 비 날씨',
            }
            _cc1, _cc2 = st.columns(2)
            for _ci, feat in enumerate([f for f in cols if f in CAT_FEATURES]):
                _lbl = _cat_labels.get(feat, feat.replace('_', ' '))
                _def = bool(int(st.session_state.get(f'pred_{feat}', 0)))
                _tgt = _cc1 if _ci % 2 == 0 else _cc2
                inputs[feat] = float(_tgt.checkbox(_lbl, value=_def, key=f'pred_{feat}'))

    row = {feat: inputs.get(feat, float(medians.get(feat, 0))) for feat in cols}
    X_one = pd.DataFrame([row])[cols]
    for _c in CAT_FEATURES:
        if _c in X_one.columns:
            X_one[_c] = X_one[_c].astype(int)
    pred = float(max(model.predict(X_one)[0], 0.0))

    v_mean      = inputs.get('Velocity_mean', 40)
    bc_max_inp  = inputs.get('Battery_Current_max', 80)
    aircon_inp  = inputs.get('AirCon_Power_lag1_mean', 1.0)
    soc_std_inp = inputs.get('SoC_lag1_std', 2.0)

    with right:
        st.markdown(
            f'<div class="pred-box">'
            f'<div style="color:{BMW_LIGHT};font-size:.8rem;letter-spacing:2px;'
            f'text-transform:uppercase;margin-bottom:8px">예측 주행거리</div>'
            f'<div class="pred-num">{pred:,.1f}<span class="pred-unit"> km</span></div>'
            f'</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode='gauge+number', value=v_mean,
            number={'suffix': ' km/h', 'font': {'size': 28, 'color': TXT, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 110], 'tickcolor': SUB, 'tickfont': {'color': SUB, 'size': 10}},
                'bar': {'color': BMW_BLUE, 'thickness': 0.3},
                'bgcolor': PANEL2, 'borderwidth': 0,
                'steps': [{'range': [0, 40], 'color': '#0d2040'},
                           {'range': [40, 75], 'color': '#102a50'},
                           {'range': [75, 110], 'color': '#0f3060'}],
                'threshold': {'line': {'color': AMBER, 'width': 3}, 'thickness': 0.8,
                              'value': min(v_mean * 1.8, 110)},
            }))
        gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', font={'color': TXT})
        st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
        st.caption(f'파란 바 = 현재 평균 속도 ({v_mean:.0f} km/h)')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col2, val, lab in [
        (k1, f'{v_mean:,.1f} km/h',    '평균 속도'),
        (k2, f'{bc_max_inp:,.1f} A',   '배터리 최대 전류'),
        (k3, f'{aircon_inp:,.2f} kW',  '에어컨 출력'),
        (k4, f'{soc_std_inp:,.2f} %',  'SoC 변화 편차'),
    ]:
        col2.markdown(f'<div class="kpi"><div class="v">{val}</div>'
                      f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-head">평균 속도 민감도 분석</div>', unsafe_allow_html=True)

        vel_range = np.linspace(10, 110, 50)
        preds_vel = []
        for vv in vel_range:
            r_tmp = dict(row); r_tmp['Velocity_mean'] = float(vv)
            _df_tmp = pd.DataFrame([r_tmp])[cols]
            for _c in CAT_FEATURES:
                if _c in _df_tmp.columns:
                    _df_tmp[_c] = _df_tmp[_c].astype(int)
            preds_vel.append(float(max(model.predict(_df_tmp)[0], 0)))

        vel_slope = (preds_vel[-1] - preds_vel[0]) / (vel_range[-1] - vel_range[0])
        cur_vel_idx = int(np.argmin(np.abs(vel_range - v_mean)))
        cur_pred_at_vel = preds_vel[cur_vel_idx]

        vel_line = go.Figure()
        vel_line.add_trace(go.Scatter(
            x=vel_range, y=preds_vel, mode='lines',
            line={'color': BMW_BLUE, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.15)',
            showlegend=False,
        ))
        vel_line.add_trace(go.Scatter(
            x=[v_mean], y=[cur_pred_at_vel], mode='markers',
            marker={'color': AMBER, 'size': 10, 'symbol': 'circle',
                    'line': {'color': TXT, 'width': 1.5}},
            showlegend=False,
        ))
        vel_line.add_vline(x=v_mean, line_color=AMBER, line_dash='dash', line_width=1.5,
                           annotation_text=f'현재 {v_mean:.0f} km/h',
                           annotation_font_color=AMBER, annotation_font_size=10)
        vel_line.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '평균 속도 (km/h)', 'font': {'size': 11, 'color': SUB}}},
            yaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '예측 주행거리 (km)', 'font': {'size': 11, 'color': SUB}}},
            showlegend=False,
        )
        st.plotly_chart(vel_line, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div class="insight">
          현재 평균 속도 <strong>{v_mean:.0f} km/h</strong> →
          예측 <strong style="color:{BMW_LIGHT}">{cur_pred_at_vel:.1f} km</strong>
          &nbsp;|&nbsp; 속도 10 km/h 변화 시 약 <strong>{abs(vel_slope)*10:.1f} km</strong> 차이
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-head">배터리 최대 전류 민감도 분석</div>', unsafe_allow_html=True)

        bc_range = np.linspace(10, 200, 60)
        preds_bc = []
        for bv in bc_range:
            r_tmp = dict(row); r_tmp['Battery_Current_max'] = float(bv)
            _df_tmp = pd.DataFrame([r_tmp])[cols]
            for _c in CAT_FEATURES:
                if _c in _df_tmp.columns:
                    _df_tmp[_c] = _df_tmp[_c].astype(int)
            preds_bc.append(float(max(model.predict(_df_tmp)[0], 0)))

        bc_slope = (preds_bc[-1] - preds_bc[0]) / (bc_range[-1] - bc_range[0])
        cur_bc_idx = int(np.argmin(np.abs(bc_range - bc_max_inp)))
        cur_pred_at_bc = preds_bc[cur_bc_idx]

        bc_line = go.Figure()
        bc_line.add_trace(go.Scatter(
            x=bc_range, y=preds_bc, mode='lines',
            line={'color': BMW_BLUE, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.15)',
            showlegend=False,
        ))
        bc_line.add_trace(go.Scatter(
            x=[bc_max_inp], y=[cur_pred_at_bc], mode='markers',
            marker={'color': AMBER, 'size': 10, 'symbol': 'circle',
                    'line': {'color': TXT, 'width': 1.5}},
            showlegend=False,
        ))
        bc_line.add_vline(x=bc_max_inp, line_color=AMBER, line_dash='dash', line_width=1.5,
                          annotation_text=f'현재 {bc_max_inp:.0f} A',
                          annotation_font_color=AMBER, annotation_font_size=10)
        bc_line.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '배터리 최대 전류 (A)', 'font': {'size': 11, 'color': SUB}}},
            yaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '예측 주행거리 (km)', 'font': {'size': 11, 'color': SUB}}},
            showlegend=False,
        )
        st.plotly_chart(bc_line, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div class="insight">
          현재 배터리 최대 전류 <strong>{bc_max_inp:.0f} A</strong> →
          예측 <strong style="color:{BMW_LIGHT}">{cur_pred_at_bc:.1f} km</strong>
          &nbsp;|&nbsp; 10 A 변화 시 약 <strong>{abs(bc_slope)*10:.1f} km</strong> 차이
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4 : 모델 분석 (성능 + 비교 통합)
# ════════════════════════════════════════════════════════════════
elif page == "모델 분석":
    st.markdown('<div class="bmw-title">모델 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">예측 성능 · 잔차 분석 · Optuna / GridSearch 튜닝 비교</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    _df_base_kpi, _df_opt_kpi = load_model_comparison()

    # 최고 R²: 베이스라인 > 현재 모델 순으로 fallback
    if not _df_base_kpi.empty and 'R2' in _df_base_kpi.columns:
        best_r2  = float(_df_base_kpi['R2'].max())
    else:
        best_r2  = metrics['R2']
    # 최저 RMSE: 튜닝 결과 > 현재 모델 순으로 fallback
    if not _df_opt_kpi.empty and 'RMSE_mean' in _df_opt_kpi.columns:
        _rmse_min = _df_opt_kpi['RMSE_mean'].min()
        best_rmse = float(_rmse_min) if not pd.isna(_rmse_min) else metrics['RMSE']
    else:
        best_rmse = metrics['RMSE']

    k1, k2, k3, k4 = st.columns(4)
    for col2, val, lab, vc in [
        (k1, f"{best_r2:.4f}",          "최고 R² (베이스라인)", GREEN),
        (k2, f"{best_rmse:.2f} km",     "최저 RMSE (튜닝 후)", GREEN),
        (k3, f"{metrics['MAE']:.2f} km", "현재 MAE",           TXT),
        (k4, f"{metrics['n']}건",        "전체 데이터",          TXT),
    ]:
        col2.markdown(
            f'<div class="kpi"><div class="v" style="color:{vc}">{val}</div>'
            f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    yte       = metrics['yte']
    pred_te   = metrics['pred_te']
    residuals = yte - pred_te
    abs_err   = np.sort(np.abs(residuals))
    cdf_y     = np.arange(1, len(abs_err) + 1) / len(abs_err)
    p50 = float(np.percentile(abs_err, 50))
    p80 = float(np.percentile(abs_err, 80))

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-head">예측값 vs 실제값</div>', unsafe_allow_html=True)
        sc = go.Figure()
        sc.add_trace(go.Scatter(
            x=yte, y=pred_te, mode='markers',
            marker={'color': BMW_BLUE, 'size': 5, 'opacity': 0.65}, name='예측 vs 실제',
        ))
        lim = max(yte.max(), pred_te.max()) * 1.05
        sc.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode='lines',
            line={'color': AMBER, 'dash': 'dash', 'width': 1.5}, name='완벽 예측선',
        ))
        sc.update_layout(
            height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': '실제 주행거리 (km)'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '예측 주행거리 (km)'},
            legend={'font': {'color': SUB}},
        )
        st.plotly_chart(sc, use_container_width=True, config={'displayModeBar': False})

    with col_r:
        st.markdown('<div class="sec-head">잔차 분포</div>', unsafe_allow_html=True)
        res_hist = go.Figure()
        res_hist.add_trace(go.Histogram(
            x=residuals, nbinsx=35, marker_color=BMW_BLUE, opacity=0.8,
        ))
        res_hist.add_vline(x=0, line_color=AMBER, line_width=2,
                           annotation_text='잔차=0', annotation_font_color=AMBER)
        res_hist.update_layout(
            height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': '잔차 (실제-예측) km'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '빈도'},
            showlegend=False,
        )
        st.plotly_chart(res_hist, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="sec-head">절대 오차 누적 분포 (CDF)</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{SUB};font-size:.83rem;margin-bottom:12px">'
        '가로축: 예측 오차(km) · 세로축: 그 오차 이하로 맞출 확률(%) — '
        '곡선이 왼쪽 위로 가파를수록 오차가 작고 정확한 모델입니다.</div>',
        unsafe_allow_html=True)

    p90 = float(np.percentile(abs_err, 90))
    pc1, pc2, pc3 = st.columns(3)
    for _pc, _pv, _lab, _col in [
        (pc1, p50, '절반(50%)의 예측이\n이 오차 이내', GREEN),
        (pc2, p80, '80%의 예측이\n이 오차 이내', AMBER),
        (pc3, p90, '90%의 예측이\n이 오차 이내', RED),
    ]:
        _pc.markdown(
            f'<div class="kpi" style="border-color:{_col}">'
            f'<div class="v" style="color:{_col};font-size:1.6rem">±{_pv:.1f} km</div>'
            f'<div class="l" style="white-space:pre-line">{_lab}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    cdf_fig = go.Figure()
    cdf_fig.add_trace(go.Scatter(
        x=abs_err, y=cdf_y * 100, mode='lines',
        line={'color': BMW_BLUE, 'width': 2.5},
        fill='tozeroy', fillcolor='rgba(28,105,212,0.12)',
        name='CDF', showlegend=False,
    ))
    cdf_fig.add_vline(x=p50, line_color=GREEN, line_dash='dash',
                      annotation_text=f'50%ile · {p50:.1f} km',
                      annotation_font_color=GREEN, annotation_font_size=10)
    cdf_fig.add_vline(x=p80, line_color=AMBER, line_dash='dash',
                      annotation_text=f'80%ile · {p80:.1f} km',
                      annotation_font_color=AMBER, annotation_font_size=10)
    cdf_fig.add_vline(x=p90, line_color=RED, line_dash='dash',
                      annotation_text=f'90%ile · {p90:.1f} km',
                      annotation_font_color=RED, annotation_font_size=10)
    cdf_fig.update_layout(
        height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis={'gridcolor': LINE, 'color': SUB,
               'title': {'text': '절대 오차 (km)', 'font': {'size': 11, 'color': SUB}}},
        yaxis={'gridcolor': LINE, 'color': SUB,
               'title': {'text': '누적 확률 (%)', 'font': {'size': 11, 'color': SUB}},
               'range': [0, 103]},
        showlegend=False,
    )
    st.plotly_chart(cdf_fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown(f"""
    <div class="insight">
      잔차 평균 <strong>{residuals.mean():.2f} km</strong>
      (0에 가까울수록 편향 없음) &nbsp;·&nbsp;
      표준편차 <strong>{residuals.std():.2f} km</strong> &nbsp;·&nbsp;
      예측 10개 중 약 <strong style="color:{GREEN}">5개</strong>는 ±{p50:.1f} km,
      <strong style="color:{AMBER}">8개</strong>는 ±{p80:.1f} km 이내로 맞춥니다
    </div>
    """, unsafe_allow_html=True)

    # ── Optuna / GridSearch 튜닝 비교 ────────────────────────
    _, df_opt = load_model_comparison()

    if not df_opt.empty:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">Optuna · GridSearch 튜닝 성능 비교</div>',
                    unsafe_allow_html=True)
        df_rmse = df_opt[df_opt['RMSE_mean'].notna()].copy()
        df_rmse['Method_short'] = (df_rmse['Method']
                                   .str.replace(r'\s*\(.*\)', '', regex=True).str.strip())
        df_rmse['Label'] = (df_rmse['Notebook'] + ' | '
                            + df_rmse['Model'] + ' | ' + df_rmse['Method_short'])

        opt_fig = go.Figure()
        for nb, nb_color in [('AB통합', BMW_BLUE), ('B만', GREEN)]:
            sub = df_rmse[df_rmse['Notebook'] == nb]
            if sub.empty:
                continue
            opt_fig.add_trace(go.Bar(
                name=nb, x=sub['Label'], y=sub['RMSE_mean'],
                marker_color=nb_color,
                error_y=dict(type='data', array=sub['RMSE_std'].fillna(0).tolist(),
                             visible=True, color=AMBER),
                text=[f'{v:.2f}' for v in sub['RMSE_mean']],
                textposition='outside', textfont={'color': TXT, 'size': 10},
            ))
        opt_fig.update_layout(
            height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=110),
            barmode='group',
            xaxis={'color': TXT, 'tickangle': -35},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'RMSE (km)'},
            legend={'font': {'color': SUB}, 'bgcolor': 'rgba(0,0,0,0)'},
        )
        st.plotly_chart(opt_fig, use_container_width=True, config={'displayModeBar': False})

        ab_best = (df_rmse[df_rmse['Notebook'] == 'AB통합']['RMSE_mean'].min()
                   if 'AB통합' in df_rmse['Notebook'].values else None)
        b_best  = (df_rmse[df_rmse['Notebook'] == 'B만']['RMSE_mean'].min()
                   if 'B만' in df_rmse['Notebook'].values else None)
        if ab_best is not None and b_best is not None:
            st.markdown(f"""
            <div class="insight">튜닝 후 최저 RMSE — AB통합: <strong>{ab_best:.3f} km</strong> ·
            B만: <strong>{b_best:.3f} km</strong> (AB통합이 {b_best - ab_best:.3f} km 낮음)</div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 5 : 변수 분석 (인사이트 + 중요도 통합)
# ════════════════════════════════════════════════════════════════
elif page == "변수 분석":
    st.markdown('<div class="bmw-title">변수 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">모델이 주행거리 예측에 어떤 변수를 얼마나 중요하게 사용하는지 한눈에 확인합니다</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Section 1 : 중요도 바차트 + 핵심 변수 카드 ───────────
    fi_l, fi_r = st.columns([1.6, 1])

    FEAT_MEANINGS = {
        'SoC_lag1_std':                  'SoC 변화율 편차가 클수록 불안정한 에너지 흐름을 나타냅니다 (SHAP #1)',
        'Elevation_MA3_std':             '고도 변동(MA3)이 클수록 언덕·경사 구간이 많은 경로입니다 (SHAP #2)',
        'Battery_Current_max':           '최대 전류는 급가속·고부하 구간의 피크 에너지 소모를 반영합니다 (SHAP #3)',
        'Velocity_mean':                 '평균 속도가 높을수록 장거리·고속 주행을 의미합니다 (SHAP #4)',
        'Throttle_lag1_std':             '스로틀 변화 편차가 클수록 가감속이 잦은 주행 패턴입니다 (SHAP #5)',
        'Battery_Temperature_diff_mean': '배터리 온도 변화 평균은 열 관리 효율을 나타냅니다 (SHAP #6)',
        'Route_Area_Munich_East':        '뮌헨 동부 경로는 특정 지형·신호 패턴이 주행거리에 영향을 줍니다 (SHAP #7)',
        'Weather_rainy':                 '강우 시 저속·안전운전으로 에너지 소모 패턴이 달라집니다 (SHAP #8)',
        'AirCon_Power_lag1_mean':        '에어컨 출력이 높을수록 HVAC 에너지 소모로 주행거리가 감소합니다 (SHAP #9)',
        'Throttle_lag1_mean':            '평균 스로틀이 높을수록 적극적인 가속으로 에너지 소모가 큽니다 (SHAP #10)',
        'Route_Area_FTMRoute_2x':        'FTM 반복 경로는 고정된 거리·조건으로 예측 신뢰도를 높입니다 (SHAP #11)',
        'Route_Area_Highway':            '고속도로 주행은 안정적 속도로 에너지 효율이 달라집니다 (SHAP #12)',
        'Route_Area_Munich_North_Fast_Charging': '급속충전 경유 경로는 특정 거리 패턴을 나타냅니다 (SHAP #13)',
    }

    total_imp = importance['Importance'].sum()
    max_imp   = importance['Importance'].max()

    with fi_l:
        st.markdown('<div class="sec-head">변수 중요도 (상위 10개)</div>', unsafe_allow_html=True)
        top10 = importance.head(10).iloc[::-1].copy()
        labels10 = [META[f][0] if f in META else f.replace('_', ' ')
                    for f in top10['Feature']]
        pct10    = top10['Importance'] / total_imp * 100
        c10 = [f'rgba(28,105,212,{0.40 + 0.60 * v / max_imp:.2f})'
               for v in top10['Importance']]
        fi_fig = go.Figure(go.Bar(
            x=top10['Importance'], y=labels10, orientation='h',
            marker_color=c10,
            text=[f'{p:.1f}%' for p in pct10],
            textposition='outside', textfont={'color': SUB, 'size': 10},
        ))
        fi_fig.update_layout(
            height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=55, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '중요도 점수', 'font': {'size': 10, 'color': SUB}}},
            yaxis={'color': TXT, 'tickfont': {'size': 11}},
        )
        st.plotly_chart(fi_fig, use_container_width=True, config={'displayModeBar': False})

    with fi_r:
        st.markdown('<div class="sec-head">핵심 변수 TOP 3</div>', unsafe_allow_html=True)
        card_colors = [BMW_BLUE, GREEN, AMBER]
        for rank, (_, row_fi) in enumerate(importance.head(3).iterrows()):
            fname  = row_fi['Feature']
            flabel = META[fname][0] if fname in META else fname.replace('_', ' ')
            pct    = row_fi['Importance'] / total_imp * 100
            bar_w  = int(row_fi['Importance'] / max_imp * 100)
            meaning = FEAT_MEANINGS.get(fname, '주행거리 예측의 핵심 변수입니다')
            cc = card_colors[rank]
            st.markdown(f"""
            <div style="background:{PANEL};border-radius:10px;padding:14px 16px;
                        border-left:4px solid {cc};margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:6px">
                <span style="color:{TXT};font-weight:700;font-size:.95rem">
                  #{rank+1} {flabel}
                </span>
                <span style="color:{cc};font-weight:700;font-size:1.0rem">{pct:.1f}%</span>
              </div>
              <div style="background:#1E3A5F;border-radius:3px;height:6px;margin-bottom:8px">
                <div style="width:{bar_w}%;background:{cc};border-radius:3px;height:6px"></div>
              </div>
              <div style="color:{SUB};font-size:.8rem">{meaning}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Section 2 : 주행 패턴 산점도 + 속도구간 박스플롯 ────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">주행 패턴 인사이트</div>', unsafe_allow_html=True)
    ins_l, ins_r = st.columns(2)

    with ins_l:
        if 'Velocity_mean' in df.columns and 'Distance' in df.columns:
            sc2 = go.Figure()
            sc2.add_trace(go.Scatter(
                x=df['Velocity_mean'], y=df['Distance'], mode='markers',
                marker={
                    'color': df['SOC_Consumed'] * 100 if 'SOC_Consumed' in df.columns
                             else BMW_BLUE,
                    'colorscale': [[0, '#0a2040'], [0.5, BMW_BLUE], [1, '#7ab8f5']],
                    'size': 6, 'opacity': 0.70,
                    'colorbar': {'title': {'text': 'SOC 소모(%)', 'font': {'color': SUB, 'size': 10}},
                                 'tickfont': {'color': SUB, 'size': 9}},
                    'showscale': 'SOC_Consumed' in df.columns,
                },
            ))
            sc2.update_layout(
                title={'text': '평균 속도 vs 주행거리 (색: SOC 소모)',
                       'font': {'color': TXT, 'size': 12}, 'x': 0.03},
                height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=80, t=42, b=42),
                xaxis={'gridcolor': LINE, 'color': SUB,
                       'title': {'text': '평균 속도 (km/h)', 'font': {'size': 10, 'color': SUB}}},
                yaxis={'gridcolor': LINE, 'color': SUB,
                       'title': {'text': '주행거리 (km)', 'font': {'size': 10, 'color': SUB}}},
                showlegend=False,
            )
            st.plotly_chart(sc2, use_container_width=True, config={'displayModeBar': False})

    with ins_r:
        if 'Velocity_mean' in df.columns and 'Distance' in df.columns:
            bins = [0, 30, 50, 70, 90, 160]
            labels_bin = ['0–30', '30–50', '50–70', '70–90', '90+']
            df_tmp = df.copy()
            df_tmp['속도구간'] = pd.cut(df_tmp['Velocity_mean'], bins=bins, labels=labels_bin)
            bx_palette = ['#0a2d5a', '#0e3d7a', BMW_BLUE, BMW_LIGHT, '#7ab8f5']
            box_fig = go.Figure()
            for i, grp in enumerate(labels_bin):
                d = df_tmp[df_tmp['속도구간'] == grp]['Distance']
                if len(d):
                    box_fig.add_trace(go.Box(
                        y=d, name=f'{grp} km/h',
                        marker_color=bx_palette[i % len(bx_palette)],
                        line_color=BMW_LIGHT, boxmean=True,
                    ))
            box_fig.update_layout(
                title={'text': '속도 구간별 주행거리 분포',
                       'font': {'color': TXT, 'size': 12}, 'x': 0.03},
                height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=42, b=42),
                xaxis={'color': TXT,
                       'title': {'text': '속도 구간 (km/h)', 'font': {'size': 10, 'color': SUB}}},
                yaxis={'gridcolor': LINE, 'color': SUB,
                       'title': {'text': '주행거리 (km)', 'font': {'size': 10, 'color': SUB}}},
                showlegend=False,
            )
            st.plotly_chart(box_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption('데이터: BMW i3 Battery & Heating Data in Real Driving Cycles (FTM, TU München)')
