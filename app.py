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

FEATURES = [
    'Duration', 'SOC_Consumed', 'Battery_Temperature_std', 'Velocity_mean',
    'Battery_Temperature_diff_max', 'Longitudinal_Acceleration_diff_std',
    'Accel_abs_mean', 'Velocity_diff_std', 'Longitudinal_Acceleration_std',
    'Velocity_std', 'Battery_State_of_Charge_End', 'Heating_Power_CAN_std',
    'Heating_Power_CAN_mean', 'Accel_abs_std', 'Battery_Current_std',
    'Battery_Power_mean', 'Accel_abs_max', 'Ambient_Temperature_std',
    'Velocity_max', 'Battery_Voltage_mean',
]
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
    'Velocity_mean':                    ('평균 속도',           'km/h', 0.0, 110.0, 1.0),
    'Velocity_max':                     ('최고 속도',           'km/h', 0.0, 160.0, 1.0),
    'Velocity_std':                     ('속도 표준편차',       'km/h', 0.0, 50.0,  0.5),
    'Velocity_diff_std':                ('속도 변화 편차',      'km/h', 0.0, 10.0,  0.1),
    'Duration':                         ('주행 시간',           'min',  1.0, 100.0, 0.5),
    'SOC_Consumed':                     ('배터리 소모량',       '%',    0.0, 60.0,  0.5),
    'Accel_abs_mean':                   ('평균 가속도',         'm/s²', 0.0, 2.0,   0.05),
    'Accel_abs_std':                    ('가속도 편차',         'm/s²', 0.0, 1.5,   0.05),
    'Accel_abs_max':                    ('최대 가속도',         'm/s²', 0.0, 5.0,   0.1),
    'Longitudinal_Acceleration_std':    ('종방향 가속 편차',    'm/s²', 0.0, 3.0,   0.05),
    'Longitudinal_Acceleration_diff_std':('종방향 가속 변화 편차','m/s²',0.0, 2.0,  0.05),
    'Heating_Power_CAN_mean':           ('평균 난방 출력',      'kW',   0.0, 6.0,   0.1),
    'Heating_Power_CAN_std':            ('난방 출력 편차',      'kW',   0.0, 3.0,   0.1),
    'Battery_Power_mean':               ('평균 배터리 출력',    'kW',   0.0, 30.0,  0.5),
    'Battery_Temperature_std':          ('배터리 온도 변동',    '°C',   0.0, 6.0,   0.1),
    'Battery_Temperature_diff_max':     ('배터리 온도 최대변화','°C',   0.0, 10.0,  0.1),
    'Battery_Current_std':              ('배터리 전류 편차',    'A',    0.0, 50.0,  1.0),
    'Battery_Voltage_mean':             ('평균 배터리 전압',    'V',    300.0,420.0, 1.0),
    'Battery_State_of_Charge_End':      ('최종 충전량',         '%',    0.0, 100.0, 1.0),
    'Ambient_Temperature_std':          ('외기온 변동',         '°C',   0.0, 4.0,   0.1),
}
PRIMARY = ['Velocity_mean', 'Velocity_max', 'Duration', 'SOC_Consumed',
           'Accel_abs_mean', 'Heating_Power_CAN_mean', 'Battery_Power_mean',
           'Battery_Temperature_std']

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
    vmean = rng.uniform(18, 95, n)
    dur = rng.uniform(5, 95, n)
    dist = vmean * dur / 60 * rng.normal(1, 0.05, n)
    dist = np.clip(dist, 1.5, None)
    soc = np.clip(dist / 120 + rng.normal(0, 0.02, n), 0.01, 0.6)
    df = pd.DataFrame({
        'Distance': dist, 'Duration': dur, 'SOC_Consumed': soc,
        'Velocity_mean': vmean,
        'Velocity_max': np.clip(vmean * rng.uniform(1.4, 2.2, n), vmean, 160),
        'Velocity_std': vmean * rng.uniform(0.30, 0.60, n),
        'Battery_Temperature_std': rng.uniform(0.3, 5.0, n),
        'Battery_Temperature_diff_max': rng.uniform(0.1, 2.0, n),
        'Longitudinal_Acceleration_diff_std': rng.uniform(0.05, 0.6, n),
        'Accel_abs_mean': rng.uniform(0.2, 1.2, n),
        'Velocity_diff_std': rng.uniform(0.5, 4.0, n),
        'Longitudinal_Acceleration_std': rng.uniform(0.2, 1.0, n),
        'Battery_State_of_Charge_End': np.clip(0.9 - soc + rng.normal(0, 0.05, n), 0.05, 0.95),
        'Heating_Power_CAN_std': rng.uniform(0, 2.0, n),
        'Heating_Power_CAN_mean': rng.uniform(0, 5.0, n),
        'Accel_abs_std': rng.uniform(0.2, 1.0, n),
        'Battery_Current_std': rng.uniform(10, 60, n),
        'Battery_Power_mean': rng.uniform(2, 25, n),
        'Accel_abs_max': rng.uniform(1, 4, n),
        'Ambient_Temperature_std': rng.uniform(0.1, 3.0, n),
        'Battery_Voltage_mean': rng.uniform(330, 400, n),
    })
    return df


@st.cache_data(show_spinner=False)
def load_data():
    candidates = ['df_final_vif.csv', 'data/df_final_vif.csv', './df_final_vif.csv']
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p), 'real'
    return make_synthetic(), 'synthetic'


@st.cache_resource(show_spinner=False)
def train_model(df_key, df):
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].copy()
    y = df[TARGET].copy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    if HAS_CATBOOST:
        model = CatBoostRegressor(**BEST_PARAMS, loss_function='RMSE',
                                  random_seed=42, verbose=False)
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


def compute_trip_features(df):
    def safe(s, default=0.0):
        return s if s is not None else pd.Series([default] * len(df))

    t   = safe(_get_col(df, ['time [s]', 'time']), 0.0)
    v   = safe(_get_col(df, ['velocity']), 0.0)
    a   = safe(_get_col(df, ['longitudinal acceleration']), 0.0)
    bt  = safe(_get_col(df, ['battery temperature'],
                        exclude=['max', 'min', 'coolant', 'exchanger', 'heater', 'cabin', 'inlet']), 20.0)
    soc = safe(_get_col(df, ['soc [%]'], exclude=['max', 'min', 'displayed']), 50.0)
    ht  = safe(_get_col(df, ['heating power can']), 0.0)
    bc  = safe(_get_col(df, ['battery current']), 0.0)
    bv  = safe(_get_col(df, ['battery voltage']), 370.0)
    at  = safe(_get_col(df, ['ambient temperature']), 15.0)

    dur            = float((t.max() - t.min()) / 60)
    vel_mean       = float(v.mean())
    vel_std        = float(v.std())
    vel_max        = float(v.max())
    vel_diff_std   = float(v.diff().std())
    a_abs          = a.abs()
    accel_abs_mean = float(a_abs.mean())
    accel_abs_std  = float(a_abs.std())
    accel_abs_max  = float(a_abs.max())
    accel_std      = float(a.std())
    accel_diff_std = float(a.diff().std())
    batt_temp_std      = float(bt.std())
    batt_temp_diff_max = float(bt.diff().abs().max()) if len(bt) > 1 else 0.0
    soc_start    = float(soc.dropna().iloc[0])  if len(soc.dropna()) > 0 else 50.0
    soc_end_val  = float(soc.dropna().iloc[-1]) if len(soc.dropna()) > 0 else 40.0
    soc_consumed = max(0.0, (soc_start - soc_end_val) / 100)
    soc_end      = soc_end_val / 100
    heat_mean = float(ht.mean()); heat_std = float(ht.std())
    batt_curr_std   = float(bc.std())
    batt_volt_mean  = float(bv.mean())
    batt_power_mean = float((bv * bc / 1000).mean())
    amb_temp_std    = float(at.std())
    dt = t.diff().fillna(0.0)
    actual_dist = float((v * dt / 3600).sum())

    feats = {
        'Duration': dur, 'SOC_Consumed': soc_consumed,
        'Battery_Temperature_std': batt_temp_std, 'Velocity_mean': vel_mean,
        'Battery_Temperature_diff_max': batt_temp_diff_max,
        'Longitudinal_Acceleration_diff_std': accel_diff_std,
        'Accel_abs_mean': accel_abs_mean, 'Velocity_diff_std': vel_diff_std,
        'Longitudinal_Acceleration_std': accel_std, 'Velocity_std': vel_std,
        'Battery_State_of_Charge_End': soc_end, 'Heating_Power_CAN_std': heat_std,
        'Heating_Power_CAN_mean': heat_mean, 'Accel_abs_std': accel_abs_std,
        'Battery_Current_std': batt_curr_std, 'Battery_Power_mean': batt_power_mean,
        'Accel_abs_max': accel_abs_max, 'Ambient_Temperature_std': amb_temp_std,
        'Velocity_max': vel_max, 'Battery_Voltage_mean': batt_volt_mean,
    }
    return feats, actual_dist, soc_start, soc_end_val


@st.cache_data(show_spinner=False)
def load_model_comparison():
    base_path = os.path.join('model comparison', 'baseline_model_comparison.csv')
    opt_path  = os.path.join('model comparison', 'optuna_gridsearch_comparison.csv')
    df_base = pd.read_csv(base_path) if os.path.exists(base_path) else pd.DataFrame()
    df_opt  = pd.read_csv(opt_path)  if os.path.exists(opt_path)  else pd.DataFrame()
    return df_base, df_opt


# ── 페이지 설정 + CSS ────────────────────────────────────────────
st.set_page_config(page_title='BMW i3 · EV Dashboard', page_icon='🚗', layout='wide')

st.markdown(f"""
<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

 .stApp {{
   background: linear-gradient(160deg, #0B1929 0%, #0D1F36 50%, #091525 100%) fixed;
   color:{TXT}; font-family:'Inter',sans-serif;
 }}
 #MainMenu, footer, header {{ visibility:hidden; }}
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
        ["데이터 현황", "트립별 주행거리 예측", "주행거리 예측", "모델 분석", "변수 분석"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    selected_trip = None
    if page == "트립별 주행거리 예측":
        if trip_list:
            st.markdown(
                f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
                f'letter-spacing:.5px;margin-bottom:6px">트립 선택</div>',
                unsafe_allow_html=True)
            selected_trip = st.selectbox("trip", trip_list, label_visibility='collapsed')
        else:
            st.warning('data/ 폴더에 트립 파일이 없습니다.')

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
    st.markdown('<div class="bmw-title">BMW i3 · 주행 데이터 현황</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">트립 집계 데이터 탐색 · 변수 분포 · 상관관계</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── 데이터 요약 KPI ───────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    dist_mean = df['Distance'].mean()          if 'Distance'      in df.columns else 0
    dist_std  = df['Distance'].std()           if 'Distance'      in df.columns else 0
    dur_mean  = df['Duration'].mean()          if 'Duration'      in df.columns else 0
    vel_mean  = df['Velocity_mean'].mean()     if 'Velocity_mean' in df.columns else 0
    soc_mean  = df['SOC_Consumed'].mean()*100  if 'SOC_Consumed'  in df.columns else 0

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
            ('Distance',      '주행거리',  'km'),
            ('Velocity_mean', '평균 속도', 'km/h'),
            ('SOC_Consumed',  'SOC 소모',  '%'),
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

    # ── Tab 2 : 변화량 분포 ────────────────────────────────────
    with tab2:
        var_vars_def = [
            ('Velocity_std',            '속도 표준편차',    'km/h'),
            ('Battery_Temperature_std', '배터리 온도 변동', '°C'),
            ('Accel_abs_std',           '절대가속 편차',    'm/s²'),
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
            st.markdown(
                '<div class="insight">값이 클수록 해당 트립의 주행 조건이 불안정했음을 나타냅니다. 주황 점선은 전체 트립 평균입니다.</div>',
                unsafe_allow_html=True)

    # ── Tab 3 : 트립 시계열 ────────────────────────────────────
    with tab3:
        if not trip_list:
            st.info('data/ 폴더에 트립 CSV 파일이 없습니다.')
        else:
            sel_ts_trip = st.selectbox(
                '트립 선택', trip_list,
                format_func=lambda x: x.replace('.csv', ''),
                key='tab3_trip_select',
            )
            ts_df = load_trip_raw(sel_ts_trip)
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
                    t3_panels = [(c, l, u, cl, fc, fi)
                                 for c, l, u, cl, fc, fi in panels_def3
                                 if c and c in ts_df.columns]
                    if t3_panels:
                        n3 = len(t3_panels)
                        t3_fig = make_subplots(
                            rows=n3, cols=1,
                            shared_xaxes=True,
                            row_heights=[1] * n3,
                            vertical_spacing=0.04,
                        )
                        for ri, (cn, lbl, unit, clr, fc, do_fill) in enumerate(t3_panels, 1):
                            y3 = pd.to_numeric(ts_df[cn], errors='coerce')
                            t3_fig.add_trace(go.Scatter(
                                x=t3_t_ser, y=y3, mode='lines', name=lbl,
                                line={'color': clr, 'width': 1.5},
                                fill='tozeroy' if do_fill else 'none',
                                fillcolor=fc if do_fill else None,
                                showlegend=True,
                            ), row=ri, col=1)
                            if 'acceleration' in cn.lower():
                                t3_fig.add_hline(y=0, line_color=SUB, line_dash='dot',
                                                 line_width=0.8, row=ri, col=1)
                            t3_fig.update_yaxes(
                                title_text=f'{lbl}<br>({unit})',
                                title_font={'size': 9, 'color': SUB},
                                gridcolor=LINE, color=SUB, tickfont={'size': 8},
                                row=ri, col=1,
                            )
                        for r in range(1, n3 + 1):
                            t3_fig.update_xaxes(
                                gridcolor=LINE, color=SUB, tickfont={'size': 9},
                                showticklabels=(r == n3),
                                title_text='시간 (분)' if r == n3 else '',
                                title_font={'size': 10, 'color': SUB},
                                row=r, col=1,
                            )
                        t3_fig.update_layout(
                            height=80 + n3 * 130,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=70, r=20, t=16, b=40),
                            legend={'font': {'color': SUB, 'size': 10},
                                    'bgcolor': 'rgba(0,0,0,0)',
                                    'orientation': 'h', 'x': 0, 'y': 1.02},
                        )
                        st.plotly_chart(t3_fig, use_container_width=True,
                                        config={'displayModeBar': False})

    # ── 상관관계 히트맵 ───────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">주요 변수 상관관계</div>', unsafe_allow_html=True)
    heat_cols = ['Distance', 'Duration', 'Velocity_mean', 'Velocity_max',
                 'SOC_Consumed', 'Accel_abs_mean', 'Battery_Power_mean',
                 'Heating_Power_CAN_mean']
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
# PAGE 2 : 트립별 주행거리 예측
# ════════════════════════════════════════════════════════════════
elif page == "트립별 주행거리 예측":
    st.markdown('<div class="bmw-title">트립별 주행거리 예측</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">실제 트립 데이터를 로드하여 AI가 주행거리를 예측합니다 · 사이드바에서 트립을 선택하세요</div>',
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
         f'<br><span style="font-size:.85rem;color:{SUB}">평균속도 · 최고 {feats["Velocity_max"]:.1f} km/h</span>'),
        (i3, '🔋', '배터리',
         f'<span style="font-size:1.0rem;font-weight:600;color:{TXT}">{feats["SOC_Consumed"]*100:.1f}%</span>'
         f'<br><span style="font-size:.85rem;color:{SUB}">SOC 소모 · 전압 {feats["Battery_Voltage_mean"]:.0f} V</span>'),
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
    st.markdown('<div class="bmw-sub">슬라이더로 주행 조건을 설정하면 AI가 주행거리를 실시간 예측합니다</div>',
                unsafe_allow_html=True)
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
            inputs[feat] = tgt.slider(f'{label} ({unit})', lo, hi, default, step)

        with st.expander('고급 변수 (나머지 모델 입력값)'):
            for feat in cols:
                if feat in PRIMARY:
                    continue
                label = feat.replace('_', ' ')
                med = float(medians.get(feat, 0))
                lo2, hi2 = 0.0, max(med * 2.5, med + 1)
                inputs[feat] = st.slider(label, float(lo2), float(hi2), float(med))

    row = {}
    for feat in cols:
        v_val = inputs.get(feat, float(medians.get(feat, 0)))
        if feat == 'SOC_Consumed':
            v_val = v_val / 100.0
        row[feat] = v_val
    X_one = pd.DataFrame([row])[cols]
    pred = float(max(model.predict(X_one)[0], 0.0))

    v_mean    = inputs.get('Velocity_mean', 40)
    dur       = inputs.get('Duration', 20)
    soc_pct   = inputs.get('SOC_Consumed', 10)
    phys_dist = v_mean * dur / 60
    eff       = pred / soc_pct if soc_pct > 0.1 else 0
    diff_pct  = (pred - phys_dist) / phys_dist * 100 if phys_dist > 0 else 0
    arrow     = '▲' if diff_pct >= 0 else '▼'
    arrow_col = GREEN if diff_pct >= 0 else RED

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
                              'value': inputs.get('Velocity_max', v_mean)},
            }))
        gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', font={'color': TXT})
        st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
        st.caption(f'파란 바 = 평균 속도 · 주황 눈금 = 최고 속도 ({inputs.get("Velocity_max",0):.0f} km/h)')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    for col2, val, lab in [
        (k1, f'{phys_dist:,.1f} km', '물리식 거리 (v×t)'),
        (k2, f'{dur:,.0f} min',      '주행 시간'),
        (k3, f'{soc_pct:,.1f} %',    'SOC 소모'),
        (k4, f'{eff:,.2f} km/%',     'SOC 효율'),
    ]:
        col2.markdown(f'<div class="kpi"><div class="v">{val}</div>'
                      f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-head">모델 예측 vs 물리식</div>', unsafe_allow_html=True)
        cmp_fig = go.Figure()
        cmp_fig.add_trace(go.Bar(
            x=['AI 모델 예측', '물리식 (v×t)'], y=[pred, phys_dist],
            marker_color=[BMW_BLUE, SUB],
            text=[f'{pred:.1f} km', f'{phys_dist:.1f} km'],
            textposition='outside', textfont={'color': TXT, 'size': 13}, width=0.45,
        ))
        cmp_fig.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'km',
                   'range': [0, max(pred, phys_dist) * 1.25]},
            xaxis={'color': TXT},
        )
        st.plotly_chart(cmp_fig, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        st.markdown('<div class="sec-head">SOC 소모 민감도</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight">AI 모델이 물리식보다 <strong style="color:{arrow_col}">{arrow} {abs(diff_pct):.1f}%</strong>
        {'높게' if diff_pct >= 0 else '낮게'} 예측했습니다.</div>
        <div class="insight">SOC 효율 <strong>{eff:.2f} km/%</strong> — 배터리 1% 소모당 {eff:.2f} km 주행</div>
        """, unsafe_allow_html=True)

        soc_range = np.linspace(1, 60, 40)
        preds_soc = []
        for sv in soc_range:
            r2_tmp = dict(row); r2_tmp['SOC_Consumed'] = sv / 100.0
            preds_soc.append(float(max(model.predict(pd.DataFrame([r2_tmp])[cols])[0], 0)))
        soc_line = go.Figure()
        soc_line.add_trace(go.Scatter(
            x=soc_range, y=preds_soc, mode='lines',
            line={'color': BMW_BLUE, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.15)',
        ))
        soc_line.add_vline(x=soc_pct, line_color=AMBER, line_dash='dash',
                           annotation_text=f'{soc_pct:.0f}%', annotation_font_color=AMBER)
        soc_line.update_layout(
            height=240, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': 'SOC 소모 (%)'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '예측 km'},
            showlegend=False,
        )
        st.plotly_chart(soc_line, use_container_width=True, config={'displayModeBar': False})
        st.caption('SOC 소모에 따른 예측 주행거리 변화 (현재 조건 고정)')


# ════════════════════════════════════════════════════════════════
# PAGE 4 : 모델 분석 (성능 + 비교 통합)
# ════════════════════════════════════════════════════════════════
elif page == "모델 분석":
    st.markdown('<div class="bmw-title">모델 분석</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">예측 성능 · 잔차 분석 · 베이스라인 / Optuna / GridSearch 비교</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    _df_base_kpi, _df_opt_kpi = load_model_comparison()

    # 최고 R²: df_base > 현재 모델 순으로 fallback
    if not _df_base_kpi.empty and 'R2' in _df_base_kpi.columns:
        best_r2  = float(_df_base_kpi['R2'].max())
    else:
        best_r2  = metrics['R2']
    # 최저 RMSE: df_opt > df_base > 현재 모델 순으로 fallback
    if not _df_opt_kpi.empty and 'RMSE_mean' in _df_opt_kpi.columns:
        best_rmse = float(_df_opt_kpi['RMSE_mean'].min())
    elif not _df_base_kpi.empty and 'RMSE' in _df_base_kpi.columns:
        best_rmse = float(_df_base_kpi['RMSE'].min())
    else:
        best_rmse = metrics['RMSE']

    k1, k2, k3, k4 = st.columns(4)
    for col2, val, lab, vc in [
        (k1, f"{best_r2:.4f}",          "최고 R² (튜닝 후)", GREEN),
        (k2, f"{best_rmse:.2f} km",     "최저 RMSE (튜닝 후)", GREEN),
        (k3, f"{metrics['MAE']:.2f} km", "현재 MAE",          TXT),
        (k4, f"{metrics['n']}건",        "전체 데이터",         TXT),
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

    st.markdown('<div class="sec-head">오차 누적 분포 (CDF)</div>', unsafe_allow_html=True)
    cdf_fig = go.Figure()
    cdf_fig.add_trace(go.Scatter(
        x=abs_err, y=cdf_y * 100, mode='lines',
        line={'color': BMW_BLUE, 'width': 2.5},
        fill='tozeroy', fillcolor='rgba(28,105,212,0.12)',
    ))
    cdf_fig.add_vline(x=p50, line_color=GREEN, line_dash='dash',
                      annotation_text=f'50%: {p50:.1f}km', annotation_font_color=GREEN)
    cdf_fig.add_vline(x=p80, line_color=AMBER, line_dash='dash',
                      annotation_text=f'80%: {p80:.1f}km', annotation_font_color=AMBER)
    cdf_fig.update_layout(
        height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis={'gridcolor': LINE, 'color': SUB, 'title': '절대 오차 (km)'},
        yaxis={'gridcolor': LINE, 'color': SUB, 'title': '누적 비율 (%)'},
        showlegend=False,
    )
    st.plotly_chart(cdf_fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown(f"""
    <div class="insight">잔차 평균 <strong>{residuals.mean():.2f} km</strong> · 표준편차 <strong>{residuals.std():.2f} km</strong> ·
    예측의 <strong>50%</strong>는 ±{p50:.1f} km, <strong>80%</strong>는 ±{p80:.1f} km 이내</div>
    """, unsafe_allow_html=True)

    # ── 베이스라인 & Optuna 비교 ──────────────────────────────
    df_base, df_opt = load_model_comparison()

    if not df_base.empty:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">베이스라인 모델 비교 (AB통합 vs B만)</div>',
                    unsafe_allow_html=True)

        best_base = (df_base.sort_values('R2', ascending=False)
                     .groupby(['Notebook', 'Model'], as_index=False).first())
        notebooks = ['AB통합', 'B만']
        models_list = ['CatBoost', 'XGBoost', 'RandomForest', 'LightGBM']
        nb_colors = {'AB통합': BMW_BLUE, 'B만': GREEN}

        bl_l, bl_r = st.columns(2)
        for out_col, mkey, ylabel, yrange in [
            (bl_l, 'R2',   'R² Score',  [0, 1.12]),
            (bl_r, 'RMSE', 'RMSE (km)', None),
        ]:
            with out_col:
                fig = go.Figure()
                for nb in notebooks:
                    sub = best_base[best_base['Notebook'] == nb]
                    vals = [float(sub[sub['Model'] == m][mkey].values[0])
                            if len(sub[sub['Model'] == m]) > 0 else 0.0
                            for m in models_list]
                    fig.add_trace(go.Bar(
                        name=nb, x=models_list, y=vals,
                        marker_color=nb_colors.get(nb, AMBER),
                        text=[f'{v:.3f}' if v > 0 else '—' for v in vals],
                        textposition='outside', textfont={'color': TXT, 'size': 10},
                    ))
                y_cfg = {'gridcolor': LINE, 'color': SUB, 'title': ylabel}
                if yrange:
                    y_cfg['range'] = yrange
                fig.update_layout(
                    height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=10, r=10, t=10, b=10),
                    barmode='group', xaxis={'color': TXT}, yaxis=y_cfg,
                    legend={'font': {'color': SUB}},
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        best_row = df_base.loc[df_base['R2'].idxmax()]
        st.markdown(f"""
        <div class="insight">최고 성능: <strong>{best_row['Model']}</strong>
        ({best_row['Notebook']}) — R² <strong>{best_row['R2']:.3f}</strong>,
        RMSE <strong>{best_row['RMSE']:.2f} km</strong></div>
        """, unsafe_allow_html=True)

    if not df_opt.empty:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">Optuna · GridSearch 튜닝 성능 비교</div>',
                    unsafe_allow_html=True)
        df_rmse = df_opt[df_opt['RMSE_mean'].notna()].copy()
        df_rmse['Label'] = (df_rmse['Notebook'] + ' | ' + df_rmse['Model'] + ' | '
                            + df_rmse['Method'].str.replace(r'\s*\(.*\)', '', regex=True).str.strip())
        nb_colors_opt = {'AB통합': BMW_BLUE, 'B만': GREEN}
        bar_colors = [nb_colors_opt.get(nb, AMBER) for nb in df_rmse['Notebook']]

        opt_fig = go.Figure()
        opt_fig.add_trace(go.Bar(
            x=df_rmse['Label'], y=df_rmse['RMSE_mean'],
            error_y=dict(type='data', array=df_rmse['RMSE_std'].fillna(0).tolist(),
                         visible=True, color=AMBER),
            marker_color=bar_colors,
            text=[f'{v:.2f}' for v in df_rmse['RMSE_mean']],
            textposition='outside', textfont={'color': TXT, 'size': 10},
        ))
        opt_fig.update_layout(
            height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=100),
            xaxis={'color': TXT, 'tickangle': -35},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'RMSE (km)'},
            showlegend=False,
        )
        st.plotly_chart(opt_fig, use_container_width=True, config={'displayModeBar': False})

        ab_best = (df_rmse[df_rmse['Notebook'] == 'AB통합']['RMSE_mean'].min()
                   if 'AB통합' in df_rmse['Notebook'].values else None)
        b_best  = (df_rmse[df_rmse['Notebook'] == 'B만']['RMSE_mean'].min()
                   if 'B만' in df_rmse['Notebook'].values else None)
        if ab_best and b_best:
            st.markdown(f"""
            <div class="insight">튜닝 후 최고 RMSE — AB통합: <strong>{ab_best:.3f} km</strong>,
            B만: <strong>{b_best:.3f} km</strong> (AB통합이 {b_best-ab_best:.3f} km 낮음)</div>
            """, unsafe_allow_html=True)
        st.dataframe(df_opt.fillna('—'), use_container_width=True)


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
        'Duration':                          '주행 시간이 길수록 더 먼 거리를 이동합니다',
        'SOC_Consumed':                      '배터리 소모가 클수록 더 오랜 주행을 반영합니다',
        'Battery_Temperature_std':           '배터리 온도 변동은 주행 강도를 나타냅니다',
        'Velocity_mean':                     '평균 속도가 높을수록 주행 효율에 영향을 줍니다',
        'Battery_Temperature_diff_max':      '배터리 온도 급변은 과부하 구간을 나타냅니다',
        'Longitudinal_Acceleration_diff_std':'급격한 가감속 변화가 에너지 소모에 영향을 줍니다',
        'Accel_abs_mean':                    '평균 가속도가 높을수록 역동적인 주행입니다',
        'Velocity_diff_std':                 '속도 변화 편차가 크면 불규칙한 주행 패턴입니다',
        'Longitudinal_Acceleration_std':     '종방향 가속 편차가 주행 안정성을 나타냅니다',
        'Velocity_std':                      '속도 변동이 클수록 도심/교외 혼합 주행입니다',
        'Heating_Power_CAN_std':             '난방 출력 변동은 외기온 변화에 따른 에너지 비용입니다',
        'Heating_Power_CAN_mean':            '평균 난방 출력이 높을수록 배터리 소모가 커집니다',
        'Battery_Current_std':               '배터리 전류 편차는 회생제동 빈도를 반영합니다',
        'Battery_Power_mean':                '평균 배터리 출력이 주행 부하를 결정합니다',
        'Accel_abs_std':                     '가속도 편차가 클수록 가감속이 불규칙합니다',
        'Ambient_Temperature_std':           '외기온 변동이 클수록 에너지 관리가 어렵습니다',
        'Velocity_max':                      '최고 속도가 높을수록 순간 전력 소모가 큽니다',
        'Battery_Voltage_mean':              '평균 전압이 낮으면 배터리 방전 수준을 의미합니다',
        'Accel_abs_max':                     '최대 가속도는 급가속 여부를 나타냅니다',
        'Battery_State_of_Charge_End':       '종료 시 충전량이 높을수록 여유 주행이었습니다',
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
