# -*- coding: utf-8 -*-
"""
BMW i3 EV 주행거리 예측 대시보드 (차량 소유자용)
실행: streamlit run app_user.py
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
    'iterations': 1862, 'depth': 3,
    'learning_rate': 0.021653374253691442,
    'l2_leaf_reg': 12.76728103986453,
    'random_strength': 4.854975752752124,
    'bagging_temperature': 4.561826584647859,
}

BMW_BLUE  = '#1C69D4'
BMW_DARK  = '#0A3D91'
BMW_LIGHT = '#5B9BD5'
PANEL     = '#0F2440'
PANEL2    = '#162d50'
LINE      = '#1E3A5F'
TXT       = '#FFFFFF'
SUB       = '#A8C4E0'
GREEN     = '#00C896'
AMBER     = '#FFB000'
RED       = '#E84040'

# ── 사용자 친화 프리셋 ────────────────────────────────────────────

ENV_PRESETS = {
    '도심': {
        'label': '도심 주행', 'desc': '신호·정체 구간',
        'Velocity_mean': 25.0, 'Velocity_max': 65.0,
        'Velocity_std': 13.0,  'Velocity_diff_std': 2.2,
    },
    '혼합': {
        'label': '혼합 주행', 'desc': '도심 + 국도',
        'Velocity_mean': 50.0, 'Velocity_max': 100.0,
        'Velocity_std': 22.0,  'Velocity_diff_std': 1.5,
    },
    '고속도로': {
        'label': '고속도로', 'desc': '자동차전용도로',
        'Velocity_mean': 90.0, 'Velocity_max': 130.0,
        'Velocity_std': 16.0,  'Velocity_diff_std': 0.8,
    },
}

STYLE_PRESETS = {
    '에코': {
        'label': '에코 드라이빙', 'desc': '부드러운 가감속',
        'Accel_abs_mean': 0.30, 'Accel_abs_std': 0.20, 'Accel_abs_max': 1.5,
        'Longitudinal_Acceleration_std': 0.30,
        'Longitudinal_Acceleration_diff_std': 0.10,
    },
    '일반': {
        'label': '일반 주행', 'desc': '평균적인 운전 패턴',
        'Accel_abs_mean': 0.60, 'Accel_abs_std': 0.40, 'Accel_abs_max': 2.5,
        'Longitudinal_Acceleration_std': 0.60,
        'Longitudinal_Acceleration_diff_std': 0.25,
    },
    '스포티': {
        'label': '스포티', 'desc': '빠른 가감속',
        'Accel_abs_mean': 1.00, 'Accel_abs_std': 0.70, 'Accel_abs_max': 3.5,
        'Longitudinal_Acceleration_std': 1.00,
        'Longitudinal_Acceleration_diff_std': 0.50,
    },
}

ENV_KEYS   = list(ENV_PRESETS.keys())
STYLE_KEYS = list(STYLE_PRESETS.keys())


def temp_to_heating(temp_c):
    if temp_c < 0:   return 5.0, 2.0
    if temp_c < 10:  return 3.5, 1.5
    if temp_c < 20:  return 1.5, 0.8
    if temp_c < 28:  return 0.3, 0.2
    return 0.0, 0.1


def build_row(current_soc, target_soc, duration, env_key, style_key, temp_c, medians, cols):
    env   = ENV_PRESETS[env_key]
    style = STYLE_PRESETS[style_key]
    hmean, hstd = temp_to_heating(temp_c)
    soc_consumed = max(0.01, (current_soc - target_soc) / 100.0)
    soc_end      = target_soc / 100.0
    row = {}
    for f in cols:
        if f in env:             row[f] = env[f]
        elif f in style:         row[f] = style[f]
        elif f == 'Duration':                    row[f] = float(duration)
        elif f == 'SOC_Consumed':                row[f] = soc_consumed
        elif f == 'Battery_State_of_Charge_End': row[f] = soc_end
        elif f == 'Heating_Power_CAN_mean':      row[f] = hmean
        elif f == 'Heating_Power_CAN_std':       row[f] = hstd
        elif f == 'Ambient_Temperature_std':     row[f] = 0.5 if temp_c < 20 else 0.2
        else: row[f] = float(medians.get(f, 0))
    return row


def predict_one(row, model, cols):
    return float(max(model.predict(pd.DataFrame([row])[cols])[0], 0.0))


# ── 데이터 / 모델 ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def make_synthetic(n=500, seed=42):
    rng = np.random.default_rng(seed)
    vmean = rng.uniform(18, 95, n)
    dur   = rng.uniform(5, 95, n)
    dist  = vmean * dur / 60 * rng.normal(1, 0.05, n)
    dist  = np.clip(dist, 1.5, None)
    soc   = np.clip(dist / 120 + rng.normal(0, 0.02, n), 0.01, 0.6)
    return pd.DataFrame({
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


@st.cache_data(show_spinner=False)
def load_data():
    for p in ['df_final_vif.csv', 'data/df_final_vif.csv']:
        if os.path.exists(p):
            return pd.read_csv(p), 'real'
    return make_synthetic(), 'synthetic'


@st.cache_resource(show_spinner=False)
def train_model(key, df):
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols]; y = df[TARGET]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    if HAS_CATBOOST:
        m = CatBoostRegressor(**BEST_PARAMS, loss_function='RMSE', random_seed=42, verbose=False)
    else:
        m = GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                      learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    return m, cols, df[cols].median()


@st.cache_data(show_spinner=False)
def load_trip_list():
    if not os.path.exists('data'):
        return []
    return sorted([f for f in os.listdir('data') if f.lower().endswith('.csv')])


def _get_col(df, kws, excl=None):
    excl = [e.lower() for e in (excl or [])]
    for c in df.columns:
        cl = c.lower()
        if any(e in cl for e in excl): continue
        if any(k.lower() in cl for k in kws):
            return pd.to_numeric(df[c], errors='coerce')
    return None


@st.cache_data(show_spinner=False)
def load_trip_raw(fname):
    path = os.path.join('data', fname)
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            d = pd.read_csv(path, sep=';', encoding=enc)
            if d.shape[1] >= 10: return d
        except Exception: continue
    return pd.DataFrame()


def trip_summary(df):
    def s(x, d=0.0): return x if x is not None else pd.Series([d]*len(df))
    t  = s(_get_col(df, ['time [s]', 'time']))
    v  = s(_get_col(df, ['velocity']))
    a  = s(_get_col(df, ['longitudinal acceleration']))
    sc = s(_get_col(df, ['soc [%]'], excl=['max','min','displayed']), 50.0)
    ht = s(_get_col(df, ['heating power can']))
    bt = s(_get_col(df, ['battery temperature'],
                    excl=['max','min','coolant','exchanger','heater','cabin','inlet']), 20.0)
    bc = s(_get_col(df, ['battery current']))
    bv = s(_get_col(df, ['battery voltage']), 370.0)
    at = s(_get_col(df, ['ambient temperature']), 15.0)

    dur   = float((t.max() - t.min()) / 60)
    vmean = float(v.mean()); vmax = float(v.max()); vstd = float(v.std())
    aa    = a.abs()
    soc_d = sc.dropna()
    soc_s = float(soc_d.iloc[0])  if len(soc_d) > 0 else 50.0
    soc_e = float(soc_d.iloc[-1]) if len(soc_d) > 0 else 40.0
    soc_c = max(0.0, (soc_s - soc_e) / 100)
    dt    = t.diff().fillna(0.0)
    actual = float((v * dt / 3600).sum())

    feats = {
        'Duration': dur, 'SOC_Consumed': soc_c,
        'Battery_Temperature_std': float(bt.std()),
        'Velocity_mean': vmean, 'Velocity_max': vmax, 'Velocity_std': vstd,
        'Battery_Temperature_diff_max': float(bt.diff().abs().max()) if len(bt)>1 else 0.0,
        'Longitudinal_Acceleration_diff_std': float(a.diff().std()),
        'Accel_abs_mean': float(aa.mean()), 'Velocity_diff_std': float(v.diff().std()),
        'Longitudinal_Acceleration_std': float(a.std()),
        'Battery_State_of_Charge_End': soc_e / 100,
        'Heating_Power_CAN_std': float(ht.std()), 'Heating_Power_CAN_mean': float(ht.mean()),
        'Accel_abs_std': float(aa.std()), 'Battery_Current_std': float(bc.std()),
        'Battery_Power_mean': float((bv * bc / 1000).mean()),
        'Accel_abs_max': float(aa.max()),
        'Ambient_Temperature_std': float(at.std()),
        'Battery_Voltage_mean': float(bv.mean()),
    }
    return feats, actual, soc_s, soc_e


# ── 페이지 설정 + CSS ──────────────────────────────────────────────
st.set_page_config(page_title='BMW i3 · 내 주행거리', page_icon='🚗',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown(f"""<style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
 .stApp {{
   background:linear-gradient(160deg,#0B1929 0%,#0D1F36 50%,#091525 100%) fixed;
   color:{TXT};font-family:'Inter',sans-serif;
 }}
 #MainMenu,footer,header{{visibility:hidden;}}
 [data-testid="collapsedControl"],button[kind="header"]{{display:none!important;}}
 .block-container{{padding-top:1.5rem;max-width:1440px;}}
 section[data-testid="stSidebar"]{{
   background:linear-gradient(180deg,#0A1E35 0%,#0D2441 100%);
   border-right:1px solid {LINE};
 }}
 section[data-testid="stSidebar"] label,
 section[data-testid="stSidebar"] label p,
 section[data-testid="stSidebar"] label span,
 section[data-testid="stSidebar"] .stRadio label,
 section[data-testid="stSidebar"] .stRadio label p,
 section[data-testid="stSidebar"] div[role="radiogroup"] label{{
   color:{TXT}!important;font-size:.95rem!important;font-weight:500!important;
 }}
 section[data-testid="stSidebar"] .stCaption,
 section[data-testid="stSidebar"] small{{color:{SUB}!important;}}
 .bmw-title{{font-weight:700;font-size:2rem;letter-spacing:.5px;color:{TXT};
   border-left:5px solid {BMW_BLUE};padding-left:16px;line-height:1.2;}}
 .bmw-sub{{color:{SUB};font-size:.9rem;padding-left:21px;margin-top:2px;}}
 .card{{background:{PANEL};border:1px solid {LINE};border-radius:12px;padding:18px 20px;}}
 .kpi{{text-align:center;background:{PANEL};border:1px solid {LINE};
   border-radius:10px;padding:14px 10px;}}
 .kpi .v{{font-size:1.7rem;font-weight:700;color:{TXT};}}
 .kpi .l{{color:{SUB};font-size:.75rem;letter-spacing:1.5px;text-transform:uppercase;margin-top:2px;}}
 .pred-box{{
   background:linear-gradient(135deg,#0d2d52 0%,#0A3D91 100%);
   border:2px solid {BMW_BLUE};border-radius:16px;padding:28px;text-align:center;
 }}
 .pred-num{{font-size:4.2rem;font-weight:700;color:#fff;line-height:1;}}
 .pred-unit{{font-size:1.3rem;color:{SUB};vertical-align:super;}}
 .divider{{height:1px;background:linear-gradient(90deg,
   transparent,{LINE} 20%,{LINE} 80%,transparent);margin:20px 0;}}
 .stSlider label{{color:{TXT}!important;font-weight:500;}}
 .stButton>button{{background:{BMW_BLUE};color:#fff;border:0;border-radius:8px;
   font-weight:600;padding:.55rem 1.4rem;}}
 .stButton>button:hover{{background:{BMW_DARK};}}
 .sec-head{{font-size:1.1rem;font-weight:600;color:{TXT};margin-bottom:12px;
   padding-bottom:8px;border-bottom:1px solid {LINE};}}
 .insight{{
   background:linear-gradient(135deg,#0d2448 0%,{PANEL} 100%);
   border-left:3px solid {BMW_BLUE};border-radius:0 8px 8px 0;
   padding:10px 14px;margin:8px 0;font-size:.88rem;color:{SUB};
 }}
 .insight strong{{color:{TXT};}}
 .tip-card{{
   background:{PANEL2};border:1px solid {LINE};border-radius:12px;
   padding:18px;height:100%;
 }}
</style>""", unsafe_allow_html=True)


# ── 로드 ────────────────────────────────────────────────────────
df, source = load_data()
model, cols, medians = train_model(source + str(len(df)), df)
trip_list = load_trip_list()

BMW_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="32" height="32" style="vertical-align:middle;margin-right:8px">
  <circle cx="50" cy="50" r="49" fill="#1a1a1a"/>
  <circle cx="50" cy="50" r="36" fill="#fff"/>
  <path d="M50 14 A36 36 0 0 1 86 50 L50 50 Z" fill="#1C69D4"/>
  <path d="M50 86 A36 36 0 0 1 14 50 L50 50 Z" fill="#1C69D4"/>
  <circle cx="50" cy="50" r="36" fill="none" stroke="#1a1a1a" stroke-width="2"/>
  <line x1="50" y1="14" x2="50" y2="86" stroke="#1a1a1a" stroke-width="2"/>
  <line x1="14" y1="50" x2="86" y2="50" stroke="#1a1a1a" stroke-width="2"/>
  <circle cx="50" cy="50" r="49" fill="none" stroke="#1a1a1a" stroke-width="2"/>
</svg>"""


# ── 사이드바 ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px 0">
      <div style="font-size:1.7rem;font-weight:700;color:{TXT};display:flex;align-items:center">
        {BMW_LOGO_SVG} BMW i3
      </div>
      <div style="color:{SUB};font-size:.95rem;margin-top:4px">내 차량 주행거리 안내</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "메뉴",
        ["내 차량 현황", "주행거리 예측", "내 주행 기록", "절약 가이드"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    selected_trip = None
    uploaded_trip = None

    if page == "내 주행 기록":
        if trip_list:
            st.markdown(f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
                        f'letter-spacing:.5px;margin-bottom:6px">주행 기록 선택</div>',
                        unsafe_allow_html=True)
            selected_trip = st.selectbox("trip", trip_list,
                                          format_func=lambda x: x.replace('.csv',''),
                                          label_visibility='collapsed')
        st.markdown(f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
                    f'letter-spacing:.5px;margin-top:12px;margin-bottom:6px">직접 업로드</div>',
                    unsafe_allow_html=True)
        uploaded_trip = st.file_uploader('CSV 업로드', type=['csv'],
                                          key='user_trip_up', label_visibility='collapsed')

    st.caption(f'엔진: {"CatBoost" if HAS_CATBOOST else "GradientBoosting"} · 샘플: {len(df)}건')


# ════════════════════════════════════════════════════════════════
# PAGE 1 : 내 차량 현황
# ════════════════════════════════════════════════════════════════
if page == "내 차량 현황":
    st.markdown('<div class="bmw-title">내 BMW i3</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">현재 배터리와 외부 온도를 입력하면 주행 환경별 예상 주행거리를 안내드립니다</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown('<div class="sec-head">현재 상태 입력</div>', unsafe_allow_html=True)
        h_soc  = st.slider('현재 배터리 잔량 (%)', 10, 100,
                            st.session_state.get('h_soc', 80), 5, key='h_soc')
        h_temp = st.slider('외부 온도 (°C)', -10, 40,
                            st.session_state.get('h_temp', 15), 1, key='h_temp')

        soc_color = GREEN if h_soc >= 50 else AMBER if h_soc >= 25 else RED
        soc_status = '충분한 배터리' if h_soc >= 50 else '보통 수준' if h_soc >= 25 else '충전 권장'
        st.markdown(f"""
        <div style="background:{PANEL};border-radius:10px;padding:16px;
                    margin-top:12px;border:1px solid {LINE}">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:8px">
            <span style="color:{SUB};font-size:.85rem">배터리 잔량</span>
            <span style="color:{soc_color};font-weight:700;font-size:1.3rem">{h_soc}%</span>
          </div>
          <div style="background:#1a3050;border-radius:6px;height:16px;overflow:hidden">
            <div style="width:{h_soc}%;background:{soc_color};height:100%;border-radius:6px"></div>
          </div>
          <div style="color:{soc_color};font-size:.78rem;margin-top:8px">{soc_status}</div>
        </div>
        """, unsafe_allow_html=True)

        hmean, _ = temp_to_heating(h_temp)
        temp_status = ('저온 — 난방 출력 높음, 배터리 소모 증가' if h_temp < 10 else
                       '적정 온도 — 최적 효율 구간' if h_temp < 28 else
                       '고온 — 에어컨 사용 시 배터리 추가 소모')
        st.markdown(f"""
        <div style="background:{PANEL};border-radius:10px;padding:14px 16px;
                    margin-top:10px;border:1px solid {LINE}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:{SUB};font-size:.85rem">외부 온도</span>
            <span style="color:{TXT};font-weight:700;font-size:1.2rem">{h_temp}°C</span>
          </div>
          <div style="color:{SUB};font-size:.8rem;margin-top:6px">{temp_status}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_out:
        st.markdown('<div class="sec-head">주행 환경별 예상 주행거리</div>', unsafe_allow_html=True)
        st.caption('배터리를 10%만 남기고 주행할 때의 예상 거리 · 일반 주행 스타일 기준')

        e_cols = st.columns(3)
        env_preds = {}
        for i, ek in enumerate(ENV_KEYS):
            row = build_row(h_soc, 10, 45, ek, '일반', h_temp, medians, cols)
            p   = predict_one(row, model, cols)
            env_preds[ek] = p
            avg_v = ENV_PRESETS[ek]['Velocity_mean']
            e_cols[i].markdown(f"""
            <div style="background:{PANEL};border-radius:12px;padding:20px;
                        text-align:center;border:1px solid {LINE}">
              <div style="color:{SUB};font-size:.78rem;letter-spacing:1px;
                          text-transform:uppercase;margin-bottom:6px">{ENV_PRESETS[ek]['desc']}</div>
              <div style="color:{TXT};font-weight:700;font-size:1.1rem;margin-bottom:14px">{ENV_PRESETS[ek]['label']}</div>
              <div style="font-size:3rem;font-weight:700;color:{TXT};line-height:1">{p:.0f}</div>
              <div style="color:{BMW_LIGHT};font-size:.9rem;margin-top:4px">km</div>
              <div style="color:{SUB};font-size:.75rem;margin-top:10px">평균 {avg_v:.0f} km/h</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        fig_env = go.Figure(go.Bar(
            x=[ENV_PRESETS[k]['label'] for k in ENV_KEYS],
            y=[env_preds[k] for k in ENV_KEYS],
            marker_color=[BMW_BLUE, BMW_LIGHT, GREEN],
            text=[f'{env_preds[k]:.0f} km' for k in ENV_KEYS],
            textposition='outside', textfont={'color': TXT, 'size': 13},
            width=0.45,
        ))
        fig_env.update_layout(
            height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': 'km', 'font': {'size': 10, 'color': SUB}},
                   'range': [0, max(env_preds.values()) * 1.3]},
            xaxis={'color': TXT},
            showlegend=False,
        )
        st.plotly_chart(fig_env, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tips = []
    if h_temp < 5:
        tips.append('저온 주의 — 배터리 성능이 저하됩니다. 출발 전 실내 예열보다 <strong>시트 히터</strong>를 우선 사용하면 주행거리를 아낄 수 있습니다.')
    if h_soc < 25:
        tips.append(f'배터리 잔량 <strong>{h_soc}%</strong> — 충전을 권장합니다. 회생제동을 적극 활용해 주세요.')
    if h_temp > 28:
        tips.append('고온 날씨 — 에어컨 사용 시 주행거리가 줄어듭니다. 창문 환기로 사전 냉각 후 출발하세요.')
    if not tips:
        tips.append('현재 조건은 양호합니다. <strong>에코 드라이빙 모드</strong>로 주행거리를 최대화하세요.')

    t_cols = st.columns(len(tips))
    for i, tip in enumerate(tips):
        t_cols[i].markdown(f'<div class="insight">{tip}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 : 주행거리 예측
# ════════════════════════════════════════════════════════════════
elif page == "주행거리 예측":
    st.markdown('<div class="bmw-title">주행거리 예측</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">내 주행 조건을 설정하면 AI가 예상 주행거리를 알려드립니다</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="sec-head">주행 조건 설정</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        p_soc  = c1.slider('현재 배터리 (%)', 10, 100, 80, 5, key='p_soc')
        p_dur  = c2.slider('예상 주행 시간 (분)', 5, 120, 40, 5, key='p_dur')
        p_temp = c1.slider('외부 온도 (°C)', -10, 40, 15, 1, key='p_temp')
        p_tsoc = c2.slider('도착 후 목표 배터리 (%)', 5, 50, 10, 5, key='p_tsoc')

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
                    f'letter-spacing:.5px;margin-bottom:6px">주행 환경</div>', unsafe_allow_html=True)
        p_env = st.radio('env', ENV_KEYS, index=1,
                          format_func=lambda k: f'{k}  —  {ENV_PRESETS[k]["desc"]}',
                          label_visibility='collapsed', key='p_env')

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:{SUB};font-size:.82rem;font-weight:600;'
                    f'letter-spacing:.5px;margin-bottom:6px">주행 스타일</div>', unsafe_allow_html=True)
        p_style = st.radio('style', STYLE_KEYS, index=1,
                            format_func=lambda k: f'{k}  —  {STYLE_PRESETS[k]["desc"]}',
                            label_visibility='collapsed', key='p_style')

    # 예측 계산
    target_soc = min(p_tsoc, p_soc - 5)
    row_pred   = build_row(p_soc, target_soc, p_dur, p_env, p_style, p_temp, medians, cols)
    pred_dist  = predict_one(row_pred, model, cols)
    soc_used   = p_soc - target_soc
    eff        = pred_dist / soc_used if soc_used > 0 else 0
    avg_v      = ENV_PRESETS[p_env]['Velocity_mean']
    vmax_v     = ENV_PRESETS[p_env]['Velocity_max']

    with right:
        # 예측 결과 박스
        st.markdown(
            f'<div class="pred-box">'
            f'<div style="color:{BMW_LIGHT};font-size:.8rem;letter-spacing:2px;'
            f'text-transform:uppercase;margin-bottom:8px">예상 주행거리</div>'
            f'<div class="pred-num">{pred_dist:.1f}<span class="pred-unit"> km</span></div>'
            f'<div style="color:{SUB};font-size:.85rem;margin-top:10px">'
            f'{ENV_PRESETS[p_env]["label"]} · {STYLE_PRESETS[p_style]["label"]} · {p_temp}°C'
            f'</div></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        # 속도계
        gauge = go.Figure(go.Indicator(
            mode='gauge+number', value=avg_v,
            number={'suffix': ' km/h', 'font': {'size': 28, 'color': TXT, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 140], 'tickcolor': SUB,
                         'tickfont': {'color': SUB, 'size': 10}},
                'bar': {'color': BMW_BLUE, 'thickness': 0.3},
                'bgcolor': PANEL2, 'borderwidth': 0,
                'steps': [
                    {'range': [0,  40], 'color': '#0d2040'},
                    {'range': [40, 80], 'color': '#102a50'},
                    {'range': [80, 140], 'color': '#0f3060'},
                ],
                'threshold': {
                    'line': {'color': AMBER, 'width': 3},
                    'thickness': 0.8, 'value': vmax_v,
                },
            },
        ))
        gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', font={'color': TXT},
        )
        st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
        st.caption(f'파란 바 = 평균 속도 ({avg_v:.0f} km/h) · 주황 눈금 = 최고 속도 ({vmax_v:.0f} km/h)')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # KPI 행
    k1, k2, k3, k4 = st.columns(4)
    for col_k, val, lab in [
        (k1, f'{soc_used}%',       '배터리 사용량'),
        (k2, f'{eff:.2f} km',      'SOC 1%당 거리'),
        (k3, f'{p_dur}분',         '예상 주행 시간'),
        (k4, f'{avg_v:.0f} km/h',  '평균 속도'),
    ]:
        col_k.markdown(f'<div class="kpi"><div class="v" style="font-size:1.4rem">{val}</div>'
                       f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    ch_l, ch_r = st.columns(2)

    with ch_l:
        st.markdown('<div class="sec-head">주행 환경별 비교</div>', unsafe_allow_html=True)
        env_cmp = {}
        for ek in ENV_KEYS:
            r = build_row(p_soc, target_soc, p_dur, ek, p_style, p_temp, medians, cols)
            env_cmp[ek] = predict_one(r, model, cols)

        cmp_colors = [BMW_BLUE if k == p_env else '#2a4a70' for k in ENV_KEYS]
        cmp_fig = go.Figure(go.Bar(
            x=[ENV_PRESETS[k]['label'] for k in ENV_KEYS],
            y=list(env_cmp.values()),
            marker_color=cmp_colors,
            text=[f'{v:.0f} km' for v in env_cmp.values()],
            textposition='outside', textfont={'color': TXT, 'size': 12},
            width=0.5,
        ))
        cmp_fig.update_layout(
            height=260, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=24, b=10),
            yaxis={'gridcolor': LINE, 'color': SUB,
                   'range': [0, max(env_cmp.values()) * 1.3],
                   'title': {'text': 'km', 'font': {'size': 10, 'color': SUB}}},
            xaxis={'color': TXT}, showlegend=False,
        )
        st.plotly_chart(cmp_fig, use_container_width=True, config={'displayModeBar': False})

    with ch_r:
        st.markdown('<div class="sec-head">배터리 잔량별 예상 주행거리</div>', unsafe_allow_html=True)
        soc_range = np.arange(15, 105, 5)
        soc_preds = []
        for s in soc_range:
            tgt = max(10, int(s) - 60)
            r   = build_row(int(s), tgt, p_dur, p_env, p_style, p_temp, medians, cols)
            soc_preds.append(predict_one(r, model, cols))

        soc_fig = go.Figure()
        soc_fig.add_trace(go.Scatter(
            x=soc_range, y=soc_preds, mode='lines',
            line={'color': BMW_BLUE, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.15)',
            showlegend=False,
        ))
        soc_fig.add_vline(x=p_soc, line_color=AMBER, line_dash='dash', line_width=2,
                          annotation_text=f'현재 {p_soc}%  →  {pred_dist:.0f} km',
                          annotation_font_color=AMBER, annotation_font_size=11)
        soc_fig.update_layout(
            height=260, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=24, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '현재 배터리 잔량 (%)', 'font': {'size': 11, 'color': SUB}}},
            yaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '예상 주행거리 (km)', 'font': {'size': 11, 'color': SUB}}},
            showlegend=False,
        )
        st.plotly_chart(soc_fig, use_container_width=True, config={'displayModeBar': False})

    hmean, _ = temp_to_heating(p_temp)
    heat_msg = f'  난방 출력이 높아 배터리 소모가 증가합니다 ({p_temp}°C).' if hmean > 3 else ''
    st.markdown(f"""
    <div class="insight">
      <strong>{p_soc}%</strong> 배터리로 <strong>{ENV_PRESETS[p_env]["label"]}</strong>
      {p_dur}분 주행 시 약 <strong style="color:{BMW_LIGHT}">{pred_dist:.1f} km</strong> 예상.
      배터리 1%당 <strong>{eff:.2f} km</strong>.{heat_msg}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 : 내 주행 기록
# ════════════════════════════════════════════════════════════════
elif page == "내 주행 기록":
    st.markdown('<div class="bmw-title">내 주행 기록</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">주행 파일을 불러오면 AI가 이번 주행의 효율과 결과를 분석해 드립니다</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    trip_df = pd.DataFrame()
    trip_label = ''

    if uploaded_trip is not None:
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                uploaded_trip.seek(0)
                _d = pd.read_csv(uploaded_trip, sep=';', encoding=enc)
                if _d.shape[1] >= 5:
                    trip_df = _d; trip_label = uploaded_trip.name.replace('.csv', '')
                    break
            except Exception:
                continue
    elif selected_trip:
        trip_df = load_trip_raw(selected_trip)
        trip_label = selected_trip.replace('.csv', '')

    if trip_df.empty:
        st.info('사이드바에서 주행 기록을 선택하거나 CSV 파일을 업로드하세요.')
        st.stop()

    feats, actual_dist, soc_s, soc_e = trip_summary(trip_df)
    X_trip  = pd.DataFrame([feats])[cols]
    pred_d  = float(max(model.predict(X_trip)[0], 0.0))
    err_pct = abs(pred_d - actual_dist) / actual_dist * 100 if actual_dist > 1 else 0
    err_col = GREEN if err_pct < 10 else AMBER if err_pct < 20 else RED

    avg_dist   = float(df['Distance'].mean()) if 'Distance' in df.columns else 20
    eff_ratio  = actual_dist / avg_dist if avg_dist > 0 else 1
    grade      = '우수' if eff_ratio > 1.2 else '양호' if eff_ratio > 0.9 else '보통'
    grade_col  = GREEN if eff_ratio > 1.2 else BMW_BLUE if eff_ratio > 0.9 else AMBER

    k1, k2, k3, k4, k5 = st.columns(5)
    for col_k, val, lab in [
        (k1, f'{actual_dist:.1f} km', '실제 주행거리'),
        (k2, f'{feats["Duration"]:.0f} 분', '주행 시간'),
        (k3, f'{feats["Velocity_mean"]:.1f} km/h', '평균 속도'),
        (k4, f'{feats["SOC_Consumed"]*100:.1f} %', '배터리 소모'),
        (k5, f'{soc_s:.0f}% → {soc_e:.0f}%', '충전 상태'),
    ]:
        col_k.markdown(f'<div class="kpi"><div class="v" style="font-size:1.3rem">{val}</div>'
                       f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    top_l, top_r = st.columns([1, 1.5])

    with top_l:
        st.markdown(
            f'<div class="pred-box">'
            f'<div style="color:{BMW_LIGHT};font-size:.8rem;letter-spacing:2px;'
            f'text-transform:uppercase;margin-bottom:8px">이번 주행</div>'
            f'<div class="pred-num">{actual_dist:.1f}<span class="pred-unit"> km</span></div>'
            f'<div style="color:{SUB};font-size:.85rem;margin-top:8px">{trip_label}</div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{PANEL};border-radius:10px;padding:16px;border:1px solid {LINE}">
          <div style="color:{SUB};font-size:.8rem;margin-bottom:6px">AI 예측과의 차이</div>
          <div style="color:{err_col};font-size:1.6rem;font-weight:700">±{err_pct:.1f}%</div>
          <div style="color:{SUB};font-size:.8rem;margin-top:4px">
            AI 예측 {pred_d:.1f} km · 실제 {actual_dist:.1f} km
          </div>
          <div style="color:{grade_col};font-size:.9rem;margin-top:8px;font-weight:600">
            효율 평가: {grade}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with top_r:
        st.markdown('<div class="sec-head">AI 예측 vs 실제 주행거리</div>', unsafe_allow_html=True)
        bar_fig = go.Figure(go.Bar(
            x=['AI 예측', '실제 주행'],
            y=[pred_d, actual_dist],
            marker_color=[BMW_BLUE, GREEN],
            text=[f'{pred_d:.1f} km', f'{actual_dist:.1f} km'],
            textposition='outside', textfont={'color': TXT, 'size': 14},
            width=0.4,
        ))
        bar_fig.update_layout(
            height=260, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=24, b=10),
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'km',
                   'range': [0, max(pred_d, actual_dist) * 1.35]},
            xaxis={'color': TXT}, showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False})

        if 'Distance' in df.columns:
            st.markdown('<div class="sec-head" style="margin-top:8px">전체 기록 대비 위치</div>',
                        unsafe_allow_html=True)
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(
                x=df['Distance'], nbinsx=25,
                marker_color=BMW_BLUE, opacity=0.7, showlegend=False,
            ))
            hist_fig.add_vline(x=actual_dist, line_color=GREEN, line_width=2.5,
                               annotation_text=f'이번 주행 {actual_dist:.0f} km',
                               annotation_font_color=GREEN, annotation_font_size=10)
            hist_fig.add_vline(x=float(df['Distance'].mean()), line_color=AMBER,
                               line_dash='dash', line_width=1.5,
                               annotation_text=f'평균 {df["Distance"].mean():.0f} km',
                               annotation_font_color=AMBER, annotation_font_size=10)
            hist_fig.update_layout(
                height=200, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=24, b=30),
                xaxis={'gridcolor': LINE, 'color': SUB,
                       'title': {'text': '주행거리 (km)', 'font': {'size': 10, 'color': SUB}}},
                yaxis={'gridcolor': LINE, 'color': SUB,
                       'title': {'text': '건수', 'font': {'size': 10, 'color': SUB}}},
            )
            st.plotly_chart(hist_fig, use_container_width=True, config={'displayModeBar': False})

    t_col = next((c for c in trip_df.columns if 'time' in c.lower()), None)
    v_col = next((c for c in trip_df.columns if 'velocity' in c.lower()), None)
    sc_col = next((c for c in trip_df.columns
                   if 'soc [%]' in c.lower()
                   and 'max' not in c.lower() and 'min' not in c.lower()), None)

    if t_col and v_col:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">속도 및 배터리 변화</div>', unsafe_allow_html=True)
        t_ser = pd.to_numeric(trip_df[t_col], errors='coerce') / 60
        v_ser = pd.to_numeric(trip_df[v_col], errors='coerce')

        ts_rows = 2 if sc_col else 1
        ts_fig  = make_subplots(rows=ts_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06)
        ts_fig.add_trace(go.Scatter(
            x=t_ser, y=v_ser, mode='lines', name='속도',
            line={'color': BMW_BLUE, 'width': 1.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.12)',
        ), row=1, col=1)
        ts_fig.update_yaxes(title_text='속도 (km/h)', title_font={'size': 9, 'color': SUB},
                            gridcolor=LINE, color=SUB, row=1, col=1)

        if sc_col:
            sc_ser = pd.to_numeric(trip_df[sc_col], errors='coerce')
            ts_fig.add_trace(go.Scatter(
                x=t_ser, y=sc_ser, mode='lines', name='배터리 (%)',
                line={'color': GREEN, 'width': 1.5},
                fill='tozeroy', fillcolor='rgba(0,200,150,0.10)',
            ), row=2, col=1)
            ts_fig.update_yaxes(title_text='배터리 (%)', title_font={'size': 9, 'color': SUB},
                                gridcolor=LINE, color=SUB, row=2, col=1)

        ts_fig.update_xaxes(gridcolor=LINE, color=SUB, tickfont={'size': 9},
                            title_text='시간 (분)', title_font={'size': 10, 'color': SUB},
                            row=ts_rows, col=1)
        ts_fig.update_layout(
            height=80 + ts_rows * 130,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=20, t=10, b=40),
            legend={'font': {'color': SUB, 'size': 10}, 'bgcolor': 'rgba(0,0,0,0)',
                    'orientation': 'h', 'x': 0, 'y': 1.05},
        )
        st.plotly_chart(ts_fig, use_container_width=True, config={'displayModeBar': False})


# ════════════════════════════════════════════════════════════════
# PAGE 4 : 절약 가이드
# ════════════════════════════════════════════════════════════════
elif page == "절약 가이드":
    st.markdown('<div class="bmw-title">절약 가이드</div>', unsafe_allow_html=True)
    st.markdown('<div class="bmw-sub">주행거리에 영향을 주는 요소와 절약 팁을 확인하세요</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown('<div class="sec-head">속도와 주행거리의 관계</div>', unsafe_allow_html=True)
        speeds = np.arange(20, 135, 5)
        speed_preds = []
        for spd in speeds:
            r = {f: float(medians.get(f, 0)) for f in cols}
            r.update({'Velocity_mean': float(spd), 'Velocity_max': float(min(spd * 1.5, 150)),
                      'Velocity_std': float(spd * 0.3), 'Duration': 40.0,
                      'SOC_Consumed': 0.30, 'Battery_State_of_Charge_End': 0.30})
            speed_preds.append(predict_one(r, model, cols))

        sp_fig = go.Figure()
        sp_fig.add_trace(go.Scatter(
            x=speeds, y=speed_preds, mode='lines',
            line={'color': BMW_BLUE, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.15)',
            showlegend=False,
        ))
        for spd, lbl, clr in [(25, '도심', GREEN), (50, '혼합', AMBER), (90, '고속', RED)]:
            sp_fig.add_vline(x=spd, line_color=clr, line_dash='dot', line_width=1.5,
                             annotation_text=lbl,
                             annotation_font_color=clr, annotation_font_size=10)
        sp_fig.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '평균 속도 (km/h)', 'font': {'size': 11, 'color': SUB}}},
            yaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '예상 주행거리 (km)', 'font': {'size': 11, 'color': SUB}}},
        )
        st.plotly_chart(sp_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f'<div class="insight"><strong>50~70 km/h</strong> 구간을 유지하면 가장 효율적입니다. 저속 도심 주행과 고속 주행은 서로 다른 이유로 효율이 낮아질 수 있습니다.</div>', unsafe_allow_html=True)

    with col_g2:
        st.markdown('<div class="sec-head">온도와 주행거리의 관계</div>', unsafe_allow_html=True)
        temps = np.arange(-10, 41, 2)
        temp_preds = []
        for tc in temps:
            hm, hs = temp_to_heating(tc)
            r = {f: float(medians.get(f, 0)) for f in cols}
            r.update({'Duration': 40.0, 'SOC_Consumed': 0.30, 'Battery_State_of_Charge_End': 0.30,
                      'Velocity_mean': 50.0, 'Velocity_max': 100.0, 'Velocity_std': 22.0,
                      'Heating_Power_CAN_mean': hm, 'Heating_Power_CAN_std': hs,
                      'Ambient_Temperature_std': 0.5 if tc < 20 else 0.2})
            temp_preds.append(predict_one(r, model, cols))

        t_fig = go.Figure()
        t_fig.add_trace(go.Scatter(
            x=temps, y=temp_preds, mode='lines',
            line={'color': GREEN, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(0,200,150,0.12)',
            showlegend=False,
        ))
        t_fig.add_vrect(x0=-10, x1=5, fillcolor='rgba(232,64,64,0.08)', line_width=0,
                        annotation_text='저온', annotation_font_color=RED, annotation_font_size=10)
        t_fig.add_vrect(x0=15, x1=25, fillcolor='rgba(0,200,150,0.08)', line_width=0,
                        annotation_text='최적', annotation_font_color=GREEN, annotation_font_size=10)
        t_fig.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=40),
            xaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '외부 온도 (°C)', 'font': {'size': 11, 'color': SUB}}},
            yaxis={'gridcolor': LINE, 'color': SUB,
                   'title': {'text': '예상 주행거리 (km)', 'font': {'size': 11, 'color': SUB}}},
        )
        st.plotly_chart(t_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f'<div class="insight"><strong>15~25°C</strong>가 배터리 효율 최적 구간입니다. 영하 날씨에는 난방으로 인해 주행거리가 줄어듭니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-head">주행 스타일별 주행거리 비교</div>', unsafe_allow_html=True)
    style_results = {}
    for sk in STYLE_KEYS:
        r = build_row(80, 10, 40, '혼합', sk, 15, medians, cols)
        style_results[sk] = predict_one(r, model, cols)

    s_fig = go.Figure(go.Bar(
        x=[STYLE_PRESETS[k]['label'] for k in STYLE_KEYS],
        y=list(style_results.values()),
        marker_color=[GREEN, BMW_BLUE, AMBER],
        text=[f'{v:.0f} km' for v in style_results.values()],
        textposition='outside', textfont={'color': TXT, 'size': 13},
        width=0.4,
    ))
    s_fig.update_layout(
        height=220, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'km',
               'range': [0, max(style_results.values()) * 1.3]},
        xaxis={'color': TXT}, showlegend=False,
    )
    st.plotly_chart(s_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-head">주행거리를 늘리는 실용 팁</div>', unsafe_allow_html=True)

    tips_data = [
        ('에코 모드 활용',
         '에코 드라이빙 모드를 켜면 가속 응답을 부드럽게 조절해 에너지 소모를 줄입니다.',
         GREEN),
        ('회생제동 극대화',
         '브레이크 밟기 전 미리 가속 페달을 놓으세요. 감속 에너지가 배터리로 회수됩니다.',
         BMW_BLUE),
        ('출발 전 예열 · 냉각',
         '추운 날에는 충전 중 히터를 미리 켜세요. 주행 중 난방 대신 충전 전력을 사용하면 주행거리가 늘어납니다.',
         AMBER),
        ('일정한 속도 유지',
         '급가속·급감속을 피하고 앞차와의 간격을 넉넉히 유지하면 에너지 낭비를 줄일 수 있습니다.',
         RED),
    ]

    tip_cols = st.columns(4)
    for i, (title, desc, clr) in enumerate(tips_data):
        tip_cols[i].markdown(f"""
        <div class="tip-card" style="border-top:3px solid {clr}">
          <div style="color:{clr};font-weight:700;font-size:.95rem;margin-bottom:10px">{title}</div>
          <div style="color:{SUB};font-size:.82rem;line-height:1.55">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    st.caption('BMW i3 Battery & Heating Data in Real Driving Cycles (FTM, TU München)')
