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
    'Velocity_mean':            ('평균 속도',       'km/h', 0.0, 110.0, 1.0),
    'Velocity_max':             ('최고 속도',       'km/h', 0.0, 160.0, 1.0),
    'Duration':                 ('주행 시간',       'min',  1.0, 100.0, 0.5),
    'SOC_Consumed':             ('배터리 소모량',   '%',    0.0, 60.0,  0.5),
    'Accel_abs_mean':           ('평균 가속도',     'm/s²', 0.0, 2.0,   0.05),
    'Heating_Power_CAN_mean':   ('평균 난방 출력',  'kW',   0.0, 6.0,   0.1),
    'Battery_Power_mean':       ('평균 배터리 출력','kW',   0.0, 30.0,  0.5),
    'Battery_Temperature_std':  ('배터리 온도 변동','°C',   0.0, 6.0,   0.1),
    'Ambient_Temperature_std':  ('외기온 변동',     '°C',   0.0, 4.0,   0.1),
    'Velocity_std':             ('속도 표준편차',   'km/h', 0.0, 50.0,  0.5),
}
PRIMARY = ['Velocity_mean', 'Velocity_max', 'Duration', 'SOC_Consumed',
           'Accel_abs_mean', 'Heating_Power_CAN_mean', 'Battery_Power_mean',
           'Battery_Temperature_std']

# ── BMW 컬러 팔레트 ──────────────────────────────────────────────
BMW_BLUE    = '#1C69D4'
BMW_DARK    = '#0A3D91'
BMW_LIGHT   = '#5B9BD5'
DARK        = '#0B1929'
PANEL       = '#0F2440'
PANEL2      = '#162d50'
LINE        = '#1E3A5F'
TXT         = '#FFFFFF'
SUB         = '#A8C4E0'
GREEN       = '#00C896'
AMBER       = '#FFB000'
RED         = '#E84040'


# ── 데이터 ──────────────────────────────────────────────────────
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


# ── 페이지 설정 + BMW CSS ────────────────────────────────────────
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

 /* 사이드바 */
 [data-testid=stSidebar] {{
   background: linear-gradient(180deg, #0A1E35 0%, #0D2441 100%);
   border-right: 1px solid {LINE};
 }}
 [data-testid=stSidebar] .stRadio label {{
   color:{TXT} !important; font-size:.95rem; font-weight:500;
 }}

 /* 타이틀 */
 .bmw-title {{
   font-weight:700; font-size:2rem; letter-spacing:.5px; color:{TXT};
   border-left:5px solid {BMW_BLUE}; padding-left:16px; line-height:1.2;
 }}
 .bmw-sub {{
   color:{SUB}; font-size:.9rem; padding-left:21px; margin-top:2px;
 }}

 /* 카드 */
 .card {{
   background:{PANEL}; border:1px solid {LINE}; border-radius:12px;
   padding:18px 20px;
 }}
 .card-blue {{
   background: linear-gradient(135deg, {PANEL} 0%, #0d2d52 100%);
   border:1px solid {BMW_BLUE};
 }}

 /* KPI */
 .kpi {{ text-align:center; background:{PANEL}; border:1px solid {LINE};
   border-radius:10px; padding:14px 10px; }}
 .kpi .v {{ font-size:1.7rem; font-weight:700; color:{TXT}; }}
 .kpi .l {{ color:{SUB}; font-size:.75rem; letter-spacing:1.5px;
   text-transform:uppercase; margin-top:2px; }}

 /* 예측 결과 */
 .pred-box {{
   background: linear-gradient(135deg, #0d2d52 0%, #0A3D91 100%);
   border:2px solid {BMW_BLUE}; border-radius:16px;
   padding:28px; text-align:center;
 }}
 .pred-num {{
   font-size:4.2rem; font-weight:700; color:#fff; line-height:1;
 }}
 .pred-unit {{ font-size:1.3rem; color:{SUB}; vertical-align:super; }}
 .pred-label {{
   color:{BMW_LIGHT}; font-size:.85rem; letter-spacing:2px;
   text-transform:uppercase; margin-top:8px;
 }}

 /* 구분선 */
 .divider {{ height:1px; background: linear-gradient(90deg,
   transparent, {LINE} 20%, {LINE} 80%, transparent);
   margin:20px 0; }}

 /* 슬라이더 */
 .stSlider label {{ color:{TXT} !important; font-weight:500; }}
 .stSlider [data-baseweb=slider] div[role=slider] {{ background:{BMW_BLUE}; }}

 /* 버튼 */
 .stButton>button {{
   background:{BMW_BLUE}; color:#fff; border:0; border-radius:8px;
   font-weight:600; padding:.55rem 1.4rem;
 }}
 .stButton>button:hover {{ background:{BMW_DARK}; }}

 /* 뱃지 */
 .badge {{ display:inline-block; padding:3px 10px; border-radius:20px;
   font-size:.75rem; font-weight:600; letter-spacing:.5px; }}

 /* 섹션 헤더 */
 .sec-head {{
   font-size:1.1rem; font-weight:600; color:{TXT}; margin-bottom:12px;
   padding-bottom:8px; border-bottom:1px solid {LINE};
 }}
 .sec-head span {{ color:{BMW_BLUE}; margin-right:6px; }}

 /* insight 박스 */
 .insight {{
   background: linear-gradient(135deg, #0d2448 0%, {PANEL} 100%);
   border-left:3px solid {BMW_BLUE}; border-radius:0 8px 8px 0;
   padding:10px 14px; margin:8px 0; font-size:.88rem; color:{SUB};
 }}
 .insight strong {{ color:{TXT}; }}
</style>
""", unsafe_allow_html=True)


# ── 데이터 & 모델 로딩 ───────────────────────────────────────────
df, source = load_data()
model, metrics, importance, cols, medians = train_model(source + str(len(df)), df)


# ── 사이드바 네비게이션 ──────────────────────────────────────────
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

with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px 0">
      <div style="font-size:1.3rem;font-weight:700;color:{TXT};display:flex;align-items:center">
        {BMW_LOGO_SVG} BMW i3
      </div>
      <div style="color:{SUB};font-size:.8rem;margin-top:2px">EV 주행거리 예측 대시보드</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "페이지 선택",
        ["🏠  개요 · 홈", "🔮  주행거리 예측", "📊  모델 성능 분석", "📈  데이터 인사이트", "🧠  변수 중요도"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 데이터 상태 뱃지
    if source == 'real':
        badge_html = f'<span class="badge" style="background:{GREEN};color:#000">● 실데이터</span>'
    else:
        badge_html = f'<span class="badge" style="background:{AMBER};color:#000">● 데모(합성) 데이터</span>'
    st.markdown(badge_html, unsafe_allow_html=True)

    eng = 'CatBoost' if HAS_CATBOOST else 'GradientBoosting'
    st.caption(f'엔진: {eng}  ·  샘플: {len(df)}건')


# ════════════════════════════════════════════════════════════════
# PAGE 1 : 개요 · 홈
# ════════════════════════════════════════════════════════════════
if page == "🏠  개요 · 홈":
    st.markdown('<div class="bmw-title">BMW i3 · EV 주행거리 예측</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bmw-sub">BMW i3 실주행 데이터 · 주행거리(Distance) 예측</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 상단 KPI 4개
    k1, k2, k3, k4 = st.columns(4)
    r2c = GREEN if metrics['R2'] >= 0.8 else AMBER
    for col, val, lab in [
        (k1, f"{metrics['R2']:.3f}", "모델 R²"),
        (k2, f"{metrics['MAE']:.1f} km", "평균 오차 (MAE)"),
        (k3, f"{metrics['RMSE']:.1f} km", "RMSE"),
        (k4, f"{metrics['n']}건", "학습 데이터"),
    ]:
        col.markdown(
            f'<div class="kpi"><div class="v">{val}</div><div class="l">{lab}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # 빠른 데이터 요약
    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.markdown('<div class="sec-head"><span>📋</span>데이터 분포 요약</div>',
                    unsafe_allow_html=True)
        sum_cols = ['Distance', 'Duration', 'Velocity_mean', 'SOC_Consumed']
        sum_cols = [c for c in sum_cols if c in df.columns]
        desc = df[sum_cols].describe().loc[['mean', 'std', 'min', '50%', 'max']].round(2)
        desc.index = ['평균', '표준편차', '최솟값', '중앙값', '최댓값']
        rename_map = {
            'Distance': '주행거리(km)', 'Duration': '주행시간(min)',
            'Velocity_mean': '평균속도(km/h)', 'SOC_Consumed': 'SOC 소모',
        }
        desc.rename(columns=rename_map, inplace=True)
        st.dataframe(desc, use_container_width=True)

        # 인사이트 박스
        dist_mean = df['Distance'].mean() if 'Distance' in df.columns else 0
        vel_mean = df['Velocity_mean'].mean() if 'Velocity_mean' in df.columns else 0
        st.markdown(f"""
        <div class="insight"><strong>평균 주행거리</strong>는 {dist_mean:.1f} km,
        평균 주행속도는 {vel_mean:.1f} km/h입니다.</div>
        <div class="insight">모델 R² <strong>{metrics['R2']:.3f}</strong> —
        실제 주행거리 변동의 {metrics['R2']*100:.0f}%를 설명합니다.</div>
        <div class="insight">MAE <strong>{metrics['MAE']:.1f} km</strong> —
        예측이 평균적으로 이 범위 안에서 맞습니다.</div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-head"><span>📊</span>주행거리 분포</div>',
                    unsafe_allow_html=True)
        if 'Distance' in df.columns:
            hist = go.Figure()
            hist.add_trace(go.Histogram(
                x=df['Distance'], nbinsx=35,
                marker_color=BMW_BLUE, opacity=0.85,
                name='주행거리',
            ))
            hist.add_vline(x=df['Distance'].mean(), line_color=AMBER,
                           line_width=2, annotation_text=f"평균 {df['Distance'].mean():.1f}km",
                           annotation_font_color=AMBER)
            hist.update_layout(
                height=310, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis={'gridcolor': LINE, 'color': SUB, 'title': '주행거리 (km)'},
                yaxis={'gridcolor': LINE, 'color': SUB, 'title': '빈도'},
                showlegend=False,
            )
            st.plotly_chart(hist, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 상관관계 미니 히트맵
    st.markdown('<div class="sec-head"><span>🔗</span>주요 변수 상관관계</div>',
                unsafe_allow_html=True)
    heat_cols = ['Distance', 'Duration', 'Velocity_mean', 'Velocity_max',
                 'SOC_Consumed', 'Accel_abs_mean', 'Battery_Power_mean',
                 'Heating_Power_CAN_mean']
    heat_cols = [c for c in heat_cols if c in df.columns]
    corr = df[heat_cols].corr().round(2)
    labels = [META[c][0] if c in META else c.replace('_', ' ') for c in corr.columns]

    heat = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
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


# ════════════════════════════════════════════════════════════════
# PAGE 2 : 주행거리 예측
# ════════════════════════════════════════════════════════════════
elif page == "🔮  주행거리 예측":
    st.markdown('<div class="bmw-title">🔮 주행거리 예측</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bmw-sub">슬라이더로 주행 조건을 설정하면 AI가 주행거리를 실시간 예측합니다</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="sec-head"><span>🎛️</span>주행 조건 설정</div>',
                    unsafe_allow_html=True)
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

        with st.expander('🔧 고급 변수 (나머지 모델 입력값)'):
            for feat in cols:
                if feat in PRIMARY:
                    continue
                label = feat.replace('_', ' ')
                med = float(medians.get(feat, 0))
                lo2, hi2 = 0.0, max(med * 2.5, med + 1)
                inputs[feat] = st.slider(label, float(lo2), float(hi2), float(med))

    # 예측 계산
    row = {}
    for feat in cols:
        v = inputs.get(feat, float(medians.get(feat, 0)))
        if feat == 'SOC_Consumed':
            v = v / 100.0
        row[feat] = v
    X_one = pd.DataFrame([row])[cols]
    pred = float(max(model.predict(X_one)[0], 0.0))

    v_mean = inputs.get('Velocity_mean', 40)
    dur = inputs.get('Duration', 20)
    soc_pct = inputs.get('SOC_Consumed', 10)
    phys_dist = v_mean * dur / 60
    eff = pred / soc_pct if soc_pct > 0.1 else 0
    diff_pct = (pred - phys_dist) / phys_dist * 100 if phys_dist > 0 else 0

    with right:
        # 예측 결과 박스
        st.markdown(
            f'<div class="pred-box">'
            f'<div style="color:{BMW_LIGHT};font-size:.8rem;letter-spacing:2px;'
            f'text-transform:uppercase;margin-bottom:8px">예측 주행거리</div>'
            f'<div class="pred-num">{pred:,.1f}<span class="pred-unit"> km</span></div>'
            f'<div class="pred-label">AI MODEL PREDICTION</div>'
            f'</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

        # 속도계 게이지
        gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=v_mean,
            number={'suffix': ' km/h', 'font': {'size': 28, 'color': TXT, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 110], 'tickcolor': SUB,
                         'tickfont': {'color': SUB, 'size': 10}},
                'bar': {'color': BMW_BLUE, 'thickness': 0.3},
                'bgcolor': PANEL2, 'borderwidth': 0,
                'steps': [
                    {'range': [0, 40],  'color': '#0d2040'},
                    {'range': [40, 75], 'color': '#102a50'},
                    {'range': [75, 110],'color': '#0f3060'}],
                'threshold': {
                    'line': {'color': AMBER, 'width': 3},
                    'thickness': 0.8,
                    'value': inputs.get('Velocity_max', v_mean),
                },
            }))
        gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)', font={'color': TXT})
        st.plotly_chart(gauge, use_container_width=True, config={'displayModeBar': False})
        st.caption(f'● 파란 바 = 평균 속도 · 주황 눈금 = 최고 속도 ({inputs.get("Velocity_max",0):.0f} km/h)')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 보조 KPI
    k1, k2, k3, k4 = st.columns(4)
    arrow = '▲' if diff_pct >= 0 else '▼'
    arrow_col = GREEN if diff_pct >= 0 else RED
    for col2, val, lab in [
        (k1, f'{phys_dist:,.1f} km', '물리식 거리 (v×t)'),
        (k2, f'{dur:,.0f} min', '주행 시간'),
        (k3, f'{soc_pct:,.1f} %', 'SOC 소모'),
        (k4, f'{eff:,.2f} km/%', 'SOC 효율'),
    ]:
        col2.markdown(f'<div class="kpi"><div class="v">{val}</div>'
                      f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # 예측 vs 물리식 비교
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-head"><span>⚖️</span>모델 예측 vs 물리식 비교</div>',
                    unsafe_allow_html=True)
        cmp_fig = go.Figure()
        cmp_fig.add_trace(go.Bar(
            x=['AI 모델 예측', '물리식 (v×t)'],
            y=[pred, phys_dist],
            marker_color=[BMW_BLUE, SUB],
            text=[f'{pred:.1f} km', f'{phys_dist:.1f} km'],
            textposition='outside',
            textfont={'color': TXT, 'size': 13},
            width=0.45,
        ))
        cmp_fig.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'km',
                   'range': [0, max(pred, phys_dist) * 1.25]},
            xaxis={'color': TXT},
        )
        st.plotly_chart(cmp_fig, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        st.markdown('<div class="sec-head"><span>💡</span>예측 인사이트</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight">
          AI 모델이 물리식보다 <strong style="color:{arrow_col}">{arrow} {abs(diff_pct):.1f}%</strong>
          {'높게' if diff_pct >= 0 else '낮게'} 예측했습니다.
          {'배터리 효율·난방 등 복합 요소가 거리를 늘립니다.' if diff_pct >= 0
           else '실제 저항·에너지 손실로 물리식보다 짧을 수 있습니다.'}
        </div>
        <div class="insight">
          SOC 효율 <strong>{eff:.2f} km/%</strong> — 배터리 1% 소모당 {eff:.2f} km 주행 가능.
        </div>
        <div class="insight">
          평균 속도 <strong>{v_mean:.0f} km/h</strong> 구간에서
          {'고속 주행으로 에너지 소모가 증가합니다.' if v_mean > 80
           else '효율적인 속도 범위입니다.' if v_mean > 40
           else '저속 주행, 도심 구간으로 판단됩니다.'}
        </div>
        """, unsafe_allow_html=True)

        # SOC vs 예측 미니 라인
        soc_range = np.linspace(1, 60, 40)
        preds_soc = []
        for sv in soc_range:
            r2 = dict(row)
            r2['SOC_Consumed'] = sv / 100.0
            preds_soc.append(float(max(model.predict(pd.DataFrame([r2])[cols])[0], 0)))
        soc_line = go.Figure()
        soc_line.add_trace(go.Scatter(
            x=soc_range, y=preds_soc, mode='lines',
            line={'color': BMW_BLUE, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.15)',
        ))
        soc_line.add_vline(x=soc_pct, line_color=AMBER, line_dash='dash',
                           annotation_text=f'{soc_pct:.0f}%', annotation_font_color=AMBER)
        soc_line.update_layout(
            height=210, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': 'SOC 소모 (%)'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '예측 km'},
            showlegend=False,
        )
        st.plotly_chart(soc_line, use_container_width=True, config={'displayModeBar': False})
        st.caption('SOC 소모에 따른 예측 주행거리 변화 (현재 조건 고정)')


# ════════════════════════════════════════════════════════════════
# PAGE 3 : 모델 성능 분석
# ════════════════════════════════════════════════════════════════
elif page == "📊  모델 성능 분석":
    st.markdown('<div class="bmw-title">📊 모델 성능 분석</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bmw-sub">CatBoost 모델의 예측 정확도 · 잔차 분석 · 오차 분포</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # KPI
    k1, k2, k3, k4 = st.columns(4)
    r2c = GREEN if metrics['R2'] >= 0.8 else AMBER
    for col2, val, lab, vc in [
        (k1, f"{metrics['R2']:.4f}", "R² Score", r2c),
        (k2, f"{metrics['MAE']:.2f} km", "MAE", TXT),
        (k3, f"{metrics['RMSE']:.2f} km", "RMSE", TXT),
        (k4, f"{metrics['n']}건", "전체 데이터", TXT),
    ]:
        col2.markdown(
            f'<div class="kpi"><div class="v" style="color:{vc}">{val}</div>'
            f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    yte = metrics['yte']
    pred_te = metrics['pred_te']
    residuals = yte - pred_te

    # 예측 vs 실제 + 잔차
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-head"><span>🎯</span>예측값 vs 실제값</div>',
                    unsafe_allow_html=True)
        sc = go.Figure()
        sc.add_trace(go.Scatter(
            x=yte, y=pred_te, mode='markers',
            marker={'color': BMW_BLUE, 'size': 5, 'opacity': 0.65},
            name='예측 vs 실제',
        ))
        lim = max(yte.max(), pred_te.max()) * 1.05
        sc.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim],
            mode='lines', line={'color': AMBER, 'dash': 'dash', 'width': 1.5},
            name='완벽 예측선',
        ))
        sc.update_layout(
            height=350, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': '실제 주행거리 (km)'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '예측 주행거리 (km)'},
            legend={'font': {'color': SUB}},
        )
        st.plotly_chart(sc, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div class="insight">점이 주황 대각선에 가까울수록 예측이 정확합니다.
        현재 R² <strong>{metrics['R2']:.3f}</strong>으로 실제 변동의
        {metrics['R2']*100:.0f}%를 설명합니다.</div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="sec-head"><span>📉</span>잔차 분포</div>',
                    unsafe_allow_html=True)
        res_hist = go.Figure()
        res_hist.add_trace(go.Histogram(
            x=residuals, nbinsx=35,
            marker_color=BMW_BLUE, opacity=0.8, name='잔차',
        ))
        res_hist.add_vline(x=0, line_color=AMBER, line_width=2,
                           annotation_text='잔차=0', annotation_font_color=AMBER)
        res_hist.update_layout(
            height=350, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': '잔차 (실제 - 예측) km'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '빈도'},
            showlegend=False,
        )
        st.plotly_chart(res_hist, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div class="insight">잔차 평균 <strong>{residuals.mean():.2f} km</strong>,
        표준편차 <strong>{residuals.std():.2f} km</strong>.
        0에 가까운 대칭 분포일수록 편향 없는 모델입니다.</div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 잔차 vs 예측값 + Q-Q 스타일
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="sec-head"><span>📌</span>잔차 vs 예측값 (이분산성 확인)</div>',
                    unsafe_allow_html=True)
        rv = go.Figure()
        rv.add_trace(go.Scatter(
            x=pred_te, y=residuals, mode='markers',
            marker={'color': BMW_BLUE, 'size': 5, 'opacity': 0.6},
        ))
        rv.add_hline(y=0, line_color=AMBER, line_width=1.5, line_dash='dash')
        rv.update_layout(
            height=300, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': '예측값 (km)'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '잔차 (km)'},
            showlegend=False,
        )
        st.plotly_chart(rv, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        st.markdown('<div class="sec-head"><span>📊</span>오차 누적 분포 (CDF)</div>',
                    unsafe_allow_html=True)
        abs_err = np.sort(np.abs(residuals))
        cdf = np.arange(1, len(abs_err) + 1) / len(abs_err)
        cdf_fig = go.Figure()
        cdf_fig.add_trace(go.Scatter(
            x=abs_err, y=cdf * 100, mode='lines',
            line={'color': BMW_BLUE, 'width': 2.5},
            fill='tozeroy', fillcolor='rgba(28,105,212,0.12)',
        ))
        p50 = float(np.percentile(abs_err, 50))
        p80 = float(np.percentile(abs_err, 80))
        cdf_fig.add_vline(x=p50, line_color=GREEN, line_dash='dash',
                          annotation_text=f'50%: {p50:.1f}km', annotation_font_color=GREEN)
        cdf_fig.add_vline(x=p80, line_color=AMBER, line_dash='dash',
                          annotation_text=f'80%: {p80:.1f}km', annotation_font_color=AMBER)
        cdf_fig.update_layout(
            height=300, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': '절대 오차 (km)'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '누적 비율 (%)'},
            showlegend=False,
        )
        st.plotly_chart(cdf_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div class="insight">예측의 <strong>50%</strong>는 ±{p50:.1f} km 이내,
        <strong>80%</strong>는 ±{p80:.1f} km 이내에서 맞습니다.</div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4 : 데이터 인사이트
# ════════════════════════════════════════════════════════════════
elif page == "📈  데이터 인사이트":
    st.markdown('<div class="bmw-title">📈 데이터 인사이트</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bmw-sub">주행 패턴 분석 · 변수 간 관계 · 주행 조건별 효율</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 속도 vs 거리 + 주행시간 vs 거리
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="sec-head"><span>⚡</span>평균 속도 vs 주행거리</div>',
                    unsafe_allow_html=True)
        if 'Velocity_mean' in df.columns and 'Distance' in df.columns:
            color_col = df['SOC_Consumed'] * 100 if 'SOC_Consumed' in df.columns else None
            sc2 = go.Figure()
            sc2.add_trace(go.Scatter(
                x=df['Velocity_mean'], y=df['Distance'],
                mode='markers',
                marker={
                    'color': df['SOC_Consumed'] * 100 if 'SOC_Consumed' in df.columns
                             else BMW_BLUE,
                    'colorscale': [[0, '#0a2040'], [0.5, BMW_BLUE], [1, '#7ab8f5']],
                    'size': 5, 'opacity': 0.65,
                    'colorbar': {'title': 'SOC%', 'tickfont': {'color': SUB}},
                    'showscale': 'SOC_Consumed' in df.columns,
                },
            ))
            sc2.update_layout(
                height=320, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis={'gridcolor': LINE, 'color': SUB, 'title': '평균 속도 (km/h)'},
                yaxis={'gridcolor': LINE, 'color': SUB, 'title': '주행거리 (km)'},
                showlegend=False,
            )
            st.plotly_chart(sc2, use_container_width=True, config={'displayModeBar': False})

    with col_r:
        st.markdown('<div class="sec-head"><span>⏱️</span>주행 시간 vs 주행거리</div>',
                    unsafe_allow_html=True)
        if 'Duration' in df.columns and 'Distance' in df.columns:
            sc3 = go.Figure()
            sc3.add_trace(go.Scatter(
                x=df['Duration'], y=df['Distance'],
                mode='markers',
                marker={'color': BMW_LIGHT, 'size': 5, 'opacity': 0.6},
            ))
            sc3.update_layout(
                height=320, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis={'gridcolor': LINE, 'color': SUB, 'title': '주행 시간 (min)'},
                yaxis={'gridcolor': LINE, 'color': SUB, 'title': '주행거리 (km)'},
                showlegend=False,
            )
            st.plotly_chart(sc3, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 속도 구간별 박스플롯
    st.markdown('<div class="sec-head"><span>🚗</span>속도 구간별 주행거리 분포</div>',
                unsafe_allow_html=True)
    if 'Velocity_mean' in df.columns and 'Distance' in df.columns:
        bins = [0, 30, 50, 70, 90, 160]
        labels_bin = ['0-30', '30-50', '50-70', '70-90', '90+']
        df['속도구간'] = pd.cut(df['Velocity_mean'], bins=bins, labels=labels_bin)
        box_colors = ['#0a2d5a', '#0e3d7a', BMW_BLUE, BMW_LIGHT, '#7ab8f5']
        box_fig = go.Figure()
        for i, grp in enumerate(labels_bin):
            d = df[df['속도구간'] == grp]['Distance']
            if len(d) > 0:
                box_fig.add_trace(go.Box(
                    y=d, name=f'{grp} km/h',
                    marker_color=box_colors[i % len(box_colors)],
                    line_color=BMW_BLUE,
                ))
        box_fig.update_layout(
            height=340, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis={'color': TXT, 'title': '속도 구간'},
            yaxis={'gridcolor': LINE, 'color': SUB, 'title': '주행거리 (km)'},
            showlegend=False,
        )
        st.plotly_chart(box_fig, use_container_width=True, config={'displayModeBar': False})
        df.drop(columns=['속도구간'], inplace=True, errors='ignore')

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 배터리 온도 / 난방 출력 영향
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-head"><span>🌡️</span>배터리 온도 변동 vs 주행거리</div>',
                    unsafe_allow_html=True)
        if 'Battery_Temperature_std' in df.columns and 'Distance' in df.columns:
            bt = go.Figure()
            bt.add_trace(go.Scatter(
                x=df['Battery_Temperature_std'], y=df['Distance'],
                mode='markers',
                marker={'color': AMBER, 'size': 4, 'opacity': 0.55},
            ))
            bt.update_layout(
                height=280, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis={'gridcolor': LINE, 'color': SUB, 'title': '배터리 온도 표준편차 (°C)'},
                yaxis={'gridcolor': LINE, 'color': SUB, 'title': '주행거리 (km)'},
                showlegend=False,
            )
            st.plotly_chart(bt, use_container_width=True, config={'displayModeBar': False})

    with col_b:
        st.markdown('<div class="sec-head"><span>♨️</span>난방 출력 vs 주행거리</div>',
                    unsafe_allow_html=True)
        if 'Heating_Power_CAN_mean' in df.columns and 'Distance' in df.columns:
            hp = go.Figure()
            hp.add_trace(go.Scatter(
                x=df['Heating_Power_CAN_mean'], y=df['Distance'],
                mode='markers',
                marker={'color': RED, 'size': 4, 'opacity': 0.55},
            ))
            hp.update_layout(
                height=280, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis={'gridcolor': LINE, 'color': SUB, 'title': '평균 난방 출력 (kW)'},
                yaxis={'gridcolor': LINE, 'color': SUB, 'title': '주행거리 (km)'},
                showlegend=False,
            )
            st.plotly_chart(hp, use_container_width=True, config={'displayModeBar': False})

    # 인사이트 요약
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    corr_v = df['Velocity_mean'].corr(df['Distance']) if 'Velocity_mean' in df.columns else 0
    corr_d = df['Duration'].corr(df['Distance']) if 'Duration' in df.columns else 0
    corr_s = df['SOC_Consumed'].corr(df['Distance']) if 'SOC_Consumed' in df.columns else 0
    st.markdown(f"""
    <div class="sec-head"><span>💡</span>주요 인사이트</div>
    <div class="insight"><strong>속도 ↔ 거리</strong> 상관계수 {corr_v:.3f} —
    {'강한 양의 상관. 빠를수록 먼 거리를 주행합니다.' if corr_v > 0.5
     else '약한 상관. 속도보다 주행시간이 더 영향이 큽니다.'}</div>
    <div class="insight"><strong>주행시간 ↔ 거리</strong> 상관계수 {corr_d:.3f} —
    {'주행 시간이 거리의 가장 강한 결정 인자입니다.' if corr_d > 0.7
     else '주행시간과 거리는 상당한 관계를 보입니다.'}</div>
    <div class="insight"><strong>SOC 소모 ↔ 거리</strong> 상관계수 {corr_s:.3f} —
    {'배터리를 더 쓸수록 더 먼 거리를 주행합니다.' if corr_s > 0 else '음의 상관 — 비효율 구간에서 SOC 소모가 큽니다.'}</div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 5 : 변수 중요도
# ════════════════════════════════════════════════════════════════
elif page == "🧠  변수 중요도":
    st.markdown('<div class="bmw-title">🧠 변수 중요도</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bmw-sub">CatBoost 피처 중요도 · 주요 변수별 주행거리 민감도 분석</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown('<div class="sec-head"><span>📊</span>피처 중요도 (상위 15개)</div>',
                    unsafe_allow_html=True)
        top15 = importance.head(15).iloc[::-1].copy()
        labels15 = [META[f][0] if f in META else f.replace('_', ' ')
                    for f in top15['Feature']]
        max_imp = top15['Importance'].max()
        colors_fi = [
            f'rgba(28,105,212,{0.45 + 0.55 * v / max_imp:.2f})'
            for v in top15['Importance']
        ]
        fi_fig = go.Figure(go.Bar(
            x=top15['Importance'], y=labels15, orientation='h',
            marker_color=colors_fi,
            text=[f'{v:.1f}' for v in top15['Importance']],
            textposition='outside',
            textfont={'color': SUB, 'size': 10},
        ))
        fi_fig.update_layout(
            height=480, paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis={'gridcolor': LINE, 'color': SUB, 'title': '중요도'},
            yaxis={'color': TXT, 'tickfont': {'size': 11}},
        )
        st.plotly_chart(fi_fig, use_container_width=True, config={'displayModeBar': False})

    with col_r:
        st.markdown('<div class="sec-head"><span>💡</span>상위 변수 해석</div>',
                    unsafe_allow_html=True)
        for _, row_fi in importance.head(5).iterrows():
            fname = row_fi['Feature']
            fimp = row_fi['Importance']
            flabel = META[fname][0] if fname in META else fname.replace('_', ' ')
            funit = META[fname][1] if fname in META else ''
            pct = fimp / importance['Importance'].sum() * 100
            st.markdown(f"""
            <div class="insight">
              <strong>{flabel}</strong> ({funit}) —
              중요도 <strong>{fimp:.1f}</strong> ({pct:.1f}%)
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        # 누적 중요도 파이
        top5_imp = importance.head(5).copy()
        top5_imp.loc[len(top5_imp)] = {
            'Feature': '기타',
            'Importance': importance['Importance'].iloc[5:].sum(),
        }
        pie_labels = [META[f][0] if f in META else f.replace('_', ' ')
                      for f in top5_imp['Feature']]
        pie_fig = go.Figure(go.Pie(
            labels=pie_labels,
            values=top5_imp['Importance'],
            hole=0.45,
            marker_colors=[BMW_BLUE, BMW_DARK, BMW_LIGHT, '#0d4a8a', '#3d7dca', LINE],
            textfont={'color': TXT, 'size': 11},
        ))
        pie_fig.update_layout(
            height=280, paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend={'font': {'color': SUB, 'size': 10}},
        )
        st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 상위 4개 변수 민감도 분석
    st.markdown('<div class="sec-head"><span>📈</span>주요 변수 민감도 분석 (다른 조건 고정)</div>',
                unsafe_allow_html=True)

    top4_feats = [f for f in importance['Feature'].head(4) if f in META][:4]
    if len(top4_feats) >= 2:
        sens_cols = st.columns(min(len(top4_feats), 4))
        base_row = {f: float(medians.get(f, 0)) for f in cols}

        for i, feat in enumerate(top4_feats):
            label, unit, lo, hi, _ = META[feat]
            x_range = np.linspace(lo, hi, 50)
            y_preds = []
            for xv in x_range:
                r_tmp = dict(base_row)
                r_tmp[feat] = xv
                y_preds.append(float(max(model.predict(pd.DataFrame([r_tmp])[cols])[0], 0)))
            s_fig = go.Figure()
            s_fig.add_trace(go.Scatter(
                x=x_range, y=y_preds, mode='lines',
                line={'color': BMW_BLUE, 'width': 2.5},
                fill='tozeroy', fillcolor='rgba(28,105,212,0.12)',
            ))
            s_fig.add_vline(x=float(medians.get(feat, 0)),
                            line_color=AMBER, line_dash='dot', line_width=1.5)
            s_fig.update_layout(
                title={'text': label, 'font': {'color': TXT, 'size': 12}, 'x': 0.5},
                height=220, paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=35, b=10),
                xaxis={'gridcolor': LINE, 'color': SUB, 'title': unit},
                yaxis={'gridcolor': LINE, 'color': SUB, 'title': 'km'},
                showlegend=False,
            )
            sens_cols[i].plotly_chart(s_fig, use_container_width=True,
                                      config={'displayModeBar': False})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption('데이터: BMW i3 Battery & Heating Data in Real Driving Cycles (FTM, TU München)')
