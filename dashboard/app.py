# Streamlit fleet dashboard for Jet Engine RUL Prediction
# two views: fleet overview + individual engine drill-down

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ---- config ----
API_URL = "https://jet-engine-rul-api.onrender.com"

st.set_page_config(
    page_title="Jet Engine Fleet Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- custom CSS ----
st.markdown("""
<style>
    .metric-red   { background:#fff0f0; border-left:4px solid #e53e3e;
                    padding:12px 16px; border-radius:8px; margin:4px 0; }
    .metric-amber { background:#fffbeb; border-left:4px solid #d69e2e;
                    padding:12px 16px; border-radius:8px; margin:4px 0; }
    .metric-green { background:#f0fff4; border-left:4px solid #38a169;
                    padding:12px 16px; border-radius:8px; margin:4px 0; }
    .metric-value { font-size:2rem; font-weight:700; line-height:1; }
    .metric-label { font-size:0.85rem; color:#666; margin-top:4px; }
    .badge-red    { background:#e53e3e; color:white; padding:2px 10px;
                    border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-amber  { background:#d69e2e; color:white; padding:2px 10px;
                    border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-green  { background:#38a169; color:white; padding:2px 10px;
                    border-radius:12px; font-size:0.78rem; font-weight:600; }
    .engine-card  { border:1px solid #e2e8f0; border-radius:10px;
                    padding:16px; margin:6px 0; background:white; }
    .stButton>button { width:100%; }
</style>
""", unsafe_allow_html=True)


# ---- data fetching ----
@st.cache_data(ttl=60)  # refresh every 60 seconds
def fetch_fleet():
    try:
        r = requests.get(f"{API_URL}/fleet", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API connection failed: {e}")
        return None


def fetch_prediction(engine_id, readings):
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"engine_id": engine_id, "readings": readings},
            timeout=30
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


# ---- helper ----
def badge(level):
    return f'<span class="badge-{level.lower()}">{level}</span>'


def rul_color(rul):
    if rul < 30:   return '#e53e3e'
    if rul < 60:   return '#d69e2e'
    return '#38a169'


# ---- sidebar ----
st.sidebar.markdown("## ✈️")
st.sidebar.title("Fleet Monitor")
st.sidebar.markdown("NASA CMAPSS FD001  \nGRU-LSTM + Attention  \nRMSE 14.07 · R² 0.877")
st.sidebar.divider()

view = st.sidebar.radio(
    "View",
    ["Fleet Overview", "Engine Drill-down"],
    index=0
)

st.sidebar.divider()
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Built by Haadhi Mohammed  \n"
    "[GitHub](https://github.com/Haadhi-Mohammed/jet-engine-rul)</small>",
    unsafe_allow_html=True
)


# ════════════════════════════════════════
# VIEW 1 — FLEET OVERVIEW
# ════════════════════════════════════════
if view == "Fleet Overview":

    st.title("✈️ Jet Engine Fleet Monitor")
    st.markdown("Predictive maintenance dashboard — 100 turbofan engines · NASA CMAPSS FD001")

    fleet = fetch_fleet()
    if not fleet:
        st.stop()

    # ---- top metrics ----
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-red">
            <div class="metric-value">{fleet['red_count']}</div>
            <div class="metric-label">🔴 Immediate maintenance</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-amber">
            <div class="metric-value">{fleet['amber_count']}</div>
            <div class="metric-label">🟡 Schedule soon</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-green">
            <div class="metric-value">{fleet['green_count']}</div>
            <div class="metric-label">🟢 Healthy</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-green">
            <div class="metric-value">{fleet['total_engines']}</div>
            <div class="metric-label">Total engines monitored</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- build dataframe ----
    df = pd.DataFrame(fleet['engines'])
    df.columns = ['Engine ID', 'Predicted RUL', 'Alert Level', 'Message']

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("Fleet Status Table")

        # filter
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            alert_filter = st.multiselect(
                "Filter by alert",
                ["RED", "AMBER", "GREEN"],
                default=["RED", "AMBER", "GREEN"]
            )
        with filter_col2:
            sort_by = st.selectbox("Sort by", ["RUL (low→high)", "RUL (high→low)", "Engine ID"])

        df_filtered = df[df['Alert Level'].isin(alert_filter)].copy()

        if sort_by == "RUL (low→high)":
            df_filtered = df_filtered.sort_values('Predicted RUL')
        elif sort_by == "RUL (high→low)":
            df_filtered = df_filtered.sort_values('Predicted RUL', ascending=False)
        else:
            df_filtered = df_filtered.sort_values('Engine ID')

        # colour-coded table
        def highlight_row(row):
            color_map = {'RED': '#fff0f0', 'AMBER': '#fffbeb', 'GREEN': '#f0fff4'}
            color = color_map.get(row['Alert Level'], 'white')
            return [f'background-color: {color}'] * len(row)

        st.dataframe(
            df_filtered[['Engine ID', 'Predicted RUL', 'Alert Level']].style.apply(
                highlight_row, axis=1
            ).format({'Predicted RUL': '{:.1f}'}),
            use_container_width=True,
            height=480
        )

    with col_right:
        st.subheader("RUL Distribution")

        # histogram
        fig_hist = px.histogram(
            df, x='Predicted RUL', nbins=20,
            color='Alert Level',
            color_discrete_map={
                'RED': '#e53e3e',
                'AMBER': '#d69e2e',
                'GREEN': '#38a169'
            },
            labels={'Predicted RUL': 'Predicted RUL (cycles)'}
        )
        fig_hist.add_vline(x=30, line_dash='dash', line_color='#e53e3e', opacity=0.5)
        fig_hist.add_vline(x=60, line_dash='dash', line_color='#d69e2e', opacity=0.5)
        fig_hist.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            legend_title_text='',
            showlegend=True,
            height=220
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # donut chart
        st.subheader("Fleet Health")
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Critical (RED)', 'Warning (AMBER)', 'Healthy (GREEN)'],
            values=[fleet['red_count'], fleet['amber_count'], fleet['green_count']],
            hole=0.6,
            marker_colors=['#e53e3e', '#d69e2e', '#38a169'],
            textinfo='percent+label',
            textfont_size=11
        )])
        fig_donut.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            height=220
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ---- critical engines callout ----
    critical = df[df['Alert Level'] == 'RED'].sort_values('Predicted RUL').head(5)
    if len(critical) > 0:
        st.divider()
        st.subheader("🚨 Most Critical Engines")
        cols = st.columns(len(critical))
        for col, (_, row) in zip(cols, critical.iterrows()):
            with col:
                st.markdown(f"""
                <div class="engine-card" style="border-left:4px solid #e53e3e;">
                    <div style="font-size:1.3rem;font-weight:700">Engine {int(row['Engine ID'])}</div>
                    <div style="font-size:2rem;font-weight:700;color:#e53e3e">{row['Predicted RUL']:.1f}</div>
                    <div style="font-size:0.8rem;color:#666">cycles remaining</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════
# VIEW 2 — ENGINE DRILL-DOWN
# ════════════════════════════════════════
else:
    st.title("🔍 Engine Drill-down")
    st.markdown("Select an engine to see detailed sensor analysis and SHAP explanation")

    fleet = fetch_fleet()
    if not fleet:
        st.stop()

    df = pd.DataFrame(fleet['engines'])

    # ---- engine selector ----
    col_sel1, col_sel2 = st.columns([1, 3])

    with col_sel1:
        # default to most critical engine
        engine_ids = df.sort_values('predicted_rul')['engine_id'].tolist()
        selected_id = st.selectbox("Select Engine", engine_ids, index=0)

    engine_row = df[df['engine_id'] == selected_id].iloc[0]
    rul   = engine_row['predicted_rul']
    level = engine_row['alert_level']
    color = rul_color(rul)

    with col_sel2:
        st.markdown(f"""
        <div style="padding:12px;background:#f8f9fa;border-radius:8px;
                    border-left:5px solid {color};margin-top:4px">
            <span style="font-size:1.1rem;font-weight:600">Engine {selected_id}</span>
            &nbsp;&nbsp;{badge(level)}&nbsp;&nbsp;
            <span style="font-size:1.5rem;font-weight:700;color:{color}">{rul:.1f} cycles remaining</span>
            &nbsp;&nbsp;
            <span style="color:#666;font-size:0.9rem">{engine_row['alert_message']}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- load preprocessed test data for sensor charts ----
    # loading directly from saved numpy arrays — no API call needed here
    try:
        import pickle
        from pathlib import Path

        MODELS_DIR = Path(__file__).parent.parent / 'models'
        DATA_DIR   = Path(__file__).parent.parent / 'data' / 'processed'

        X_test = np.load(DATA_DIR / 'X_test.npy')   # (100, 30, 14)
        y_test = np.load(DATA_DIR / 'y_test.npy')

        with open(MODELS_DIR / 'feature_cols.pkl', 'rb') as f:
            feature_cols = pickle.load(f)

        # engine index is engine_id - 1
        engine_idx = selected_id - 1
        engine_seq = X_test[engine_idx]  # (30, 14)

        data_loaded = True
    except Exception as e:
        st.warning(f"Could not load local data: {e}")
        data_loaded = False

    if data_loaded:
        col_charts, col_shap = st.columns([1.5, 1])

        with col_charts:
            st.subheader("Sensor Trends — Last 30 Cycles")
            st.markdown("*Scaled values — positive = above normal, negative = below normal*")

            # top 6 most important sensors based on global SHAP knowledge
            # s_11, s_12, s_20, s_9, s_7, s_17 from Cell 8 output
            top_sensors = ['s_11', 's_12', 's_20', 's_9', 's_7', 's_17']
            top_indices = [feature_cols.index(s) for s in top_sensors]

            fig_sensors = go.Figure()
            cycles = list(range(1, 31))

            colors_sensors = [
                '#e53e3e', '#d69e2e', '#38a169',
                '#3182ce', '#805ad5', '#dd6b20'
            ]

            for sensor, idx, c in zip(top_sensors, top_indices, colors_sensors):
                fig_sensors.add_trace(go.Scatter(
                    x=cycles,
                    y=engine_seq[:, idx],
                    name=sensor,
                    line=dict(color=c, width=1.5),
                    mode='lines'
                ))

            fig_sensors.add_hline(y=0, line_dash='dash', line_color='gray',
                                  opacity=0.4, annotation_text='normal baseline')
            fig_sensors.update_layout(
                xaxis_title='cycle (last 30)',
                yaxis_title='scaled sensor value',
                legend=dict(orientation='h', y=-0.2),
                margin=dict(t=10, b=60, l=10, r=10),
                height=320
            )
            st.plotly_chart(fig_sensors, use_container_width=True)

            # RUL gauge
            st.subheader("RUL Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number+delta',
                value=rul,
                delta={'reference': 60, 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 125]},
                    'bar':  {'color': color},
                    'steps': [
                        {'range': [0, 30],  'color': '#fff0f0'},
                        {'range': [30, 60], 'color': '#fffbeb'},
                        {'range': [60, 125],'color': '#f0fff4'},
                    ],
                    'threshold': {
                        'line': {'color': '#e53e3e', 'width': 3},
                        'thickness': 0.75,
                        'value': 30
                    }
                },
                title={'text': 'Predicted RUL (cycles)'},
                number={'suffix': ' cycles', 'valueformat': '.1f'}
            ))
            fig_gauge.update_layout(
                margin=dict(t=30, b=10, l=30, r=30),
                height=220
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_shap:
            st.subheader("SHAP — Sensor Importance")
            st.markdown("*Which sensors are driving this prediction?*")

            # compute SHAP live for this engine via API
            # building a dummy request using the scaled sensor values
            # converting back to approximate raw values for the API
            # (in production the API would receive raw sensor data)

            # for the dashboard we call /predict with the scaled sequence
            # reusing the scaled X_test data directly
            readings = []
            for cycle in engine_seq:
                readings.append({
                    's_2':  float(cycle[feature_cols.index('s_2')]),
                    's_3':  float(cycle[feature_cols.index('s_3')]),
                    's_4':  float(cycle[feature_cols.index('s_4')]),
                    's_7':  float(cycle[feature_cols.index('s_7')]),
                    's_8':  float(cycle[feature_cols.index('s_8')]),
                    's_9':  float(cycle[feature_cols.index('s_9')]),
                    's_11': float(cycle[feature_cols.index('s_11')]),
                    's_12': float(cycle[feature_cols.index('s_12')]),
                    's_13': float(cycle[feature_cols.index('s_13')]),
                    's_14': float(cycle[feature_cols.index('s_14')]),
                    's_15': float(cycle[feature_cols.index('s_15')]),
                    's_17': float(cycle[feature_cols.index('s_17')]),
                    's_20': float(cycle[feature_cols.index('s_20')]),
                    's_21': float(cycle[feature_cols.index('s_21')]),
                })

            with st.spinner("Computing SHAP..."):
                pred = fetch_prediction(selected_id, readings)

            if pred and pred.get('shap_values'):
                shap_df = pd.DataFrame(pred['shap_values'])

                fig_shap = go.Figure(go.Bar(
                    x=shap_df['importance'],
                    y=shap_df['sensor'],
                    orientation='h',
                    marker_color=[
                        '#e53e3e' if v > shap_df['importance'].median()
                        else '#3182ce'
                        for v in shap_df['importance']
                    ],
                    text=[f"{v:.3f}" for v in shap_df['importance']],
                    textposition='outside'
                ))
                fig_shap.update_layout(
                    xaxis_title='mean |SHAP| importance',
                    yaxis=dict(autorange='reversed'),
                    margin=dict(t=10, b=20, l=10, r=60),
                    height=340
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                # maintenance recommendation
                top_sensor = shap_df.iloc[0]['sensor']
                st.markdown(f"""
                <div style="background:#fff8f0;border-left:4px solid {color};
                            padding:14px;border-radius:8px;margin-top:8px">
                    <div style="font-weight:600;margin-bottom:6px">
                        🔧 Maintenance Recommendation
                    </div>
                    <div style="font-size:0.9rem;color:#444">
                        Engine {selected_id} shows strongest degradation signal in
                        <strong>{top_sensor}</strong>.
                        Predicted RUL: <strong>{rul:.1f} cycles</strong>.
                        {'<span style="color:#e53e3e;font-weight:600"> Immediate inspection recommended.</span>' if level == 'RED'
                         else '<span style="color:#d69e2e;font-weight:600"> Schedule within next maintenance window.</span>' if level == 'AMBER'
                         else '<span style="color:#38a169;font-weight:600"> Continue standard monitoring.</span>'}
                    </div>
                </div>""", unsafe_allow_html=True)