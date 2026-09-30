# Streamlit fleet dashboard for Jet Engine RUL Prediction
# two views: fleet overview + individual engine drill-down

import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ---- config ----
# override with RUL_API_URL=http://localhost:8001 to test against a local API
API_URL = os.environ.get("RUL_API_URL", "https://jet-engine-rul-api.onrender.com")

st.set_page_config(
    page_title="Jet Engine Fleet Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- custom CSS ----
# semi-transparent tints + inherited text colour, so every card reads well in
# both Streamlit's light and dark themes (solid pastels + grey text don't)
st.markdown("""
<style>
    .metric-red, .metric-amber, .metric-green, .metric-neutral {
                    padding:12px 16px; border-radius:8px; margin:4px 0; border-left:4px solid; }
    .metric-red     { background:rgba(229,62,62,0.12);   border-left-color:#e53e3e; }
    .metric-amber   { background:rgba(214,158,46,0.12);  border-left-color:#d69e2e; }
    .metric-green   { background:rgba(56,161,105,0.12);  border-left-color:#38a169; }
    .metric-neutral { background:rgba(128,128,128,0.10); border-left-color:#718096; }
    .metric-value { font-size:2rem; font-weight:700; line-height:1; }
    .metric-label { font-size:0.85rem; opacity:0.75; margin-top:4px; }
    .muted        { opacity:0.7; }
    .panel        { background:rgba(128,128,128,0.08); border-radius:8px; }
    .badge-red    { background:#e53e3e; color:white; padding:2px 10px;
                    border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-amber  { background:#d69e2e; color:white; padding:2px 10px;
                    border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-green  { background:#38a169; color:white; padding:2px 10px;
                    border-radius:12px; font-size:0.78rem; font-weight:600; }
    .engine-card  { border:1px solid rgba(128,128,128,0.3); border-radius:10px;
                    padding:16px; margin:6px 0; }
    .stButton>button { width:100%; }
</style>
""", unsafe_allow_html=True)


# ---- data fetching ----
@st.cache_data(ttl=3600)  # fleet data is static — "Refresh Data" clears this
def _api_get(path):
    # raises on failure — Streamlit doesn't cache exceptions, so errors get retried
    r = requests.get(f"{API_URL}{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def api_get(path):
    try:
        return _api_get(path)
    except Exception as e:
        st.error(f"API request failed ({path}): {e}")
        return None


# ---- helper ----
LEVEL_COLORS = {'RED': '#e53e3e', 'AMBER': '#d69e2e', 'GREEN': '#38a169'}


def badge(level):
    return f'<span class="badge-{level.lower()}">{level}</span>'


# ---- model metadata — thresholds, RUL cap, metrics all come from the API ----
# the API is on a free tier that sleeps when idle — the first request can take ~1 minute
with st.spinner("Connecting to the prediction API… (first load can take up to a minute while it wakes up)"):
    meta = api_get("/")
if not meta:
    st.stop()

RED_BELOW   = meta['alert_thresholds']['red_below']
AMBER_BELOW = meta['alert_thresholds']['amber_below']
RUL_CAP     = meta['rul_cap']
perf        = meta['performance']


# ---- sidebar ----
st.sidebar.markdown("## ✈️")
st.sidebar.title("Fleet Monitor")
st.sidebar.markdown(
    f"{meta['dataset']}  \n{meta['model']}  \n"
    f"RMSE {perf['test_rmse']:.2f} · R² {perf['test_r2']:.3f}"
)
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

    fleet = api_get("/fleet")
    if not fleet:
        st.stop()

    st.markdown(
        f"Predictive maintenance dashboard — {fleet['total_engines']} turbofan engines · {meta['dataset']}"
    )

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
        <div class="metric-neutral">
            <div class="metric-value">{fleet['total_engines']}</div>
            <div class="metric-label">Total engines monitored</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- build dataframe ----
    # rename by name, not position — adding a field to the API can't shift the labels
    df = pd.DataFrame(fleet['engines']).rename(columns={
        'engine_id':     'Engine ID',
        'predicted_rul': 'Predicted RUL',
        'actual_rul':    'Actual RUL',
        'alert_level':   'Alert Level',
        'alert_message': 'Message',
    })

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
            color_map = {'RED':   'rgba(229,62,62,0.15)',
                         'AMBER': 'rgba(214,158,46,0.15)',
                         'GREEN': 'rgba(56,161,105,0.15)'}
            color = color_map.get(row['Alert Level'], 'transparent')
            return [f'background-color: {color}'] * len(row)

        st.dataframe(
            df_filtered[['Engine ID', 'Predicted RUL', 'Actual RUL', 'Alert Level']].style.apply(
                highlight_row, axis=1
            ).format({'Predicted RUL': '{:.1f}', 'Actual RUL': '{:.0f}'}),
            width='stretch',
            height=480,
            hide_index=True
        )
        st.caption(
            "Actual RUL is known here because these are NASA's benchmark test engines — "
            f"a real fleet wouldn't have it. The model's predictions are capped at {RUL_CAP} cycles."
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
        fig_hist.add_vline(x=RED_BELOW,   line_dash='dash', line_color=LEVEL_COLORS['RED'],   opacity=0.5)
        fig_hist.add_vline(x=AMBER_BELOW, line_dash='dash', line_color=LEVEL_COLORS['AMBER'], opacity=0.5)
        fig_hist.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            legend_title_text='',
            showlegend=True,
            height=220
        )
        st.plotly_chart(fig_hist, width='stretch')

        # donut chart
        st.subheader("Fleet Health")
        fig_donut = go.Figure(data=[go.Pie(
            labels=['Critical (RED)', 'Warning (AMBER)', 'Healthy (GREEN)'],
            values=[fleet['red_count'], fleet['amber_count'], fleet['green_count']],
            hole=0.6,
            marker_colors=[LEVEL_COLORS['RED'], LEVEL_COLORS['AMBER'], LEVEL_COLORS['GREEN']],
            textinfo='percent+label',
            textfont_size=11
        )])
        fig_donut.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            height=220
        )
        st.plotly_chart(fig_donut, width='stretch')

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
                    <div class="muted" style="font-size:0.8rem">cycles remaining · actual {row['Actual RUL']:.0f}</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════
# VIEW 2 — ENGINE DRILL-DOWN
# ════════════════════════════════════════
else:
    st.title("🔍 Engine Drill-down")
    st.markdown("Select an engine to see detailed sensor analysis and SHAP explanation")

    fleet = api_get("/fleet")
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
    color = LEVEL_COLORS[level]

    with col_sel2:
        st.markdown(f"""
        <div class="panel" style="padding:12px;border-left:5px solid {color};margin-top:4px">
            <span style="font-size:1.1rem;font-weight:600">Engine {selected_id}</span>
            &nbsp;&nbsp;{badge(level)}&nbsp;&nbsp;
            <span style="font-size:1.5rem;font-weight:700;color:{color}">{rul:.1f} cycles remaining</span>
            &nbsp;&nbsp;
            <span class="muted" style="font-size:0.9rem">{engine_row['alert_message']}
                · actual RUL {engine_row['actual_rul']:.0f}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ---- everything for this engine comes from the API ----
    # the API holds each fleet engine's sensor window and does its own scaling,
    # so the dashboard never touches model files or data files
    with st.spinner("Loading engine data and computing SHAP..."):
        sensors = api_get(f"/engines/{selected_id}/sensors")
        pred    = api_get(f"/engines/{selected_id}/explain")

    if sensors:
        feature_cols = sensors['feature_cols']
        engine_seq   = np.array(sensors['scaled_readings'])   # (30, 14)
        n_cycles     = len(engine_seq)

        col_charts, col_shap = st.columns([1.5, 1])

        with col_charts:
            st.subheader(f"Sensor Trends — Last {n_cycles} Cycles")
            st.markdown("*Scaled values — 0 = median reading across the training fleet*")

            # the 6 sensors driving THIS engine's prediction (falls back to
            # the first 6 features if SHAP is unavailable)
            if pred and pred.get('shap_values'):
                top_sensors = [s['sensor'] for s in pred['shap_values'][:6]]
            else:
                top_sensors = feature_cols[:6]
            top_indices = [feature_cols.index(s) for s in top_sensors]

            fig_sensors = go.Figure()
            cycles = list(range(1, n_cycles + 1))

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
                                  opacity=0.4, annotation_text='training median')
            fig_sensors.update_layout(
                xaxis_title=f'cycle (last {n_cycles})',
                yaxis_title='scaled sensor value',
                legend=dict(orientation='h', y=-0.2),
                margin=dict(t=10, b=60, l=10, r=10),
                height=320
            )
            st.plotly_chart(fig_sensors, width='stretch')

            # RUL gauge
            st.subheader("RUL Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode='gauge+number+delta',
                value=rul,
                delta={'reference': AMBER_BELOW, 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, RUL_CAP]},
                    'bar':  {'color': color},
                    'steps': [
                        {'range': [0, RED_BELOW],           'color': '#fff0f0'},
                        {'range': [RED_BELOW, AMBER_BELOW], 'color': '#fffbeb'},
                        {'range': [AMBER_BELOW, RUL_CAP],   'color': '#f0fff4'},
                    ],
                    'threshold': {
                        'line': {'color': LEVEL_COLORS['RED'], 'width': 3},
                        'thickness': 0.75,
                        'value': RED_BELOW
                    }
                },
                title={'text': 'Predicted RUL (cycles)'},
                number={'suffix': ' cycles', 'valueformat': '.1f'}
            ))
            fig_gauge.update_layout(
                margin=dict(t=30, b=10, l=30, r=30),
                height=220
            )
            st.plotly_chart(fig_gauge, width='stretch')

        with col_shap:
            st.subheader("SHAP — Sensor Importance")
            st.markdown("*Which sensors are driving this prediction?*")

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
                st.plotly_chart(fig_shap, width='stretch')

                # maintenance recommendation
                # SHAP says which sensor most influenced the prediction — not
                # which part is physically degrading, so word it that way
                top_sensor = shap_df.iloc[0]['sensor']
                action = {
                    'RED':   'Immediate inspection recommended.',
                    'AMBER': 'Schedule within next maintenance window.',
                    'GREEN': 'Continue standard monitoring.',
                }[level]
                st.markdown(f"""
                <div class="panel" style="border-left:4px solid {color};padding:14px;margin-top:8px">
                    <div style="font-weight:600;margin-bottom:6px">
                        🔧 Maintenance Recommendation
                    </div>
                    <div style="font-size:0.9rem">
                        Predicted RUL for engine {selected_id}: <strong>{rul:.1f} cycles</strong>.
                        <span style="color:{color};font-weight:600">{action}</span><br>
                        <span class="muted">Sensor with the most influence on this prediction:
                        <strong>{top_sensor}</strong>.</span>
                    </div>
                </div>""", unsafe_allow_html=True)