import time
from datetime import datetime

import pandas as pd
import streamlit as st

from live_backend import LiveNIDS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-NIDS Live Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #8b949e;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 17px;
        font-weight: 700;
    }

    .status-live {
        background: #073b27;
        border: 1px solid #238636;
        color: #3fb950;
    }

    .status-stop {
        background: #3b2610;
        border: 1px solid #d29922;
        color: #d29922;
    }

    .attack-box {
        background: #4b1515;
        border: 1px solid #f85149;
        color: #ff7b72;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 800;
    }

    .benign-box {
        background: #073b27;
        border: 1px solid #238636;
        color: #3fb950;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: 800;
    }

    .pipeline {
        background: #161b22;
        border-radius: 10px;
        padding: 20px;
        line-height: 2;
        border: 1px solid #30363d;
    }

    .flow-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .flow-label {
        color: #8b949e;
        font-size: 14px;
    }

    .flow-value {
        font-size: 22px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "nids" not in st.session_state:
    st.session_state.nids = LiveNIDS()

nids = st.session_state.nids


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def running_status():
    """
    Supports both:
        nids.is_running
    and
        nids.is_running()
    """
    try:
        value = nids.is_running

        if callable(value):
            return bool(value())

        return bool(value)

    except Exception:
        return False


def safe_get(name, default=0):
    try:
        return getattr(nids, name, default)
    except Exception:
        return default


def prediction_dataframe():
    predictions = safe_get("predictions", [])

    if not predictions:
        return pd.DataFrame()

    try:
        df = pd.DataFrame(predictions)
    except Exception:
        return pd.DataFrame()

    return df


def format_probability(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def get_device():
    try:
        device = getattr(nids, "device", None)

        if device is None:
            return "Unknown"

        return str(device).upper()

    except Exception:
        return "Unknown"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Live Capture")

    st.markdown("### Network Interface")

    interface = st.selectbox(
        "Select network interface",
        [
            "Wi-Fi",
            "Ethernet",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    running = running_status()

    if running:

        if st.button(
            "■ Stop Live Monitoring",
            use_container_width=True,
        ):
            try:
                nids.stop_capture()
                st.success("Live monitoring stopped.")
                st.rerun()

            except Exception as e:
                st.error(f"Unable to stop monitoring: {e}")

    else:

        if st.button(
            "▶ Start Live Monitoring",
            type="primary",
            use_container_width=True,
        ):
            try:

                # Current backend starts the configured capture
                # process itself.
                nids.start_capture(interface)

                st.success("Live monitoring started.")
                time.sleep(0.5)
                st.rerun()

            except Exception as e:
                st.error(f"Unable to start monitoring: {e}")

    st.divider()

    st.markdown("### AI Pipeline")

    st.markdown(
        """
        <div class="pipeline">

        <b>Wi-Fi</b>

        ↓

        <b>CICFlowMeter</b>

        ↓

        <b>61 Features</b>

        ↓

        <b>StandardScaler</b>

        ↓

        <b>Binary Transformer</b>

        ↓

        <b>Multi-class Transformer V2</b>

        ↓

        <b>AI Prediction</b>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("AI-NIDS")
    st.caption("Real-time Network Intrusion Detection")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🛡️ AI-NIDS Live Monitor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Real Wi-Fi traffic capture → CICFlowMeter → 61 features → "
    "Transformer AI detection"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# PROCESS NEW FLOWS
# ============================================================

if running:

    try:
        nids.process_new_flows()

    except Exception as e:
        st.session_state.last_processing_error = repr(e)


# ============================================================
# STATUS
# ============================================================

running = running_status()

if running:

    st.markdown(
        """
        <div class="status-box status-live">
        🟢 LIVE CAPTURE ACTIVE
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="status-box status-stop">
        🟡 LIVE CAPTURE STOPPED
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TOP METRICS
# ============================================================

total_flows = int(safe_get("total_flows", 0))
total_benign = int(safe_get("total_benign", 0))
total_attacks = int(safe_get("total_attacks", 0))
high_severity = int(safe_get("high_severity", 0))

attack_rate = (
    (total_attacks / total_flows) * 100
    if total_flows > 0
    else 0
)

device = get_device()

m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        "Device",
        device,
    )

with m2:
    st.metric(
        "Total Flows",
        f"{total_flows:,}",
    )

with m3:
    st.metric(
        "Benign",
        f"{total_benign:,}",
    )

with m4:
    st.metric(
        "Attacks",
        f"{total_attacks:,}",
    )

with m5:
    st.metric(
        "High Severity",
        f"{high_severity:,}",
    )


st.divider()


# ============================================================
# CURRENT FLOW
# ============================================================

st.markdown("## 🔎 Current Flow Detection")

df = prediction_dataframe()

if df.empty:

    st.info(
        "Waiting for live network flows..."
    )

else:

    latest = df.iloc[-1]

    prediction = str(
        latest.get("prediction", "UNKNOWN")
    )

    attack_probability = latest.get(
        "attack_probability",
        0,
    )

    family = str(
        latest.get(
            "attack_family",
            "UNKNOWN",
        )
    )

    confidence = latest.get(
        "family_confidence",
        0,
    )

    severity = str(
        latest.get(
            "severity",
            "NONE",
        )
    )

    src_ip = latest.get(
        "src_ip",
        "",
    )

    src_port = latest.get(
        "src_port",
        "",
    )

    dst_ip = latest.get(
        "dst_ip",
        "",
    )

    dst_port = latest.get(
        "dst_port",
        "",
    )

    protocol = latest.get(
        "protocol",
        "",
    )

    timestamp = latest.get(
        "timestamp",
        "",
    )


    # --------------------------------------------------------
    # Prediction Banner
    # --------------------------------------------------------

    if prediction == "ATTACK":

        st.markdown(
            """
            <div class="attack-box">
            🚨 ATTACK DETECTED
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="benign-box">
            🟢 BENIGN TRAFFIC
            </div>
            """,
            unsafe_allow_html=True,
        )


    st.write("")


    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            '<div class="flow-label">Prediction</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="flow-value">{prediction}</div>',
            unsafe_allow_html=True,
        )


    with c2:

        st.markdown(
            '<div class="flow-label">Attack Probability</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="flow-value">'
            f'{format_probability(attack_probability)}'
            f'</div>',
            unsafe_allow_html=True,
        )


    with c3:

        st.markdown(
            '<div class="flow-label">Attack Family</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="flow-value">{family}</div>',
            unsafe_allow_html=True,
        )


    with c4:

        st.markdown(
            '<div class="flow-label">Severity</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="flow-value">{severity}</div>',
            unsafe_allow_html=True,
        )


    st.write("")


    st.markdown(
        f"""
        <div class="flow-card">

        <span class="flow-label">Flow</span><br>

        <code>
        {src_ip}:{src_port}
        →
        {dst_ip}:{dst_port}
        </code>

        <br><br>

        <span class="flow-label">Protocol</span><br>

        <code>{protocol}</code>

        &nbsp;&nbsp;&nbsp;

        <span class="flow-label">Family Confidence</span><br>

        <code>{format_probability(confidence)}</code>

        &nbsp;&nbsp;&nbsp;

        <span class="flow-label">Time</span><br>

        <code>{timestamp}</code>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# TRAFFIC ANALYTICS
# ============================================================

st.markdown("## 📊 Traffic Analytics")

if df.empty:

    st.info("Analytics will appear when traffic is captured.")

else:

    chart1, chart2 = st.columns(2)

    # --------------------------------------------------------
    # Flow count over time
    # --------------------------------------------------------

    with chart1:

        st.markdown("### 📈 Flow Activity")

        temp = df.copy()

        temp["Flow Number"] = range(
            1,
            len(temp) + 1,
        )

        temp["Flows"] = 1

        flow_chart = (
            temp[
                [
                    "Flow Number",
                    "Flows",
                ]
            ]
            .set_index("Flow Number")
        )

        st.line_chart(
            flow_chart,
            height=300,
        )


    # --------------------------------------------------------
    # Benign vs Attack
    # --------------------------------------------------------

    with chart2:

        st.markdown("### 📊 Benign vs Attack")

        chart_data = pd.DataFrame(
            {
                "Traffic": [
                    "Benign",
                    "Attack",
                ],
                "Count": [
                    total_benign,
                    total_attacks,
                ],
            }
        )

        st.bar_chart(
            chart_data.set_index("Traffic"),
            height=300,
        )


st.divider()


# ============================================================
# ATTACK FAMILY DISTRIBUTION
# ============================================================

st.markdown("## 🧠 Attack Family Distribution")

if df.empty:

    st.info(
        "Attack-family statistics will appear here."
    )

else:

    family_df = df.copy()

    family_df["attack_family"] = (
        family_df["attack_family"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    family_counts = (
        family_df["attack_family"]
        .value_counts()
        .rename("Count")
        .to_frame()
    )

    st.bar_chart(
        family_counts,
        height=300,
    )


st.divider()


# ============================================================
# RECENT NETWORK TRAFFIC
# ============================================================

st.markdown("## 🔬 Recent Network Traffic")

if df.empty:

    st.info(
        "Waiting for live flows..."
    )

else:

    display_df = df.copy()

    # Keep latest 100 rows
    display_df = display_df.tail(100)

    # --------------------------------------------------------
    # Format columns
    # --------------------------------------------------------

    if "timestamp" in display_df.columns:

        display_df["Time"] = (
            display_df["timestamp"]
            .astype(str)
        )

    else:

        display_df["Time"] = ""


    if "src_ip" in display_df.columns:

        display_df["Source"] = (
            display_df["src_ip"].astype(str)
            + ":"
            + display_df["src_port"].astype(str)
        )

    else:

        display_df["Source"] = ""


    if "dst_ip" in display_df.columns:

        display_df["Destination"] = (
            display_df["dst_ip"].astype(str)
            + ":"
            + display_df["dst_port"].astype(str)
        )

    else:

        display_df["Destination"] = ""


    if "attack_probability" in display_df.columns:

        display_df["Attack Probability"] = (
            display_df["attack_probability"]
            .apply(format_probability)
        )

    else:

        display_df["Attack Probability"] = ""


    if "family_confidence" in display_df.columns:

        display_df["Confidence"] = (
            display_df["family_confidence"]
            .apply(format_probability)
        )

    else:

        display_df["Confidence"] = ""


    columns = [
        "Time",
        "Source",
        "Destination",
        "prediction",
        "Attack Probability",
        "attack_family",
        "Confidence",
        "severity",
    ]

    columns = [
        x
        for x in columns
        if x in display_df.columns
    ]


    final_df = display_df[columns].copy()


    final_df = final_df.rename(
        columns={
            "prediction": "Prediction",
            "attack_family": "Family",
            "severity": "Severity",
        }
    )


    st.dataframe(
        final_df,
        use_container_width=True,
        hide_index=True,
        height=450,
    )


st.divider()


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.markdown("## ⚙️ System Information")

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Attack Rate",
        f"{attack_rate:.2f}%",
    )

with info2:

    st.metric(
        "Flows Analyzed",
        f"{total_flows:,}",
    )

with info3:

    if total_flows > 0:

        benign_rate = (
            total_benign
            / total_flows
            * 100
        )

    else:

        benign_rate = 0

    st.metric(
        "Benign Rate",
        f"{benign_rate:.2f}%",
    )


# ============================================================
# ERROR DISPLAY
# ============================================================

last_error = safe_get(
    "last_error",
    None,
)

if last_error:

    st.error(
        f"Live monitoring error: {last_error}"
    )


processing_error = st.session_state.get(
    "last_processing_error",
    None,
)

if processing_error:

    st.error(
        f"Processing error: {processing_error}"
    )


# ============================================================
# AUTO REFRESH
# ============================================================

if running:

    time.sleep(1.5)

    st.rerun()