import time
from datetime import datetime

import pandas as pd
import streamlit as st

from live_backend import LiveNIDS


st.set_page_config(
    page_title="AI-NIDS Security Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "nids" not in st.session_state:
    st.session_state.nids = LiveNIDS()

if "started_at" not in st.session_state:
    st.session_state.started_at = None

if "selected_interface" not in st.session_state:
    st.session_state.selected_interface = "Wi-Fi"

if "capture_mode" not in st.session_state:
    st.session_state.capture_mode = "LAB"

nids = st.session_state.nids


def running():
    value = nids.is_running
    return bool(value() if callable(value) else value)


def pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "0.00%"


def elapsed():
    if st.session_state.started_at is None:
        return "00:00:00"

    seconds = int(
        (datetime.now() - st.session_state.started_at).total_seconds()
    )

    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    return f"{h:02d}:{m:02d}:{s:02d}"


# -----------------------------
# CSS
# -----------------------------

st.markdown(
    """
    <style>
    .stApp { background:#0b0f14; }
    [data-testid="stSidebar"] { background:#0d1219; }

    .hero {
        background:#111821;
        border:1px solid #26303b;
        border-radius:16px;
        padding:25px;
        margin-bottom:18px;
    }

    .hero h1 { margin:0; font-size:38px; }
    .muted { color:#8492a2; }

    .status {
        padding:13px 18px;
        border-radius:12px;
        font-weight:800;
        margin-bottom:18px;
    }

    .online {
        background:#063b27;
        color:#3fb950;
        border:1px solid #238636;
    }

    .offline {
        background:#3b2b0d;
        color:#d29922;
        border:1px solid #9e6a03;
    }

    .attack {
        background:#351416;
        border:1px solid #8b2d31;
        border-radius:14px;
        padding:20px;
    }

    .benign {
        background:#0b2d20;
        border:1px solid #238636;
        border-radius:14px;
        padding:20px;
    }

    .attack-title {
        font-size:25px;
        font-weight:850;
    }

    .health {
        background:#111821;
        border:1px solid #26303b;
        border-radius:14px;
        padding:18px;
    }

    .lab-box {
        background:#111821;
        border:1px solid #26303b;
        border-radius:14px;
        padding:18px;
        margin-bottom:12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.markdown("# 🛡️ AI-NIDS")
    st.caption("Security Operations Center")

    st.divider()

    st.markdown("### 🎛️ Capture Control")

    mode = st.radio(
        "Capture mode",
        ["LAB", "REAL"],
        index=0 if st.session_state.capture_mode == "LAB" else 1,
        help="LAB uses the isolated Docker test network. REAL captures Wi-Fi/Ethernet.",
    )

    st.session_state.capture_mode = mode

    if mode == "REAL":
        interface = st.selectbox(
            "Network interface",
            ["Wi-Fi", "Ethernet"],
            index=(
                0
                if st.session_state.selected_interface == "Wi-Fi"
                else 1
            ),
        )
        st.session_state.selected_interface = interface

    else:
        interface = "Docker Lab / nids-client"

    if running():
        st.success(f"Capture process is running ({mode}).")

        if st.button(
            "■ Stop Live Monitoring",
            use_container_width=True,
        ):
            nids.stop_capture()
            st.session_state.started_at = None
            st.rerun()

    else:
        st.warning("Capture is stopped.")

        if st.button(
            "▶ Start Live Monitoring",
            type="primary",
            use_container_width=True,
        ):
            try:
                if mode == "LAB":
                    nids.start_lab_capture()
                else:
                    nids.start_real_capture(interface)

                st.session_state.started_at = datetime.now()
                st.rerun()

            except Exception as exc:
                st.error(
                    "Unable to start monitoring:\n"
                    + str(exc)
                )

    st.divider()

    st.markdown("### ⚡ Runtime")
    st.metric("Interface", interface)
    st.metric("Mode", mode)
    st.metric("Inference", str(nids.device).upper())
    st.metric("Capture Time", elapsed())

    st.divider()

    st.markdown("### 🧠 AI Pipeline")

    if mode == "LAB":
        st.code(
            """Docker Lab
  ↓
nids-client
  ↓
tcpdump PCAP
  ↓
CICFlowMeter
  ↓
61 Features
  ↓
StandardScaler
  ↓
Binary Transformer
  ↓
Multi-class Transformer
  ↓
Prediction""",
            language="text",
        )
    else:
        st.code(
            """Wi-Fi / Ethernet
  ↓
CICFlowMeter
  ↓
61 Features
  ↓
StandardScaler
  ↓
Binary Transformer
  ↓
Multi-class Transformer
  ↓
Prediction""",
            language="text",
        )


# -----------------------------
# Backend processing
# -----------------------------

if running():
    nids.process_new_flows()


df = pd.DataFrame(nids.predictions)

total = nids.total_flows
benign = nids.total_benign
attacks = nids.total_attacks
high = nids.high_severity

attack_rate = attacks / total * 100 if total else 0


# -----------------------------
# Header
# -----------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ AI-NIDS Security Monitor</h1>
        <div class="muted">
            Real-time network-flow analysis powered by Transformer AI
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if running():
    status_text = (
        "🟢 LAB CAPTURE ACTIVE — Isolated traffic is being monitored automatically"
        if nids.capture_mode == "LAB"
        else "🟢 REAL CAPTURE ACTIVE — Traffic is being captured automatically"
    )

    st.markdown(
        f'<div class="status online">{status_text}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status offline">🟡 CAPTURE STOPPED</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# KPIs
# -----------------------------

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Flows", f"{total:,}")
c2.metric("Benign", f"{benign:,}")
c3.metric("Attacks", f"{attacks:,}")
c4.metric("High Severity", f"{high:,}")
c5.metric("Attack Rate", f"{attack_rate:.2f}%")

st.divider()


# -----------------------------
# LAB CONTROL PANEL
# -----------------------------

if mode == "LAB":
    st.subheader("🧪 Controlled Lab Traffic")

    st.markdown(
        """
        <div class="lab-box">
        <b>Isolated test environment</b><br>
        <span class="muted">
        Traffic is generated only between the Docker lab client and the
        local OWASP Juice Shop target. No external target is used.
        </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not running():
        st.info("Start LAB monitoring first, then run a traffic scenario.")

    a1, a2, a3, a4 = st.columns(4)

    with a1:
        if st.button(
            "🟢 Normal Traffic",
            use_container_width=True,
            disabled=not running(),
        ):
            try:
                nids.run_lab_action("normal")
                st.success("Normal browsing traffic started.")
            except Exception as exc:
                st.error(str(exc))

    with a2:
        if st.button(
            "🔴 Port Scan Test",
            use_container_width=True,
            disabled=not running(),
        ):
            try:
                nids.run_lab_action("portscan")
                st.warning("Controlled port-scan test started.")
            except Exception as exc:
                st.error(str(exc))

    with a3:
        if st.button(
            "🟠 Web Security Test",
            use_container_width=True,
            disabled=not running(),
        ):
            try:
                nids.run_lab_action("webtest")
                st.warning("Local web-security test started.")
            except Exception as exc:
                st.error(str(exc))

    with a4:
        if st.button(
            "🔴 Stress Test (8s)",
            use_container_width=True,
            disabled=not running(),
        ):
            try:
                nids.run_lab_action("stress")
                st.warning("Bounded local stress test started.")
            except Exception as exc:
                st.error(str(exc))

    if nids.lab_action_running:
        st.info(
            f"Lab scenario running: {getattr(nids, 'lab_action_name', 'test')}"
        )
        if st.button("■ Stop Current Lab Test"):
            nids.stop_lab_action()
            st.rerun()

    if nids.last_lab_message:
        st.caption("Lab: " + nids.last_lab_message)

    st.divider()


# -----------------------------
# Current Detection
# -----------------------------

st.subheader("🔎 Current Flow Detection")

if df.empty:
    st.info(
        "Waiting for network flows. "
        "In LAB mode, start monitoring and use the traffic buttons above."
    )
else:
    latest = df.iloc[-1]

    prediction = str(latest.get("prediction", "UNKNOWN"))
    probability = float(latest.get("attack_probability", 0))
    family = str(latest.get("attack_family", "UNKNOWN"))
    confidence = float(latest.get("family_confidence", 0))
    severity = str(latest.get("severity", "NONE"))

    src = f"{latest.get('src_ip', '')}:{latest.get('src_port', '')}"
    dst = f"{latest.get('dst_ip', '')}:{latest.get('dst_port', '')}"

    css = "attack" if prediction == "ATTACK" else "benign"
    title = (
        "🚨 ATTACK DETECTED"
        if prediction == "ATTACK"
        else "🟢 BENIGN TRAFFIC"
    )

    st.markdown(
        f"""
        <div class="{css}">
            <div class="attack-title">{title}</div>
            <br>
            <b>Flow:</b> {src} → {dst}<br>
            <b>Protocol:</b> {latest.get('protocol', '')}<br>
            <b>Attack probability:</b> {pct(probability)}<br>
            <b>Attack family:</b> {family}<br>
            <b>Family confidence:</b> {pct(confidence)}<br>
            <b>Severity:</b> {severity}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# -----------------------------
# Charts
# -----------------------------

st.subheader("📊 Live Traffic Analytics")

if df.empty:
    st.info("No traffic data yet.")
else:
    left, right = st.columns(2)

    with left:
        st.markdown("**Flow count over session**")
        activity = pd.DataFrame(
            {
                "Flows": range(1, len(df) + 1),
                "Total": 1,
            }
        )
        st.line_chart(activity.set_index("Flows"))

    with right:
        st.markdown("**Benign vs Attack**")
        st.bar_chart(
            pd.DataFrame(
                {"Count": [benign, attacks]},
                index=["Benign", "Attack"],
            )
        )


st.divider()


# -----------------------------
# Attack intelligence
# -----------------------------

st.subheader("🧠 Network & Attack Intelligence")

left, right = st.columns(2)

with left:
    st.markdown("**Attack families**")
    if df.empty:
        st.info("No predictions yet.")
    else:
        st.bar_chart(df["attack_family"].value_counts())

with right:
    st.markdown("**Protocols**")
    if df.empty:
        st.info("No traffic yet.")
    else:
        st.bar_chart(df["protocol"].astype(str).value_counts())


st.divider()


# -----------------------------
# Alerts
# -----------------------------

st.subheader("🚨 Security Alerts")

if df.empty:
    st.info("No alerts.")
else:
    alerts = df[
        df["prediction"].astype(str) == "ATTACK"
    ].tail(10)

    if alerts.empty:
        st.success("No attacks detected in this session.")
    else:
        for _, row in alerts.iloc[::-1].iterrows():
            st.error(
                f"🚨 {row.get('attack_family', 'UNKNOWN')} | "
                f"{row.get('severity', 'MEDIUM')} | "
                f"{row.get('src_ip', '')}:{row.get('src_port', '')} → "
                f"{row.get('dst_ip', '')}:{row.get('dst_port', '')} | "
                f"Probability: {pct(row.get('attack_probability', 0))}"
            )


st.divider()


# -----------------------------
# Recent traffic
# -----------------------------

st.subheader("📋 Recent Network Traffic")

if df.empty:
    st.info("Waiting for network flows.")
else:
    table = df.tail(100).copy()

    table["Source"] = (
        table["src_ip"].astype(str)
        + ":"
        + table["src_port"].astype(str)
    )

    table["Destination"] = (
        table["dst_ip"].astype(str)
        + ":"
        + table["dst_port"].astype(str)
    )

    table["Attack Probability"] = table["attack_probability"].apply(pct)
    table["Confidence"] = table["family_confidence"].apply(pct)

    table = table[
        [
            "timestamp",
            "Source",
            "Destination",
            "protocol",
            "prediction",
            "Attack Probability",
            "attack_family",
            "Confidence",
            "severity",
        ]
    ].rename(
        columns={
            "timestamp": "Time",
            "protocol": "Protocol",
            "prediction": "Prediction",
            "attack_family": "Family",
            "severity": "Severity",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=450,
    )


st.divider()


# -----------------------------
# System Health
# -----------------------------

st.subheader("⚙️ System Health")

model_ok = (
    nids.binary_model is not None
    and nids.multi_model is not None
)

scaler_ok = nids.scaler is not None

h1, h2, h3 = st.columns(3)

with h1:
    st.markdown(
        f"""
        <div class="health">
            <b>📡 Packet Capture</b><br><br>
            {"🟢 ONLINE" if running() else "🟡 STOPPED"}<br>
            <span class="muted">{interface}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h2:
    st.markdown(
        f"""
        <div class="health">
            <b>🤖 AI Models</b><br><br>
            {"🟢 READY" if model_ok else "🔴 ERROR"}<br>
            <span class="muted">
                Binary + Multi-class Transformer
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h3:
    st.markdown(
        f"""
        <div class="health">
            <b>⚙️ Preprocessing</b><br><br>
            {"🟢 READY" if scaler_ok else "🔴 ERROR"}<br>
            <span class="muted">
                61-feature StandardScaler
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if nids.last_error:
    st.error("Backend error: " + nids.last_error)

st.caption(
    "AI-NIDS • Real-time network intrusion detection • "
    "Use only on networks you are authorized to monitor."
)


# -----------------------------
# Automatic refresh
# -----------------------------

if running():
    time.sleep(2)
    st.rerun()
