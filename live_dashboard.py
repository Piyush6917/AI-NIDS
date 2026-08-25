import time
import streamlit as st
from live_backend import LiveNIDS

st.set_page_config(page_title="AI-NIDS Live Monitor",page_icon="🛡️",layout="wide")

st.markdown("""
<style>
.title{font-size:42px;font-weight:800;margin-bottom:0}
.sub{color:#9ca3af;font-size:16px;margin:4px 0 24px}
.live{padding:12px 16px;border-radius:10px;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.35);color:#4ade80;font-weight:700}
.off{padding:12px 16px;border-radius:10px;background:rgba(239,68,68,.10);border:1px solid rgba(239,68,68,.30);color:#f87171;font-weight:700}
.attack{padding:20px;border-radius:12px;background:rgba(239,68,68,.13);border:1px solid rgba(239,68,68,.45)}
.benign{padding:20px;border-radius:12px;background:rgba(34,197,94,.11);border:1px solid rgba(34,197,94,.35)}
</style>
""",unsafe_allow_html=True)

if "nids" not in st.session_state:
    st.session_state.nids=LiveNIDS()
nids=st.session_state.nids

st.markdown('<div class="title">🛡️ AI-NIDS Live Monitor</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">Real Wi-Fi traffic capture → CICFlowMeter → 61 features → Transformer AI detection</div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Live Capture")
    interface=st.selectbox("Network Interface",["Wi-Fi","Ethernet"],index=0)
    st.divider()
    if not nids.is_running:
        if st.button("▶ Start Live Monitoring",type="primary",use_container_width=True):
            try:
                nids.start_capture(interface)
                st.rerun()
            except Exception as e:
                nids.last_error=str(e)
                st.error(str(e))
    else:
        if st.button("■ Stop Live Monitoring",use_container_width=True):
            nids.stop_capture()
            st.rerun()
    st.divider()
    st.caption("AI pipeline")
    st.code("Wi-Fi\n ↓\nCICFlowMeter\n ↓\n61 Features\n ↓\nBinary Transformer\n ↓\nMulti-class Transformer V2\n ↓\nPrediction",language="text")

if nids.is_running:
    try: nids.process_new_flows()
    except Exception as e: nids.last_error=repr(e)

if nids.is_running:
    st.markdown('<div class="live">🟢 LIVE CAPTURE ACTIVE</div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="off">🔴 CAPTURE STOPPED</div>',unsafe_allow_html=True)

st.write("")
a,b,c,d,e=st.columns(5)
a.metric("Device",str(nids.device).upper())
b.metric("Total Flows",nids.total_flows)
c.metric("Benign",nids.total_benign)
d.metric("Attacks",nids.total_attacks)
e.metric("High Severity",nids.high_severity)

st.divider()
st.subheader("🔎 Current Flow Detection")
latest=nids.predictions[-1] if nids.predictions else None

if not latest:
    st.info("Click Start Live Monitoring. The system will capture traffic from the selected adapter automatically.")
else:
    box="attack" if latest["prediction"]=="ATTACK" else "benign"
    label="🚨 ATTACK DETECTED" if latest["prediction"]=="ATTACK" else "🟢 BENIGN TRAFFIC"
    st.markdown(f'<div class="{box}"><b>{label}</b></div>',unsafe_allow_html=True)
    st.write("")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Prediction",latest["prediction"])
    c2.metric("Attack Probability",f'{latest["attack_probability"]*100:.2f}%')
    c3.metric("Attack Family",latest["attack_family"])
    c4.metric("Severity",latest["severity"])
    st.markdown(
        f'**Flow:** `{latest["src_ip"]}:{latest["src_port"]}` → `{latest["dst_ip"]}:{latest["dst_port"]}`  \n'
        f'**Protocol:** `{latest["protocol"]}` • **Family Confidence:** `{latest["family_confidence"]*100:.2f}%` • **Time:** `{latest["timestamp"]}`'
    )

st.divider()
st.subheader("📡 Recent Network Traffic")

if nids.predictions:
    rows=[]
    for r in reversed(nids.predictions[-30:]):
        rows.append({
            "Time":r["timestamp"],
            "Source":f'{r["src_ip"]}:{r["src_port"]}',
            "Destination":f'{r["dst_ip"]}:{r["dst_port"]}',
            "Prediction":r["prediction"],
            "Attack Probability":f'{r["attack_probability"]*100:.2f}%',
            "Family":r["attack_family"],
            "Confidence":f'{r["family_confidence"]*100:.2f}%',
            "Severity":r["severity"]
        })
    st.dataframe(rows,use_container_width=True,hide_index=True)
else:
    st.caption("Waiting for live flows...")

if nids.last_error:
    st.error("Live monitoring error: "+nids.last_error)

if nids.is_running:
    time.sleep(1.5)
    st.rerun()
