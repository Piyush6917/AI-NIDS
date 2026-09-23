import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import joblib
from pathlib import Path

BASE=Path(__file__).resolve().parent
F=BASE/"data/final/features"
M=BASE/"models"
CLASSES=["BENIGN","DoS_DDoS","PortScan","BruteForce","WebAttack","Bot","Infiltration"]

class NetworkTransformer(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.feature_embedding=nn.Linear(1,64)
        self.feature_position=nn.Parameter(torch.randn(1,61,64))
        layer=nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=256,dropout=.1,activation="gelu",batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,2)
        self.norm=nn.LayerNorm(64)
        self.classifier=nn.Sequential(nn.Linear(64,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,n))
    def forward(self,x):
        x=self.feature_embedding(x.unsqueeze(-1))+self.feature_position
        return self.classifier(self.norm(self.transformer(x).mean(1)))

@st.cache_resource
def load():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler=joblib.load(F/"standard_scaler.joblib")
    def load_model(path,n):
        m=NetworkTransformer(n)
        c=torch.load(path,map_location=device,weights_only=False)
        s=c.get("model_state_dict",c.get("state_dict",c))
        m.load_state_dict(s,strict=True)
        return m.to(device).eval()
    b=load_model(M/"transformer_binary_gpu.pth",2)
    mc=load_model(M/"transformer_multiclass_gpu_v2.pth",7)
    return device,scaler,b,mc,np.load(F/"X_test.npy"),np.load(F/"y_test_binary.npy"),np.load(F/"y_test_family.npy",allow_pickle=True).astype(str)

def infer(X,b,mc,device):
    with torch.inference_mode():
        t=torch.from_numpy(X.astype(np.float32)).to(device)
        bp=torch.softmax(b(t),1); mp=torch.softmax(mc(t),1)
    out=[]
    for i in range(len(X)):
        ap=float(bp[i,1]); j=int(torch.argmax(mp[i])); fam=CLASSES[j]; conf=float(mp[i,j])
        pred="ATTACK" if ap>=.80 else "BENIGN"
        if pred=="BENIGN": fam="BENIGN"
        sev="NONE" if pred=="BENIGN" else ("HIGH" if fam in ("DoS_DDoS","Infiltration") else "MEDIUM")
        out.append((pred,ap,fam,conf,sev))
    return out

st.set_page_config(page_title="AI-NIDS Simulation Lab",page_icon="🧪",layout="wide")
st.title("🧪 AI-NIDS — Local Simulation Lab")
st.caption("Controlled replay of held-out test flows through the real trained Transformer models.")

try:
    device,scaler,b,mc,X,yb,yf=load()
except Exception as e:
    st.error("Could not load the simulation files/models.")
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Simulation Control")
    st.success("Models ready • "+str(device).upper())
    scenario=st.selectbox("Traffic scenario",["BENIGN","DoS_DDoS","PortScan","BruteForce","WebAttack","Bot","Infiltration","MIXED"])
    count=st.slider("Number of flows",1,100,20)
    if st.button("▶ Run Simulation",type="primary",use_container_width=True):
        eligible=np.arange(len(X)) if scenario=="MIXED" else np.where(np.char.upper(yf)==scenario.upper())[0]
        chosen=np.random.default_rng().choice(eligible,size=count,replace=len(eligible)<count)
        pred=infer(X[chosen],b,mc,device)
        rows=[]
        for k,(pr,ap,pf,cf,se) in enumerate(pred):
            truth=yf[chosen[k]]
            rows.append({"Flow":k+1,"Ground Truth":truth,"Prediction":pr,"Predicted Family":pf,"Attack Probability":ap,"Family Confidence":cf,"Severity":se,"Binary Correct":pr==("ATTACK" if yb[chosen[k]] else "BENIGN"),"Family Correct":pf==truth})
        st.session_state.results=pd.DataFrame(rows)
    if st.button("↻ Clear Results",use_container_width=True):
        st.session_state.pop("results",None)
        st.rerun()

r=st.session_state.get("results",pd.DataFrame())
if r.empty:
    st.info("Select a scenario and click ▶ Run Simulation.")
    st.markdown("""### How it works
This is not a fake attack counter. The simulator selects real held-out samples from `X_test.npy` and sends their 61 prepared features through the existing Binary and Multi-class Transformer models. Ground truth is shown only after inference for validation.""")
else:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Simulated Flows",len(r))
    c2.metric("Predicted Attacks",int((r["Prediction"]=="ATTACK").sum()))
    c3.metric("Predicted Benign",int((r["Prediction"]=="BENIGN").sum()))
    c4.metric("Binary Accuracy",f"{r['Binary Correct'].mean()*100:.1f}%")
    l,rr=st.columns(2)
    with l:
        st.subheader("AI Prediction Distribution")
        st.bar_chart(r["Predicted Family"].value_counts())
    with rr:
        st.subheader("Ground Truth Distribution")
        st.bar_chart(r["Ground Truth"].value_counts())
    show=r.copy()
    show["Attack Probability"]=(show["Attack Probability"]*100).round(2).astype(str)+"%"
    show["Family Confidence"]=(show["Family Confidence"]*100).round(2).astype(str)+"%"
    st.subheader("📋 Simulation Results")
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.subheader("🚨 Detected Attacks")
    attacks=r[r["Prediction"]=="ATTACK"]
    if attacks.empty:
        st.success("No attacks were predicted in this run.")
    else:
        for _,x in attacks.iterrows():
            st.error(f"Flow {int(x['Flow'])} | Predicted: {x['Predicted Family']} | Ground truth: {x['Ground Truth']} | Probability: {x['Attack Probability']*100:.2f}% | Severity: {x['Severity']}")
    st.info("Ground truth is displayed only after inference and is never supplied to the model.")
