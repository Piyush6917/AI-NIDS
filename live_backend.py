import os, time, subprocess
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
LIVE_DIR = BASE / "data" / "live"
LIVE_CSV = LIVE_DIR / "live_flows.csv"
OUT_CSV = LIVE_DIR / "live_predictions.csv"
SCALER_FILE = BASE / "data" / "final" / "features" / "standard_scaler.joblib"
BINARY_FILE = BASE / "models" / "transformer_binary_gpu.pth"
MULTI_FILE = BASE / "models" / "transformer_multiclass_gpu_v2.pth"
CICFLOWMETER = Path(r"C:\Users\piyus\anaconda3\envs\AI-NIDS\Scripts\cicflowmeter.exe")
BINARY_THRESHOLD = 0.80
CLASSES = ["BENIGN","DoS_DDoS","PortScan","BruteForce","WebAttack","Bot","Infiltration"]

FEATURE_MAP = {
"Destination Port":"dst_port","Flow Duration":"flow_duration",
"Total Fwd Packets":"tot_fwd_pkts","Total Backward Packets":"tot_bwd_pkts",
"Total Length of Fwd Packets":"totlen_fwd_pkts","Total Length of Bwd Packets":"totlen_bwd_pkts",
"Fwd Packet Length Max":"fwd_pkt_len_max","Fwd Packet Length Min":"fwd_pkt_len_min",
"Fwd Packet Length Mean":"fwd_pkt_len_mean","Fwd Packet Length Std":"fwd_pkt_len_std",
"Bwd Packet Length Max":"bwd_pkt_len_max","Bwd Packet Length Min":"bwd_pkt_len_min",
"Bwd Packet Length Mean":"bwd_pkt_len_mean","Bwd Packet Length Std":"bwd_pkt_len_std",
"Flow Bytes/s":"flow_byts_s","Flow Packets/s":"flow_pkts_s",
"Flow IAT Mean":"flow_iat_mean","Flow IAT Std":"flow_iat_std",
"Flow IAT Max":"flow_iat_max","Flow IAT Min":"flow_iat_min",
"Fwd IAT Total":"fwd_iat_tot","Fwd IAT Mean":"fwd_iat_mean",
"Fwd IAT Std":"fwd_iat_std","Fwd IAT Max":"fwd_iat_max","Fwd IAT Min":"fwd_iat_min",
"Bwd IAT Total":"bwd_iat_tot","Bwd IAT Mean":"bwd_iat_mean",
"Bwd IAT Std":"bwd_iat_std","Bwd IAT Max":"bwd_iat_max","Bwd IAT Min":"bwd_iat_min",
"Fwd PSH Flags":"fwd_psh_flags","Fwd Header Length":"fwd_header_len",
"Bwd Header Length":"bwd_header_len","Fwd Packets/s":"fwd_pkts_s","Bwd Packets/s":"bwd_pkts_s",
"MinPacket Length":"pkt_len_min","Max Packet Length":"pkt_len_max",
"Packet Length Mean":"pkt_len_mean","Packet Length Std":"pkt_len_std",
"Packet Length Variance":"pkt_len_var","FIN Flag Count":"fin_flag_cnt",
"SYN Flag Count":"syn_flag_cnt","RST Flag Count":"rst_flag_cnt","PSH Flag Count":"psh_flag_cnt",
"ACK Flag Count":"ack_flag_cnt","URG Flag Count":"urg_flag_cnt","ECE Flag Count":"ece_flag_cnt",
"Down/Up Ratio":"down_up_ratio","Average Packet Size":"pkt_size_avg",
"Init_Win_bytes_forward":"init_fwd_win_byts","Init_Win_bytes_backward":"init_bwd_win_byts",
"act_data_pkt_fwd":"fwd_act_data_pkts","min_seg_size_forward":"fwd_seg_size_min",
"Active Mean":"active_mean","Active Std":"active_std","Active Max":"active_max","Active Min":"active_min",
"Idle Mean":"idle_mean","Idle Std":"idle_std","Idle Max":"idle_max","Idle Min":"idle_min"
}

class NetworkTransformer(nn.Module):
    def __init__(self,num_classes,num_features=61,d_model=64,nhead=4,num_layers=2,dim_feedforward=256,dropout=0.1):
        super().__init__()
        self.feature_embedding=nn.Linear(1,d_model)
        self.feature_position=nn.Parameter(torch.randn(1,num_features,d_model))
        layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,activation="gelu",batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=num_layers)
        self.norm=nn.LayerNorm(d_model)
        self.classifier=nn.Sequential(nn.Linear(d_model,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,num_classes))
    def forward(self,x):
        x=self.feature_embedding(x.unsqueeze(-1))
        x=x+self.feature_position
        x=self.transformer(x)
        x=self.norm(x.mean(dim=1))
        return self.classifier(x)

def load_model(path,classes,device):
    model=NetworkTransformer(classes)
    ckpt=torch.load(path,map_location=device,weights_only=False)
    state=ckpt.get("model_state_dict",ckpt.get("state_dict",ckpt))
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()

class LiveNIDS:
    def __init__(self):
        self.capture_process=None
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler=None; self.binary_model=None; self.multi_model=None
        self.processed_rows=0; self.predictions=[]; self.last_error=""
        self.total_flows=0; self.total_benign=0; self.total_attacks=0; self.high_severity=0

    @property
    def is_running(self):
        return self.capture_process is not None and self.capture_process.poll() is None

    def load_models(self):
        if self.scaler is None: self.scaler=joblib.load(SCALER_FILE)
        if self.binary_model is None: self.binary_model=load_model(BINARY_FILE,2,self.device)
        if self.multi_model is None: self.multi_model=load_model(MULTI_FILE,7,self.device)

    def reset_files(self):
        LIVE_DIR.mkdir(parents=True,exist_ok=True)
        for f in (LIVE_CSV,OUT_CSV):
            if f.exists():
                try: f.unlink()
                except PermissionError: pass
        self.processed_rows=0; self.predictions=[]
        self.total_flows=self.total_benign=self.total_attacks=self.high_severity=0

    def start_capture(self,interface):
        self.last_error=""
        if self.is_running: return
        self.load_models(); self.reset_files()
        if not CICFLOWMETER.exists():
            raise FileNotFoundError(f"CICFlowMeter not found: {CICFLOWMETER}")
        flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        self.capture_process=subprocess.Popen([str(CICFLOWMETER),"-i",interface,"-c",str(LIVE_CSV)],
                                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags)
        time.sleep(1)
        if self.capture_process.poll() is not None:
            self.capture_process=None
            raise RuntimeError("CICFlowMeter stopped immediately. Check Npcap, permissions and interface name.")

    def stop_capture(self):
        p=self.capture_process
        self.capture_process=None
        if p is not None:
            try:
                if p.poll() is None: p.terminate(); p.wait(timeout=3)
            except Exception:
                try: p.kill()
                except Exception: pass
    def make_features(self, df):
        expected = list(self.scaler.feature_names_in_)
    
        # Mapping from trained scaler feature names
        # to CICFlowMeter live CSV column names.
        FEATURE_MAP = {
            "Destination Port": "dst_port",
            "Flow Duration": "flow_duration",
    
            "Total Fwd Packets": "tot_fwd_pkts",
            "Total Backward Packets": "tot_bwd_pkts",
    
            "Total Length of Fwd Packets": "totlen_fwd_pkts",
            "Total Length of Bwd Packets": "totlen_bwd_pkts",
    
            "Fwd Packet Length Max": "fwd_pkt_len_max",
            "Fwd Packet Length Min": "fwd_pkt_len_min",
            "Fwd Packet Length Mean": "fwd_pkt_len_mean",
            "Fwd Packet Length Std": "fwd_pkt_len_std",
    
            "Bwd Packet Length Max": "bwd_pkt_len_max",
            "Bwd Packet Length Min": "bwd_pkt_len_min",
            "Bwd Packet Length Mean": "bwd_pkt_len_mean",
            "Bwd Packet Length Std": "bwd_pkt_len_std",
    
            "Flow Bytes/s": "flow_byts_s",
            "Flow Packets/s": "flow_pkts_s",
    
            "Flow IAT Mean": "flow_iat_mean",
            "Flow IAT Std": "flow_iat_std",
            "Flow IAT Max": "flow_iat_max",
            "Flow IAT Min": "flow_iat_min",
    
            "Fwd IAT Total": "fwd_iat_tot",
            "Fwd IAT Mean": "fwd_iat_mean",
            "Fwd IAT Std": "fwd_iat_std",
            "Fwd IAT Max": "fwd_iat_max",
            "Fwd IAT Min": "fwd_iat_min",
    
            "Bwd IAT Total": "bwd_iat_tot",
            "Bwd IAT Mean": "bwd_iat_mean",
            "Bwd IAT Std": "bwd_iat_std",
            "Bwd IAT Max": "bwd_iat_max",
            "Bwd IAT Min": "bwd_iat_min",
    
            "Fwd PSH Flags": "fwd_psh_flags",
            "Fwd Header Length": "fwd_header_len",
            "Bwd Header Length": "bwd_header_len",
    
            "Fwd Packets/s": "fwd_pkts_s",
            "Bwd Packets/s": "bwd_pkts_s",
    
            # IMPORTANT:
            # CICFlowMeter uses pkt_len_min.
            # Training scaler may contain either spelling.
            "MinPacket Length": "pkt_len_min",
            "Min Packet Length": "pkt_len_min",
    
            "Max Packet Length": "pkt_len_max",
    
            "Packet Length Mean": "pkt_len_mean",
            "Packet Length Std": "pkt_len_std",
            "Packet Length Variance": "pkt_len_var",
    
            "FIN Flag Count": "fin_flag_cnt",
            "SYN Flag Count": "syn_flag_cnt",
            "RST Flag Count": "rst_flag_cnt",
            "PSH Flag Count": "psh_flag_cnt",
            "ACK Flag Count": "ack_flag_cnt",
            "URG Flag Count": "urg_flag_cnt",
            "ECE Flag Count": "ece_flag_cnt",
    
            "Down/Up Ratio": "down_up_ratio",
            "Average Packet Size": "pkt_size_avg",
    
            "Init_Win_bytes_forward": "init_fwd_win_byts",
            "Init_Win_bytes_backward": "init_bwd_win_byts",
    
            "act_data_pkt_fwd": "fwd_act_data_pkts",
            "min_seg_size_forward": "fwd_seg_size_min",
    
            "Active Mean": "active_mean",
            "Active Std": "active_std",
            "Active Max": "active_max",
            "Active Min": "active_min",
    
            "Idle Mean": "idle_mean",
            "Idle Std": "idle_std",
            "Idle Max": "idle_max",
            "Idle Min": "idle_min"
        }
    
        resolved = []
        missing = []
    
        for feature in expected:
    
            if feature not in FEATURE_MAP:
                missing.append(
                    f"No mapping for scaler feature: {feature}"
                )
                continue
    
            live_column = FEATURE_MAP[feature]
    
            if live_column not in df.columns:
                missing.append(
                    f"{feature} -> {live_column}"
                )
                continue
    
            resolved.append(live_column)
    
        if missing:
            raise RuntimeError(
                "Missing/unmapped CICFlowMeter columns:\n"
                + "\n".join(missing)
            )
    
        # Keep EXACTLY the same order as training.
        x = df[resolved].copy()
    
        # Convert everything to numeric.
        x = x.apply(
            pd.to_numeric,
            errors="coerce"
        )
    
        # Replace invalid values.
        x = x.replace(
            [np.inf, -np.inf],
            np.nan
        )
    
        x = x.fillna(0)
    
        # IMPORTANT:
        # Send numpy array to scaler.
        # This avoids sklearn checking the live CSV's
        # lowercase CICFlowMeter column names against
        # the training feature names.
        x = self.scaler.transform(
            x.to_numpy(dtype=np.float64)
        )
    
        return x.astype(np.float32)
    
    def predict(self,x):
        t=torch.from_numpy(x).to(self.device)
        with torch.inference_mode():
            bp=torch.softmax(self.binary_model(t),1)
            mp=torch.softmax(self.multi_model(t),1)
        out=[]
        for i in range(len(x)):
            attack=float(bp[i,1]); idx=int(torch.argmax(mp[i]))
            family=CLASSES[idx]; conf=float(mp[i,idx])
            if attack>=BINARY_THRESHOLD:
                pred="ATTACK"
                if family=="BENIGN":
                    for j in torch.argsort(mp[i],descending=True):
                        j=int(j)
                        if CLASSES[j]!="BENIGN":
                            family=CLASSES[j]; conf=float(mp[i,j]); break
            else: pred="BENIGN"; family="BENIGN"
            severity="NONE" if pred=="BENIGN" else ("HIGH" if family in ["DoS_DDoS","Infiltration"] else "MEDIUM")
            out.append((pred,attack,family,conf,severity))
        return out

    def process_new_flows(self):
        if not LIVE_CSV.exists(): return
        try: df=pd.read_csv(LIVE_CSV)
        except (pd.errors.EmptyDataError,pd.errors.ParserError): return
        if len(df)<=self.processed_rows: return
        new=df.iloc[self.processed_rows:].copy()
        results=self.predict(self.make_features(new)); rows=[]
        for raw,res in zip(new.to_dict("records"),results):
            pred,attack,family,conf,severity=res
            item={"timestamp":raw.get("timestamp",""),"src_ip":raw.get("src_ip",""),"src_port":raw.get("src_port",""),
                  "dst_ip":raw.get("dst_ip",""),"dst_port":raw.get("dst_port",""),"protocol":raw.get("protocol",""),
                  "prediction":pred,"attack_probability":attack,"attack_family":family,"family_confidence":conf,"severity":severity}
            rows.append(item); self.predictions.append(item); self.total_flows+=1
            if pred=="ATTACK": self.total_attacks+=1
            else: self.total_benign+=1
            if severity=="HIGH": self.high_severity+=1
        self.processed_rows=len(df); self.predictions=self.predictions[-200:]
        if rows: pd.DataFrame(rows).to_csv(OUT_CSV,mode="a",header=not OUT_CSV.exists(),index=False)

    def shutdown(self): self.stop_capture()