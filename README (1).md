# 🛡️ AI-NIDS — AI-Based Network Intrusion Detection System

AI-NIDS is a real-time, AI-powered Network Intrusion Detection System (NIDS). It captures network traffic from a selected local network interface, converts traffic into network-flow features, preprocesses those features, and uses trained Transformer models to classify the traffic.

The project includes a Streamlit dashboard for live monitoring and visualization.

---

## 1. What does AI-NIDS do?

The system follows this pipeline:

```text
Wi-Fi / Ethernet
       ↓
Live Packet Capture
       ↓
CICFlowMeter
       ↓
Network Flow
       ↓
61 Selected Features
       ↓
StandardScaler
       ↓
Binary Transformer
       ↓
BENIGN / ATTACK
       ↓
Multi-class Transformer V2
       ↓
Attack Family
       ↓
Severity
       ↓
Streamlit Live Dashboard
```

The goal is to combine **networking + cybersecurity + machine learning + real-time visualization** in one system.

---

## 2. Main Features

### 🔴 Real-time traffic capture
The system can capture traffic from a selected adapter such as Wi-Fi or Ethernet.

### 📊 Live dashboard
The dashboard displays:

- Capture status
- Inference device
- Total flows
- Benign flows
- Detected attacks
- High-severity events
- Current flow prediction
- Attack probability
- Attack family
- Family confidence
- Severity
- Recent network traffic

### 🧠 Two-stage AI detection

**Binary Transformer**

```text
Network Flow → BENIGN / ATTACK
```

**Multi-class Transformer V2**

```text
Network Flow → Attack Family
```

Configured classes:

```text
BENIGN
DoS_DDoS
PortScan
BruteForce
WebAttack
Bot
Infiltration
```

---

## 3. Important Project Files

### `live_dashboard.py`

The Streamlit frontend.

Responsible for:

- UI
- Network-interface selection
- Start/stop monitoring
- Live statistics
- Current detection
- Recent traffic table
- AI pipeline display

Start the project using this file.

### `live_backend.py`

The main backend/control layer.

Responsible for:

- Loading models
- Loading StandardScaler
- Starting/stopping capture
- Reading live flow data
- Preparing features
- Running inference
- Maintaining statistics
- Writing predictions

### `live_detector.py`

Standalone live detector/testing program. It can be used to test the inference pipeline without relying entirely on the Streamlit frontend.

### `data/live/live_flows.csv`

Live flow records generated from network traffic.

Typical fields include:

- Source IP
- Destination IP
- Source port
- Destination port
- Protocol
- Timestamp
- Flow duration
- Packet statistics
- Byte statistics
- IAT statistics
- TCP flags
- Window statistics
- Active/idle statistics

### `data/live/live_predictions.csv`

AI-NIDS prediction output.

Typical fields:

```text
timestamp
src_ip
src_port
dst_ip
dst_port
protocol
prediction
attack_probability
attack_family
family_confidence
severity
```

---

## 4. Model Files

The project contains Transformer checkpoints such as:

```text
models/
├── transformer_binary_cpu.pth
├── transformer_binary_gpu.pth
├── transformer_multiclass_cpu.pth
└── transformer_multiclass_gpu_v2.pth
```

The live backend is configured to use the GPU models when CUDA is available.

### Binary model

```text
61 features
     ↓
Binary Transformer
     ↓
BENIGN / ATTACK
```

### Multi-class model

```text
61 features
     ↓
Multi-class Transformer V2
     ↓
Attack Family
```

---

## 5. The 61 Features

The trained scaler expects exactly 61 features:

```text
Destination Port
Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Fwd Packet Length Max
Fwd Packet Length Min
Fwd Packet Length Mean
Fwd Packet Length Std
Bwd Packet Length Max
Bwd Packet Length Min
Bwd Packet Length Mean
Bwd Packet Length Std
Flow Bytes/s
Flow Packets/s
Flow IAT Mean
Flow IAT Std
Flow IAT Max
Flow IAT Min
Fwd IAT Total
Fwd IAT Mean
Fwd IAT Std
Fwd IAT Max
Fwd IAT Min
Bwd IAT Total
Bwd IAT Mean
Bwd IAT Std
Bwd IAT Max
Bwd IAT Min
Fwd PSH Flags
Fwd Header Length
Bwd Header Length
Fwd Packets/s
Bwd Packets/s
MinPacket Length
Max Packet Length
Packet Length Mean
Packet Length Std
Packet Length Variance
FIN Flag Count
SYN Flag Count
RST Flag Count
PSH Flag Count
ACK Flag Count
URG Flag Count
ECE Flag Count
Down/Up Ratio
Average Packet Size
Init_Win_bytes_forward
Init_Win_bytes_backward
act_data_pkt_fwd
min_seg_size_forward
Active Mean
Active Std
Active Max
Active Min
Idle Mean
Idle Std
Idle Max
Idle Min
```

### Important: feature order

The feature names and order must match the training pipeline.

The live CICFlowMeter names are therefore mapped to the names expected by the trained scaler/model.

For example:

```text
pkt_len_min
      ↓
MinPacket Length
```

and:

```text
ack_flag_cnt
      ↓
ACK Flag Count
```

A feature mismatch can cause:

```text
Feature names should match those that were passed during fit.
```

---

## 6. StandardScaler

Before inference, the 61 features are standardized using:

```text
data/final/features/standard_scaler.joblib
```

Pipeline:

```text
Live flow
   ↓
Feature mapping
   ↓
61 training features
   ↓
StandardScaler
   ↓
float32 tensor
   ↓
Transformer
```

### Scikit-learn version

The scaler was previously saved with scikit-learn 1.7.2. The project therefore pins:

```text
scikit-learn==1.7.2
```

Using another version can produce an `InconsistentVersionWarning`.

---

## 7. GPU Support

The project supports NVIDIA CUDA through PyTorch.

Example:

```text
Device: cuda
GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

The backend chooses:

```python
cuda
```

when available and otherwise falls back to:

```python
cpu
```

For NVIDIA GPU support, install the PyTorch build appropriate for your CUDA environment from:

https://pytorch.org/get-started/locally/

Verify:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 8. Windows Packet Capture Requirements

Python packages alone are not enough for Windows live packet capture.

You may need **Npcap** for packet capture support.

Install Npcap and enable WinPcap-compatible support if required by your capture setup.

After installation, restart VS Code/PowerShell if necessary.

You should only capture traffic on networks you are authorized to monitor.

---

## 9. Installation

Open PowerShell in the project folder:

```powershell
cd "C:\Users\piyus\OneDrive\Desktop\AI-NIDS"
```

Activate the existing environment:

```powershell
conda activate AI-NIDS
```

Or create a new environment:

```powershell
conda create -n AI-NIDS python=3.12
conda activate AI-NIDS
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install/verify PyTorch with the appropriate CUDA build if GPU inference is required.

---

## 10. Verify Important Files

Run:

```powershell
Get-Item "live_dashboard.py"
Get-Item "live_backend.py"
Get-Item "data\final\features\standard_scaler.joblib"
Get-Item "models\transformer_binary_gpu.pth"
Get-Item "models\transformer_multiclass_gpu_v2.pth"
```

Check live-data files:

```powershell
Get-ChildItem "data\live"
```

---

## 11. Start the Dashboard

Use:

```powershell
python -m streamlit run live_dashboard.py
```

If necessary, use the environment executable explicitly:

```powershell
C:\Users\piyus\anaconda3\envs\AI-NIDS\python.exe -m streamlit run live_dashboard.py
```

Then open:

```text
http://localhost:8501
```

---

## 12. Starting Live Monitoring

Inside the dashboard:

```text
Network Interface
       ↓
Select Wi-Fi / Ethernet
       ↓
Start Live Monitoring
```

The backend should pass the selected interface into the capture function.

The intended flow is:

```text
Selected Interface
       ↓
Packet Capture
       ↓
CICFlowMeter
       ↓
live_flows.csv
       ↓
Feature Mapping
       ↓
61 Features
       ↓
StandardScaler
       ↓
Binary Transformer
       ↓
Multi-class Transformer
       ↓
Dashboard
```

---

## 13. Generate Normal Test Traffic

Open another PowerShell window:

```powershell
ping google.com -n 20
```

Then:

```powershell
curl.exe https://example.com
```

You can also browse normally.

The dashboard should begin showing flows such as:

```text
Total Flows: 1053
Benign:      1053
Attacks:     0
```

The exact numbers depend on the traffic generated by the machine.

---

## 14. Understanding the Dashboard

### Device

Shows:

```text
CUDA
```

or:

```text
CPU
```

### Total Flows

Total processed network flows.

### Benign

Number of flows classified as benign.

### Attacks

Number of flows classified as attacks.

### High Severity

Number of detected flows assigned HIGH severity.

### Current Flow Detection

Displays the latest AI decision:

```text
Prediction: BENIGN
Attack Probability: 0.14%
Attack Family: BENIGN
Severity: NONE
```

### Recent Network Traffic

Shows recent predictions with:

```text
Time
Source
Destination
Prediction
Attack Probability
Family
Confidence
Severity
```

---

## 15. Prediction Logic

The current binary threshold is:

```text
0.80
```

Conceptually:

```text
Attack probability >= 0.80
              ↓
           ATTACK
```

and:

```text
Attack probability < 0.80
              ↓
           BENIGN
```

If the flow is classified as an attack, the multi-class model identifies the most likely attack family.

---

## 16. Severity Logic

Current backend rules:

```text
BENIGN
  ↓
NONE
```

High severity:

```text
DoS_DDoS
Infiltration
  ↓
HIGH
```

Other attack families:

```text
PortScan
BruteForce
WebAttack
Bot
  ↓
MEDIUM
```

This is a project-level severity rule and can be modified later.

---

## 17. Common Errors and Meaning

### `standard_scaler.joblib` not found

The live inference pipeline cannot reproduce the preprocessing used during training.

The scaler must exist at:

```text
data/final/features/standard_scaler.joblib
```

### `Unexpected key(s) in state_dict: "feature_position"`

The model architecture used for loading does not match the architecture used during training.

The Transformer definition must contain the same `feature_position` parameter used by the checkpoint.

### Feature-name mismatch

Example:

```text
ack_flag_cnt
active_max
```

versus:

```text
ACK Flag Count
Active Max
```

This indicates that the live flow names and training feature names are different. The backend mapping layer must resolve them before `scaler.transform()`.

### `Min Packet Length`

CICFlowMeter may produce:

```text
pkt_len_min
```

while the trained dataset expects:

```text
MinPacket Length
```

This must be explicitly mapped.

### `process_new_flows` not found

`process_new_flows()` must be inside:

```python
class LiveNIDS:
```

not outside it because of incorrect indentation.

Check:

```powershell
python -c "import ast; p='live_backend.py'; t=ast.parse(open(p,encoding='utf-8').read()); c=[n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='LiveNIDS'][0]; print([n.name for n in c.body if isinstance(n,ast.FunctionDef)])"
```

The output should contain:

```text
process_new_flows
```

### `start_capture() missing 1 required positional argument: 'interface'`

The frontend must call:

```python
nids.start_capture(interface)
```

instead of:

```python
nids.start_capture()
```

---

## 18. Security and Privacy

AI-NIDS is intended for:

- Academic demonstration
- Cybersecurity research
- Authorized network monitoring
- Controlled testing
- Learning

Packet/flow monitoring may expose sensitive network information.

**Only monitor networks and devices for which you have permission.**

AI-NIDS is an IDS, not a firewall. It primarily detects and reports suspicious traffic; it does not automatically block traffic.

---

## 19. Recommended Final-Year Project Demonstration

### Step 1 — Explain architecture

```text
Wi-Fi
 ↓
Packet Capture
 ↓
CICFlowMeter
 ↓
61 Features
 ↓
Binary Transformer
 ↓
Multi-class Transformer
 ↓
Dashboard
```

### Step 2 — Start live monitoring

Show:

```text
LIVE CAPTURE ACTIVE
```

### Step 3 — Generate normal traffic

```powershell
ping google.com -n 20
```

and:

```powershell
curl.exe https://example.com
```

### Step 4 — Show live predictions

Explain:

- Prediction
- Attack probability
- Family
- Confidence
- Severity

### Step 5 — Controlled anomaly/attack evaluation

For the final evaluation, use only an authorized and controlled test environment with known ground truth. Compare the AI prediction with the known class instead of assuming every high probability is a correct detection.

---

## 20. Future Improvements

Possible extensions include:

- Real-time charts
- Attack timeline
- Protocol distribution
- Source-IP statistics
- Destination-IP statistics
- Alert history
- CSV/report export
- Database storage
- Precision/Recall/F1 dashboard
- Confusion matrix
- Threshold tuning
- Explainable AI
- Alert notifications
- Multi-host monitoring
- Network topology visualization
- Controlled response/containment integration

---

## 21. Project Goal

The central idea is:

```text
Traditional monitoring:
Network Traffic → Human Inspection

AI-NIDS:
Network Traffic
      ↓
Automatic Flow Generation
      ↓
Feature Extraction
      ↓
AI Inference
      ↓
Attack Classification
      ↓
Real-Time Visualization
```

This demonstrates the integration of:

- Computer Networking
- Cybersecurity
- Machine Learning
- Deep Learning
- Network-flow analysis
- Real-time systems
- Streamlit web development

---

## 22. Quick Start

For an already configured project:

```powershell
cd "C:\Users\piyus\OneDrive\Desktop\AI-NIDS"
conda activate AI-NIDS
python -m streamlit run live_dashboard.py
```

Open:

```text
http://localhost:8501
```

Select:

```text
Wi-Fi
```

Click:

```text
Start Live Monitoring
```

Generate normal traffic:

```powershell
ping google.com -n 20
```

Then watch the dashboard.

---

## Current Project Status

The project currently has the major components of a live AI-NIDS:

- Transformer binary model
- Transformer multi-class model
- 61-feature preprocessing pipeline
- StandardScaler
- Live network-flow generation
- Live backend
- Streamlit dashboard
- Real-time prediction display
- Benign/attack statistics
- Attack-family classification
- Severity classification

The next major stage should be **validation**:

```text
Live capture
    ↓
Correct feature extraction
    ↓
Correct preprocessing
    ↓
Correct model inference
    ↓
Validate predictions
    ↓
Improve analytics
    ↓
Controlled anomaly/attack evaluation
```

---

## Academic / Authorized Use

This project is intended for academic, educational, and authorized cybersecurity research.

Always obtain permission before capturing or analyzing network traffic.
