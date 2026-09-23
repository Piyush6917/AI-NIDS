# AI-NIDS Simulation Lab

This mode replays the project's held-out test flows through the actual Transformer inference pipeline.

Run from the AI-NIDS project root:

```powershell
conda activate AI-NIDS
python -m streamlit run simulation_dashboard.py
```

Required project files:
- `data/final/features/X_test.npy`
- `data/final/features/y_test_binary.npy`
- `data/final/features/y_test_family.npy`
- `data/final/features/standard_scaler.joblib`
- `models/transformer_binary_gpu.pth`
- `models/transformer_multiclass_gpu_v2.pth`

Scenarios: BENIGN, DoS_DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration, MIXED.

Ground truth is never supplied to the models; it is shown afterward only for validation.
