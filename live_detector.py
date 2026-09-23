"""
Compatibility wrapper for the AI-NIDS live detector.

The main implementation now lives in live_backend.py.
Keeping this wrapper means older commands/imports that use
live_detector.py continue to work.
"""

from live_backend import (
    BASE,
    LIVE_DIR,
    LIVE_CSV,
    OUT_CSV,
    SCALER_FILE,
    BINARY_FILE,
    MULTI_FILE,
    CICFLOWMETER,
    BINARY_THRESHOLD,
    CLASSES,
    FEATURE_MAP,
    NetworkTransformer,
    load_model,
    LiveNIDS,
)


def main():
    print("AI-NIDS Live Detector")
    print("---------------------")
    print(f"Backend: {BASE / 'live_backend.py'}")
    print(f"Device: {LiveNIDS().device}")
    print()
    print("This module is controlled by live_dashboard.py.")
    print("Use the Streamlit dashboard to start REAL or LAB capture.")


if __name__ == "__main__":
    main()
