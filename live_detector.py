import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

LIVE_CSV = BASE / "data" / "live" / "live_flows.csv"
OUT_CSV = BASE / "data" / "live" / "live_predictions.csv"

SCALER_FILE = (
    BASE
    / "data"
    / "final"
    / "features"
    / "standard_scaler.joblib"
)

BINARY_FILE = (
    BASE
    / "models"
    / "transformer_binary_gpu.pth"
)

MULTI_FILE = (
    BASE
    / "models"
    / "transformer_multiclass_gpu_v2.pth"
)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.80

CLASSES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration",
]


# ============================================================
# CICFLOWMETER -> MODEL FEATURE MAPPING
# ============================================================

FEATURE_MAP = {

    # --------------------------------------------------------
    # Basic flow information
    # --------------------------------------------------------

    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",

    # --------------------------------------------------------
    # Packet counts
    # --------------------------------------------------------

    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",

    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",

    # --------------------------------------------------------
    # Forward packet length
    # --------------------------------------------------------

    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",

    # --------------------------------------------------------
    # Backward packet length
    # --------------------------------------------------------

    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",

    # --------------------------------------------------------
    # Flow rates
    # --------------------------------------------------------

    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",

    # --------------------------------------------------------
    # Flow IAT
    # --------------------------------------------------------

    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",

    # --------------------------------------------------------
    # Forward IAT
    # --------------------------------------------------------

    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",

    # --------------------------------------------------------
    # Backward IAT
    # --------------------------------------------------------

    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",

    # --------------------------------------------------------
    # Flags / headers
    # --------------------------------------------------------

    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",

    # --------------------------------------------------------
    # Packet rates
    # --------------------------------------------------------

    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",

    # --------------------------------------------------------
    # Packet length statistics
    # IMPORTANT: exact scaler names
    # --------------------------------------------------------

    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",

    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",

    # --------------------------------------------------------
    # TCP flags
    # --------------------------------------------------------

    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "ECE Flag Count": "ece_flag_cnt",

    # --------------------------------------------------------
    # Other features
    # --------------------------------------------------------

    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",

    # --------------------------------------------------------
    # TCP window
    # --------------------------------------------------------

    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",

    # --------------------------------------------------------
    # Forward segment information
    # --------------------------------------------------------

    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",

    # --------------------------------------------------------
    # Active statistics
    # --------------------------------------------------------

    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",

    # --------------------------------------------------------
    # Idle statistics
    # --------------------------------------------------------

    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}


# ============================================================
# TRANSFORMER MODEL
# ============================================================

class NetworkTransformer(nn.Module):

    def __init__(
        self,
        num_classes,
        num_features=61,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()

        # Each feature is a scalar.
        # Convert scalar -> d_model.
        self.feature_embedding = nn.Linear(
            1,
            d_model
        )

        # IMPORTANT:
        # Your trained models contain this parameter.
        self.feature_position = nn.Parameter(
            torch.randn(
                1,
                num_features,
                d_model
            )
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                d_model,
                64
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # [batch, 61]

        x = x.unsqueeze(-1)

        # [batch, 61, 1]
        # ->
        # [batch, 61, 64]

        x = self.feature_embedding(
            x
        )

        # Add feature identity
        x = x + self.feature_position

        # Transformer
        x = self.transformer(
            x
        )

        # Mean pooling
        x = x.mean(
            dim=1
        )

        # Normalization
        x = self.norm(
            x
        )

        # Classification
        return self.classifier(
            x
        )


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    path,
    num_classes,
    device
):

    print(
        f"Loading model: {path.name}"
    )

    model = NetworkTransformer(
        num_classes=num_classes,
        num_features=61
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state = checkpoint[
                "state_dict"
            ]

        else:

            state = checkpoint

    else:

        state = checkpoint

    model.load_state_dict(
        state,
        strict=True
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# FEATURE PREPARATION
# ============================================================

def make_features(
    df,
    scaler
):

    # --------------------------------------------------------
    # Get exact feature names used during scaler training
    # --------------------------------------------------------

    expected = list(
        scaler.feature_names_in_
    )

    resolved_columns = []
    missing_columns = []

    # --------------------------------------------------------
    # Resolve each model feature
    # --------------------------------------------------------

    for feature in expected:

        if feature not in FEATURE_MAP:

            missing_columns.append(
                f"No mapping for model feature: {feature}"
            )

            continue

        live_column = FEATURE_MAP[
            feature
        ]

        if live_column not in df.columns:

            missing_columns.append(
                f"{feature} -> {live_column}"
            )

            continue

        resolved_columns.append(
            live_column
        )

    # --------------------------------------------------------
    # Check missing features
    # --------------------------------------------------------

    if missing_columns:

        print(
            "\nERROR: Missing CICFlowMeter features:"
        )

        for item in missing_columns:

            print(
                "  -",
                item
            )

        raise RuntimeError(
            "CICFlowMeter does not contain "
            "all required model features."
        )

    # --------------------------------------------------------
    # Select EXACTLY 61 columns
    # in scaler training order
    # --------------------------------------------------------

    x = df[
        resolved_columns
    ].copy()

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    x = x.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace infinity
    # --------------------------------------------------------

    x = x.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    # --------------------------------------------------------
    # Replace missing values
    # --------------------------------------------------------

    x = x.fillna(
        0
    )

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # scaler was trained with feature names such as:
    #
    # ACK Flag Count
    # Active Max
    # Min Packet Length
    #
    # But our live DataFrame contains:
    #
    # ack_flag_cnt
    # active_max
    # pkt_len_min
    #
    # The columns are already in the correct order,
    # so pass NumPy data to avoid sklearn feature-name
    # validation.
    # --------------------------------------------------------

    x = scaler.transform(
        x.to_numpy()
    )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    x = x.astype(
        np.float32
    )

    return x


# ============================================================
# PREDICTION
# ============================================================

def predict(
    x,
    binary_model,
    multi_model,
    device
):

    tensor = torch.from_numpy(
        x
    ).to(
        device
    )

    with torch.inference_mode():

        # ----------------------------------------------------
        # Binary model
        # ----------------------------------------------------

        binary_logits = binary_model(
            tensor
        )

        binary_probabilities = torch.softmax(
            binary_logits,
            dim=1
        )

        # ----------------------------------------------------
        # Multi-class model
        # ----------------------------------------------------

        multi_logits = multi_model(
            tensor
        )

        multi_probabilities = torch.softmax(
            multi_logits,
            dim=1
        )

    results = []

    # ========================================================
    # Process every flow
    # ========================================================

    for i in range(
        len(x)
    ):

        # ----------------------------------------------------
        # Attack probability
        # ----------------------------------------------------

        attack_probability = float(
            binary_probabilities[
                i,
                1
            ]
        )

        # ----------------------------------------------------
        # Multi-class prediction
        # ----------------------------------------------------

        class_index = int(
            torch.argmax(
                multi_probabilities[i]
            )
        )

        family = CLASSES[
            class_index
        ]

        confidence = float(
            multi_probabilities[
                i,
                class_index
            ]
        )

        # ----------------------------------------------------
        # Binary decision
        # ----------------------------------------------------

        if attack_probability >= THRESHOLD:

            prediction = "ATTACK"

            # If multi-class model says BENIGN,
            # choose strongest attack family.
            if family == "BENIGN":

                sorted_indices = torch.argsort(
                    multi_probabilities[i],
                    descending=True
                )

                for j in sorted_indices:

                    j = int(j)

                    if CLASSES[j] != "BENIGN":

                        family = CLASSES[
                            j
                        ]

                        confidence = float(
                            multi_probabilities[
                                i,
                                j
                            ]
                        )

                        break

        else:

            prediction = "BENIGN"

            family = "BENIGN"

        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        if prediction == "BENIGN":

            severity = "NONE"

        elif family in [
            "DoS_DDoS",
            "Infiltration"
        ]:

            severity = "HIGH"

        else:

            severity = "MEDIUM"

        results.append(
            (
                prediction,
                attack_probability,
                family,
                confidence,
                severity
            )
        )

    return results


# ============================================================
# INITIAL FLOW COUNT
# ============================================================

def get_existing_flow_count():

    try:

        df = pd.read_csv(
            LIVE_CSV
        )

        return len(df)

    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError
    ):

        return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "AI-NIDS - REAL LIVE PACKET DETECTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    print()

    # --------------------------------------------------------
    # Load scaler
    # --------------------------------------------------------

    print(
        "Loading StandardScaler..."
    )

    scaler = joblib.load(
        SCALER_FILE
    )

    print(
        "Scaler loaded:",
        len(
            scaler.feature_names_in_
        ),
        "features"
    )

    # --------------------------------------------------------
    # Load binary Transformer
    # --------------------------------------------------------

    print(
        "\nLoading Binary Transformer..."
    )

    binary_model = load_model(
        BINARY_FILE,
        2,
        device
    )

    # --------------------------------------------------------
    # Load multi-class Transformer
    # --------------------------------------------------------

    print(
        "\nLoading Multi-class Transformer V2..."
    )

    multi_model = load_model(
        MULTI_FILE,
        7,
        device
    )

    print(
        "\nBoth Transformer models loaded."
    )

    # --------------------------------------------------------
    # Check live CSV
    # --------------------------------------------------------

    if not LIVE_CSV.exists():

        print(
            "\nERROR:"
        )

        print(
            "live_flows.csv was not found."
        )

        print(
            "\nStart CICFlowMeter first."
        )

        return

    # --------------------------------------------------------
    # Ignore flows that already exist
    #
    # This is important because we want the detector
    # to process only NEW live flows.
    # --------------------------------------------------------

    processed = get_existing_flow_count()

    print(
        "\n" + "=" * 70
    )

    print(
        "WATCHING LIVE TRAFFIC"
    )

    print(
        "=" * 70
    )

    print(
        "File:",
        LIVE_CSV
    )

    print(
        "Existing flows:",
        processed
    )

    print(
        "\nWaiting for NEW network flows..."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    # ========================================================
    # LIVE LOOP
    # ========================================================

    while True:

        try:

            # ------------------------------------------------
            # Read current CSV
            # ------------------------------------------------

            try:

                df = pd.read_csv(
                    LIVE_CSV
                )

            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError
            ):

                time.sleep(
                    1
                )

                continue

            # ------------------------------------------------
            # Check for new flows
            # ------------------------------------------------

            if len(df) <= processed:

                time.sleep(
                    1
                )

                continue

            # ------------------------------------------------
            # Extract only new flows
            # ------------------------------------------------

            new_flows = df.iloc[
                processed:
            ].copy()

            print(
                "\n" + "=" * 70
            )

            print(
                f"NEW FLOWS DETECTED: "
                f"{len(new_flows)}"
            )

            print(
                "=" * 70
            )

            # ------------------------------------------------
            # Convert live CICFlowMeter features
            # into model features
            # ------------------------------------------------

            x = make_features(
                new_flows,
                scaler
            )

            print(
                "Features prepared:",
                x.shape
            )

            # ------------------------------------------------
            # AI inference
            # ------------------------------------------------

            results = predict(
                x,
                binary_model,
                multi_model,
                device
            )

            output = []

            # =================================================
            # Display every prediction
            # =================================================

            for row, result in zip(
                new_flows.to_dict(
                    "records"
                ),
                results
            ):

                (
                    prediction,
                    attack_probability,
                    family,
                    confidence,
                    severity
                ) = result

                # ---------------------------------------------
                # Save prediction
                # ---------------------------------------------

                output.append({

                    "timestamp":
                        row.get(
                            "timestamp",
                            ""
                        ),

                    "src_ip":
                        row.get(
                            "src_ip",
                            ""
                        ),

                    "src_port":
                        row.get(
                            "src_port",
                            ""
                        ),

                    "dst_ip":
                        row.get(
                            "dst_ip",
                            ""
                        ),

                    "dst_port":
                        row.get(
                            "dst_port",
                            ""
                        ),

                    "protocol":
                        row.get(
                            "protocol",
                            ""
                        ),

                    "prediction":
                        prediction,

                    "attack_probability":
                        attack_probability,

                    "attack_family":
                        family,

                    "family_confidence":
                        confidence,

                    "severity":
                        severity
                })

                # ---------------------------------------------
                # Console output
                # ---------------------------------------------

                print(
                    "\n" + "-" * 70
                )

                print(
                    "FLOW"
                )

                print(
                    f"{row.get('src_ip', '?')}:"
                    f"{row.get('src_port', '?')}"
                    f"  ->  "
                    f"{row.get('dst_ip', '?')}:"
                    f"{row.get('dst_port', '?')}"
                )

                print(
                    "Protocol:",
                    row.get(
                        "protocol",
                        "?"
                    )
                )

                print(
                    "Prediction:",
                    prediction
                )

                print(
                    f"Attack probability: "
                    f"{attack_probability * 100:.2f}%"
                )

                print(
                    "Attack family:",
                    family
                )

                print(
                    f"Family confidence: "
                    f"{confidence * 100:.2f}%"
                )

                print(
                    "Severity:",
                    severity
                )

                if prediction == "ATTACK":

                    print(
                        "\n🚨 ATTACK DETECTED 🚨"
                    )

                else:

                    print(
                        "\n🟢 BENIGN TRAFFIC"
                    )

            # ------------------------------------------------
            # Save predictions to CSV
            # ------------------------------------------------

            if output:

                output_df = pd.DataFrame(
                    output
                )

                output_df.to_csv(
                    OUT_CSV,
                    mode="a",
                    header=not OUT_CSV.exists(),
                    index=False
                )

                print(
                    "\nPredictions saved to:"
                )

                print(
                    OUT_CSV
                )

            # ------------------------------------------------
            # VERY IMPORTANT:
            # Update processed count only after successful
            # feature extraction and prediction.
            # ------------------------------------------------

            processed = len(
                df
            )

        # ====================================================
        # CTRL+C
        # ====================================================

        except KeyboardInterrupt:

            print(
                "\n\n"
                + "=" * 70
            )

            print(
                "AI-NIDS LIVE DETECTOR STOPPED"
            )

            print(
                "=" * 70
            )

            break

        # ====================================================
        # Other runtime errors
        # ====================================================

        except Exception as e:

            print(
                "\nLive detector error:"
            )

            print(
                repr(e)
            )

            print(
                "\nWaiting for next flow..."
            )

            time.sleep(
                2
            )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()