import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
FEATURE_DIR = BASE_DIR / "data" / "final" / "features"
MULTICLASS_DIR = BASE_DIR / "data" / "final" / "multiclass"

BINARY_MODEL = MODEL_DIR / "transformer_binary_gpu.pth"
MULTICLASS_MODEL = MODEL_DIR / "transformer_multiclass_gpu_v2.pth"

X_TEST_FILE = FEATURE_DIR / "X_test.npy"
Y_BINARY_FILE = FEATURE_DIR / "y_test_binary.npy"
Y_FAMILY_FILE = MULTICLASS_DIR / "y_test.npy"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 512

# Binary model remains at our optimized threshold.
BINARY_THRESHOLD = 0.80

# We will experiment with these multi-class confidence levels.
FAMILY_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

CLASS_NAMES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration"
]


# ============================================================
# MODEL
# ============================================================

class NetworkTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        num_classes,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.feature_embedding = nn.Linear(
            1,
            d_model
        )

        self.feature_position = nn.Parameter(
            torch.randn(
                1,
                num_features,
                d_model
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = x.unsqueeze(-1)
        x = self.feature_embedding(x)
        x = x + self.feature_position
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)

        return self.classifier(x)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("AI-NIDS - SOFT CASCADE ANALYSIS")
print("=" * 80)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

X_test = np.load(
    X_TEST_FILE
).astype(np.float32)

y_binary = np.load(
    Y_BINARY_FILE
).astype(np.int64
)

y_family = np.load(
    Y_FAMILY_FILE
).astype(np.int64)

print(
    f"Test samples: {len(X_test):,}"
)

print(
    f"Features: {X_test.shape[1]}"
)


# ============================================================
# LOAD BINARY MODEL
# ============================================================

print("\nLoading binary Transformer...")

binary_model = NetworkTransformer(
    num_features=61,
    num_classes=2
).to(DEVICE)

checkpoint = torch.load(
    BINARY_MODEL,
    map_location=DEVICE,
    weights_only=False
)

binary_model.load_state_dict(
    checkpoint["model_state_dict"]
)

binary_model.eval()

print("Binary model loaded.")


# ============================================================
# LOAD MULTI-CLASS MODEL
# ============================================================

print("\nLoading multi-class V2 Transformer...")

multiclass_model = NetworkTransformer(
    num_features=61,
    num_classes=7
).to(DEVICE)

checkpoint = torch.load(
    MULTICLASS_MODEL,
    map_location=DEVICE,
    weights_only=False
)

multiclass_model.load_state_dict(
    checkpoint["model_state_dict"]
)

multiclass_model.eval()

print("Multi-class V2 model loaded.")


# ============================================================
# GENERATE PROBABILITIES
# ============================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

dataset = TensorDataset(
    torch.from_numpy(X_test)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

binary_probs_all = []
family_probs_all = []

with torch.no_grad():

    for (X_batch,) in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        # Binary probabilities
        binary_logits = binary_model(
            X_batch
        )

        binary_probs = torch.softmax(
            binary_logits,
            dim=1
        )

        binary_attack_prob = (
            binary_probs[:, 1]
            .cpu()
            .numpy()
        )

        binary_probs_all.extend(
            binary_attack_prob
        )

        # Multi-class probabilities
        family_logits = multiclass_model(
            X_batch
        )

        family_probs = torch.softmax(
            family_logits,
            dim=1
        )

        family_probs_all.extend(
            family_probs.cpu().numpy()
        )

binary_probs_all = np.array(
    binary_probs_all
)

family_probs_all = np.array(
    family_probs_all
)

family_predictions = np.argmax(
    family_probs_all,
    axis=1
)

family_confidences = np.max(
    family_probs_all,
    axis=1
)

print(
    f"Generated predictions: "
    f"{len(binary_probs_all):,}"
)


# ============================================================
# SOFT CASCADE
# ============================================================

print("\n" + "=" * 80)
print("SOFT CASCADE RESULTS")
print("=" * 80)

print(
    "\nBinary threshold fixed at: "
    f"{BINARY_THRESHOLD:.2f}"
)

print(
    "\nFamily Threshold | Accuracy | "
    "Precision | Recall | F1 | Macro F1"
)

print("-" * 80)

results = []


for family_threshold in FAMILY_THRESHOLDS:

    final_binary = np.zeros(
        len(X_test),
        dtype=np.int64
    )

    final_family = np.zeros(
        len(X_test),
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Normal binary detection
    # --------------------------------------------------------

    binary_attack = (
        binary_probs_all >= BINARY_THRESHOLD
    )

    # --------------------------------------------------------
    # Multi-class confident attack
    #
    # If multi-class predicts an attack family with
    # sufficient confidence, allow it to override
    # the binary BENIGN decision.
    # --------------------------------------------------------

    confident_family_attack = (
        (family_predictions != 0) &
        (family_confidences >= family_threshold)
    )

    final_attack = (
        binary_attack |
        confident_family_attack
    )

    final_family[
        final_attack
    ] = family_predictions[
        final_attack
    ]

    final_binary[
        final_attack
    ] = 1

    accuracy = accuracy_score(
        y_binary,
        final_binary
    )

    precision = precision_score(
        y_binary,
        final_binary,
        zero_division=0
    )

    recall = recall_score(
        y_binary,
        final_binary,
        zero_division=0
    )

    f1 = f1_score(
        y_binary,
        final_binary,
        zero_division=0
    )

    family_p, family_r, family_f1, _ = (
        precision_recall_fscore_support(
            y_family,
            final_family,
            labels=np.arange(7),
            zero_division=0
        )
    )

    macro_f1 = family_f1.mean()

    results.append(
        (
            family_threshold,
            accuracy,
            precision,
            recall,
            f1,
            macro_f1
        )
    )

    print(
        f"{family_threshold:<17.2f}"
        f"{accuracy:<11.4f}"
        f"{precision:<11.4f}"
        f"{recall:<9.4f}"
        f"{f1:<9.4f}"
        f"{macro_f1:.4f}"
    )


# ============================================================
# ATTACK-FAMILY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("ATTACK-FAMILY PERFORMANCE")
print("=" * 80)


for family_threshold in FAMILY_THRESHOLDS:

    binary_attack = (
        binary_probs_all >= BINARY_THRESHOLD
    )

    confident_family_attack = (
        (family_predictions != 0) &
        (family_confidences >= family_threshold)
    )

    final_attack = (
        binary_attack |
        confident_family_attack
    )

    final_family = np.zeros(
        len(X_test),
        dtype=np.int64
    )

    final_family[
        final_attack
    ] = family_predictions[
        final_attack
    ]

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_family,
            final_family,
            labels=np.arange(7),
            zero_division=0
        )
    )

    print(
        f"\nFamily confidence threshold: "
        f"{family_threshold:.2f}"
    )

    print(
        f"{'Class':<18}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print("-" * 50)

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{name:<18}"
            f"{precision[i]:>10.4f}"
            f"{recall[i]:>10.4f}"
            f"{f1[i]:>10.4f}"
        )


# ============================================================
# BEST RESULTS
# ============================================================

print("\n" + "=" * 80)
print("BEST SOFT-CASCADE RESULTS")
print("=" * 80)

best_binary = max(
    results,
    key=lambda x: x[4]
)

best_macro = max(
    results,
    key=lambda x: x[5]
)

print(
    f"\nBest binary F1:"
)

print(
    f"Family threshold : "
    f"{best_binary[0]:.2f}"
)

print(
    f"Accuracy         : "
    f"{best_binary[1]:.4f}"
)

print(
    f"Precision        : "
    f"{best_binary[2]:.4f}"
)

print(
    f"Recall           : "
    f"{best_binary[3]:.4f}"
)

print(
    f"F1               : "
    f"{best_binary[4]:.4f}"
)

print(
    f"\nBest attack-family Macro F1:"
)

print(
    f"Family threshold : "
    f"{best_macro[0]:.2f}"
)

print(
    f"Macro F1         : "
    f"{best_macro[5]:.4f}"
)

print("\n" + "=" * 80)
print("SOFT CASCADE ANALYSIS COMPLETE")
print("=" * 80)