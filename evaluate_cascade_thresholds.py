import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80
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
            nn.Linear(
                d_model,
                64
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                64,
                num_classes
            )
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
print("AI-NIDS - TWO-STAGE CASCADE THRESHOLD ANALYSIS")
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
print("GENERATING MODEL PROBABILITIES")
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

binary_probabilities = []
multiclass_predictions = []

with torch.no_grad():

    for (X_batch,) in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Binary
        # ----------------------------------------------------

        binary_logits = binary_model(
            X_batch
        )

        binary_probs = torch.softmax(
            binary_logits,
            dim=1
        )

        attack_probs = (
            binary_probs[:, 1]
            .cpu()
            .numpy()
        )

        binary_probabilities.extend(
            attack_probs
        )

        # ----------------------------------------------------
        # Multi-class
        # ----------------------------------------------------

        multiclass_logits = multiclass_model(
            X_batch
        )

        multiclass_pred = torch.argmax(
            multiclass_logits,
            dim=1
        )

        multiclass_predictions.extend(
            multiclass_pred
            .cpu()
            .numpy()
        )

binary_probabilities = np.array(
    binary_probabilities
)

np.save(
    FEATURE_DIR / "binary_test_probabilities.npy",
    binary_probabilities
)

multiclass_predictions = np.array(
    multiclass_predictions
)

print(
    f"Probabilities generated: "
    f"{len(binary_probabilities):,}"
)


# ============================================================
# BINARY MISS ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("BINARY GATE ANALYSIS")
print("=" * 80)

print(
    "\nActual ATTACK samples rejected "
    "by the binary gate:"
)

for threshold in THRESHOLDS:

    predicted_attack = (
        binary_probabilities >= threshold
    )

    actual_attack = (
        y_binary == 1
    )

    missed_attacks = np.sum(
        actual_attack &
        ~predicted_attack
    )

    detected_attacks = np.sum(
        actual_attack &
        predicted_attack
    )

    attack_recall = (
        detected_attacks /
        np.sum(actual_attack)
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Detected: {detected_attacks:,} | "
        f"Missed: {missed_attacks:,} | "
        f"Recall: {attack_recall:.4f}"
    )


# ============================================================
# CASCADE EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("TWO-STAGE CASCADE RESULTS")
print("=" * 80)

print(
    "\nThreshold    Accuracy    Precision    Recall      F1"
)

print("-" * 65)

for threshold in THRESHOLDS:

    # Binary gate
    binary_attack = (
        binary_probabilities >= threshold
    )

    # Final prediction
    #
    # BENIGN = class 0
    # ATTACK = multi-class prediction
    #
    final_predictions = np.zeros(
        len(X_test),
        dtype=np.int64
    )

    attack_indices = np.where(
        binary_attack
    )[0]

    final_predictions[
        attack_indices
    ] = multiclass_predictions[
        attack_indices
    ]

    # Convert family prediction into binary
    final_binary_prediction = (
        final_predictions != 0
    ).astype(np.int64)

    accuracy = accuracy_score(
        y_binary,
        final_binary_prediction
    )

    precision = precision_score(
        y_binary,
        final_binary_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_binary,
        final_binary_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_binary,
        final_binary_prediction,
        zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<13.4f}"
        f"{recall:<12.4f}"
        f"{f1:.4f}"
    )


# ============================================================
# ATTACK FAMILY PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("ATTACK-FAMILY RESULTS AT EACH THRESHOLD")
print("=" * 80)

for threshold in THRESHOLDS:

    binary_attack = (
        binary_probabilities >= threshold
    )

    final_predictions = np.zeros(
        len(X_test),
        dtype=np.int64
    )

    attack_indices = np.where(
        binary_attack
    )[0]

    final_predictions[
        attack_indices
    ] = multiclass_predictions[
        attack_indices
    ]

    precision, recall, f1, support = (
        __import__(
            "sklearn.metrics",
            fromlist=[
                "precision_recall_fscore_support"
            ]
        ).precision_recall_fscore_support(
            y_family,
            final_predictions,
            labels=np.arange(7),
            zero_division=0
        )
    )

    print(
        f"\nThreshold: {threshold:.2f}"
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
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("CASCADE THRESHOLD ANALYSIS COMPLETE")
print("=" * 80)