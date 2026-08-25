import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# CONFIG
# ============================================================

FEATURE_DIR = Path("data/final/features")
MODEL_FILE = Path("models/transformer_binary_gpu.pth")

BATCH_SIZE = 512
THRESHOLD = 0.80

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# MODEL
# ============================================================

class NetworkTransformer(nn.Module):

    def __init__(
        self,
        num_features=61,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.feature_embedding = nn.Linear(
            1, d_model
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

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
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
# START
# ============================================================

print("=" * 80)
print("AI-NIDS - FINAL TEST WITH OPTIMIZED THRESHOLD")
print("=" * 80)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print(f"Detection threshold: {THRESHOLD}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

X_test = np.load(
    FEATURE_DIR / "X_test.npy"
)

y_test = np.load(
    FEATURE_DIR / "y_test_binary.npy"
)

print(
    f"Test samples: {len(y_test):,}"
)

print(
    f"Features: {X_test.shape[1]}"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = NetworkTransformer(
    num_features=X_test.shape[1]
).to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nGPU Transformer loaded successfully.")


# ============================================================
# PREDICT
# ============================================================

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

probabilities = []

print("\nGenerating test probabilities...")

with torch.no_grad():

    for start in range(
        0,
        len(X_test_tensor),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_test_tensor)
        )

        batch = X_test_tensor[
            start:end
        ].to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(batch)

        probs = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        probabilities.extend(
            probs.cpu().numpy()
        )


probabilities = np.array(
    probabilities
)

# ============================================================
# APPLY 0.80 THRESHOLD
# ============================================================

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn)

fnr = fn / (fn + tp)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS - THRESHOLD 0.80")
print("=" * 80)

print(f"\nAccuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"ROC-AUC        : {auc:.4f}")
print(f"False Positive Rate : {fpr:.4f}")
print(f"False Negative Rate : {fnr:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print("\n                Predicted")
print("              BENIGN  ATTACK")

print(
    f"Actual BENIGN "
    f"{tn:>9,}"
    f"{fp:>9,}"
)

print(
    f"Actual ATTACK "
    f"{fn:>9,}"
    f"{tp:>9,}"
)


# ============================================================
# ALERT STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("IDS ALERT STATISTICS")
print("=" * 80)

print(
    f"\nTotal benign flows : "
    f"{tn + fp:,}"
)

print(
    f"False alarms       : "
    f"{fp:,}"
)

print(
    f"Total attacks      : "
    f"{fn + tp:,}"
)

print(
    f"Attacks detected   : "
    f"{tp:,}"
)

print(
    f"Attacks missed     : "
    f"{fn:,}"
)


print("\n" + "=" * 80)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 80)