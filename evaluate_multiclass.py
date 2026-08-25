import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/final/multiclass")
MODEL_FILE = Path(
    "models/transformer_multiclass_gpu_v2.pth"
)

BATCH_SIZE = 512
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration"
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("AI-NIDS - MULTI-CLASS TRANSFORMER TEST EVALUATION")
print("=" * 80)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING TEST DATA")
print("=" * 80)

X_test = np.load(
    DATA_DIR / "X_test.npy"
).astype(np.float32)

y_test = np.load(
    DATA_DIR / "y_test.npy"
).astype(np.int64)

print(
    f"Test samples: {len(y_test):,}"
)

print(
    f"Features: {X_test.shape[1]}"
)


# ============================================================
# MODEL
# ============================================================

class NetworkTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        num_classes=7,
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
# LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING BEST MODEL")
print("=" * 80)

model = NetworkTransformer(
    num_features=X_test.shape[1],
    num_classes=NUM_CLASSES
).to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Best multi-class Transformer loaded.")

print(
    f"Best validation Macro F1: "
    f"{checkpoint.get('best_val_macro_f1', 'N/A')}"
)


# ============================================================
# DATA LOADER
# ============================================================

dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("GENERATING TEST PREDICTIONS")
print("=" * 80)

all_predictions = []
all_labels = []

with torch.no_grad():

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            X_batch
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            y_batch.numpy()
        )

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        average="macro",
        zero_division=0
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
)


print("\n" + "=" * 80)
print("FINAL MULTI-CLASS TEST RESULTS")
print("=" * 80)

print(
    f"\nAccuracy         : {accuracy:.4f}"
)

print(
    f"Macro Precision  : {macro_precision:.4f}"
)

print(
    f"Macro Recall     : {macro_recall:.4f}"
)

print(
    f"Macro F1         : {macro_f1:.4f}"
)

print(
    f"Weighted Precision: {weighted_precision:.4f}"
)

print(
    f"Weighted Recall   : {weighted_recall:.4f}"
)

print(
    f"Weighted F1       : {weighted_f1:.4f}"
)


# ============================================================
# PER CLASS
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0
    )
)

print("\n" + "=" * 80)
print("PER-CLASS PERFORMANCE")
print("=" * 80)

print(
    f"\n{'Class':<18}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'Support':>12}"
)

print("-" * 66)

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{name:<18}"
        f"{precision[i]:>12.4f}"
        f"{recall[i]:>12.4f}"
        f"{f1[i]:>12.4f}"
        f"{support[i]:>12,}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    f"{'':<18}"
    + "".join(
        f"{name[:10]:>12}"
        for name in CLASS_NAMES
    )
)

for i, row in enumerate(cm):

    print(
        f"{CLASS_NAMES[i]:<18}"
        + "".join(
            f"{value:>12,}"
            for value in row
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("MULTI-CLASS TEST EVALUATION COMPLETE")
print("=" * 80)

print(
    f"\nModel: {MODEL_FILE.resolve()}"
)

print(
    f"Test Macro F1: {macro_f1:.4f}"
)