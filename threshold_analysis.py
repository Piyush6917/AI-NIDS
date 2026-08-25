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
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("data/final/features")
MODEL_FILE = Path("models/transformer_binary_gpu.pth")

BATCH_SIZE = 512

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
print("AI-NIDS - TRANSFORMER THRESHOLD ANALYSIS")
print("=" * 80)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING VALIDATION DATA")
print("=" * 80)

X_val = np.load(
    FEATURE_DIR / "X_validation.npy"
)

y_val = np.load(
    FEATURE_DIR / "y_validation_binary.npy"
)

print(
    f"Validation samples: {len(y_val):,}"
)

print(
    f"Features: {X_val.shape[1]}"
)


# ============================================================
# CREATE MODEL
# ============================================================

model = NetworkTransformer(
    num_features=X_val.shape[1],
    d_model=64,
    n_heads=4,
    n_layers=2,
    dropout=0.1
).to(DEVICE)


# ============================================================
# LOAD TRAINED GPU MODEL
# ============================================================

print("\nLoading trained GPU Transformer...")

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# CONVERT DATA
# ============================================================

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)


# ============================================================
# GET ATTACK PROBABILITIES
# ============================================================

print("\n" + "=" * 80)
print("GENERATING ATTACK PROBABILITIES")
print("=" * 80)

probabilities = []

with torch.no_grad():

    for start in range(
        0,
        len(X_val_tensor),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_val_tensor)
        )

        batch = X_val_tensor[
            start:end
        ].to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            batch
        )

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


print(
    f"Generated probabilities: "
    f"{len(probabilities):,}"
)

print(
    f"Validation ROC-AUC: "
    f"{roc_auc_score(y_val, probabilities):.4f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


print(
    "\n"
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FPR':<12}"
    f"{'FNR':<12}"
)

print("-" * 72)


results = []


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )


    cm = confusion_matrix(
        y_val,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    fpr = fp / (
        fp + tn
    )

    fnr = fn / (
        fn + tp
    )


    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "fnr": fnr
        }
    )


    print(
        f"{threshold:<12.2f}"
        f"{accuracy:<12.4f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fpr:<12.4f}"
        f"{fnr:<12.4f}"
    )


# ============================================================
# BEST THRESHOLDS
# ============================================================

print("\n" + "=" * 80)
print("BEST THRESHOLDS")
print("=" * 80)


best_f1 = max(
    results,
    key=lambda x: x["f1"]
)

best_precision = max(
    results,
    key=lambda x: x["precision"]
)


# Best threshold while maintaining >=95% recall

high_recall_results = [
    r for r in results
    if r["recall"] >= 0.95
]

if high_recall_results:

    best_high_recall = max(
        high_recall_results,
        key=lambda x: x["precision"]
    )

else:

    best_high_recall = None


print(
    "\nBest F1 threshold:"
)

print(
    f"Threshold : {best_f1['threshold']:.2f}"
)

print(
    f"Precision : {best_f1['precision']:.4f}"
)

print(
    f"Recall    : {best_f1['recall']:.4f}"
)

print(
    f"F1        : {best_f1['f1']:.4f}"
)

print(
    f"FPR       : {best_f1['fpr']:.4f}"
)


print(
    "\nBest Precision threshold:"
)

print(
    f"Threshold : {best_precision['threshold']:.2f}"
)

print(
    f"Precision : {best_precision['precision']:.4f}"
)

print(
    f"Recall    : {best_precision['recall']:.4f}"
)

print(
    f"F1        : {best_precision['f1']:.4f}"
)


if best_high_recall:

    print(
        "\nBest Precision with Recall >= 95%:"
    )

    print(
        f"Threshold : "
        f"{best_high_recall['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best_high_recall['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_high_recall['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{best_high_recall['f1']:.4f}"
    )

    print(
        f"FPR       : "
        f"{best_high_recall['fpr']:.4f}"
    )


print("\n" + "=" * 80)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 80)