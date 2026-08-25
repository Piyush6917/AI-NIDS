import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/final/multiclass")
MODEL_FILE = Path("models/transformer_multiclass_gpu.pth")

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
# START
# ============================================================

print("=" * 80)
print("AI-NIDS - MULTI-CLASS ERROR ANALYSIS")
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


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("LOADING MODEL")
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

print("Model loaded successfully.")


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
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

all_predictions = []

with torch.no_grad():

    for X_batch, _ in loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(X_batch)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

y_pred = np.array(
    all_predictions
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)


# ============================================================
# PER-CLASS ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 80)

for class_id, class_name in enumerate(CLASS_NAMES):

    actual_count = np.sum(
        y_test == class_id
    )

    correct = cm[
        class_id,
        class_id
    ]

    incorrect = (
        actual_count - correct
    )

    recall = (
        correct / actual_count
        if actual_count > 0
        else 0
    )

    print("\n" + "-" * 70)

    print(
        f"{class_name}"
    )

    print(
        f"Actual samples       : "
        f"{actual_count:,}"
    )

    print(
        f"Correct predictions  : "
        f"{correct:,}"
    )

    print(
        f"Incorrect predictions: "
        f"{incorrect:,}"
    )

    print(
        f"Recall               : "
        f"{recall:.4f}"
    )

    # Where did this actual class go?

    if incorrect > 0:

        print(
            "Misclassified as:"
        )

        errors = []

        for predicted_id in range(
            NUM_CLASSES
        ):

            if predicted_id == class_id:
                continue

            count = cm[
                class_id,
                predicted_id
            ]

            if count > 0:

                errors.append(
                    (
                        count,
                        CLASS_NAMES[
                            predicted_id
                        ]
                    )
                )

        errors.sort(
            reverse=True
        )

        for count, predicted_name in errors:

            percentage = (
                count / actual_count * 100
            )

            print(
                f"  → {predicted_name:<15}"
                f"{count:>8,} "
                f"({percentage:.2f}%)"
            )


# ============================================================
# FALSE PREDICTION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FALSE PREDICTION ANALYSIS")
print("=" * 80)

print(
    "\nFor each predicted class, we show where "
    "the predictions actually came from."
)

for predicted_id, predicted_name in enumerate(
    CLASS_NAMES
):

    predicted_count = np.sum(
        y_pred == predicted_id
    )

    correct = cm[
        predicted_id,
        predicted_id
    ]

    false_predictions = (
        predicted_count - correct
    )

    precision = (
        correct / predicted_count
        if predicted_count > 0
        else 0
    )

    print("\n" + "-" * 70)

    print(
        f"Predicted: {predicted_name}"
    )

    print(
        f"Total predicted     : "
        f"{predicted_count:,}"
    )

    print(
        f"Correct             : "
        f"{correct:,}"
    )

    print(
        f"False predictions   : "
        f"{false_predictions:,}"
    )

    print(
        f"Precision           : "
        f"{precision:.4f}"
    )

    if false_predictions > 0:

        print(
            "Actually belonged to:"
        )

        sources = []

        for actual_id in range(
            NUM_CLASSES
        ):

            if actual_id == predicted_id:
                continue

            count = cm[
                actual_id,
                predicted_id
            ]

            if count > 0:

                sources.append(
                    (
                        count,
                        CLASS_NAMES[
                            actual_id
                        ]
                    )
                )

        sources.sort(
            reverse=True
        )

        for count, actual_name in sources:

            percentage = (
                count /
                false_predictions *
                100
            )

            print(
                f"  ← {actual_name:<15}"
                f"{count:>8,} "
                f"({percentage:.2f}%)"
            )


# ============================================================
# TOP CONFUSION PAIRS
# ============================================================

print("\n" + "=" * 80)
print("TOP CONFUSION PAIRS")
print("=" * 80)

pairs = []

for actual_id in range(
    NUM_CLASSES
):

    for predicted_id in range(
        NUM_CLASSES
    ):

        if actual_id == predicted_id:
            continue

        count = cm[
            actual_id,
            predicted_id
        ]

        if count > 0:

            pairs.append(
                (
                    count,
                    CLASS_NAMES[actual_id],
                    CLASS_NAMES[predicted_id]
                )
            )

pairs.sort(
    reverse=True
)

print(
    "\nActual → Predicted"
)

for count, actual, predicted in pairs[:20]:

    print(
        f"{actual:<18} → "
        f"{predicted:<18} "
        f"{count:>8,}"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION DISTRIBUTION")
print("=" * 80)

prediction_counts = Counter(
    y_pred.tolist()
)

for class_id, name in enumerate(
    CLASS_NAMES
):

    count = prediction_counts.get(
        class_id,
        0
    )

    percentage = (
        count / len(y_pred) * 100
    )

    print(
        f"{name:<18}"
        f"{count:>10,} "
        f"{percentage:>7.2f}%"
    )


# ============================================================
# TOTAL ERRORS
# ============================================================

total_errors = np.sum(
    y_test != y_pred
)

total_correct = np.sum(
    y_test == y_pred
)

print("\n" + "=" * 80)
print("OVERALL ERROR SUMMARY")
print("=" * 80)

print(
    f"\nCorrect predictions : "
    f"{total_correct:,}"
)

print(
    f"Incorrect predictions: "
    f"{total_errors:,}"
)

print(
    f"Total test samples  : "
    f"{len(y_test):,}"
)

print(
    f"Error rate          : "
    f"{total_errors / len(y_test):.4f}"
)

print("\n" + "=" * 80)
print("ERROR ANALYSIS COMPLETE")
print("=" * 80)