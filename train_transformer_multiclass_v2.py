import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data/final/multiclass")
TRAIN_DIR = Path("data/final/multiclass_training")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "transformer_multiclass_gpu_v2.pth"

BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.0005

NUM_CLASSES = 7
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

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

# Softer class weights
CLASS_WEIGHTS = np.array([
    1.0,   # BENIGN
    1.0,   # DoS_DDoS
    1.2,   # PortScan
    1.5,   # BruteForce
    2.0,   # WebAttack
    2.0,   # Bot
    2.5    # Infiltration
], dtype=np.float32)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("AI-NIDS - MULTI-CLASS TRANSFORMER V2")
print("=" * 80)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

X_train = np.load(
    TRAIN_DIR / "X_train.npy"
).astype(np.float32)

y_train = np.load(
    TRAIN_DIR / "y_train.npy"
).astype(np.int64)

X_val = np.load(
    DATA_DIR / "X_validation.npy"
).astype(np.float32)

y_val = np.load(
    DATA_DIR / "y_validation.npy"
).astype(np.int64)

X_test = np.load(
    DATA_DIR / "X_test.npy"
).astype(np.float32)

y_test = np.load(
    DATA_DIR / "y_test.npy"
).astype(np.int64)

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")


# ============================================================
# DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

val_dataset = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val)
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
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
# CREATE MODEL
# ============================================================

model = NetworkTransformer(
    num_features=X_train.shape[1],
    num_classes=NUM_CLASSES,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

print("\n" + "=" * 80)
print("MODEL")
print("=" * 80)

print(model)

print(
    f"\nParameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# V2 CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 80)
print("V2 CLASS WEIGHTS")
print("=" * 80)

for i, name in enumerate(CLASS_NAMES):
    print(
        f"{i} - {name:<15} "
        f"weight={CLASS_WEIGHTS[i]:.2f}"
    )

weights_tensor = torch.tensor(
    CLASS_WEIGHTS,
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(
    weight=weights_tensor
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

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

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            zero_division=0
        )
    )

    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()

    weighted_f1 = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )[2]
    )

    return (
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        weighted_f1
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 80)
print("V2 TRAINING")
print("=" * 80)

best_macro_f1 = 0.0

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE,
            non_blocking=True
        )

        y_batch = y_batch.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            X_batch
        )

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X_batch.size(0)
        )

    epoch_loss = (
        running_loss /
        len(train_dataset)
    )

    (
        val_accuracy,
        val_precision,
        val_recall,
        val_macro_f1,
        val_weighted_f1
    ) = evaluate(
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f} | "
        f"Val Weighted F1: {val_weighted_f1:.4f}"
    )

    if val_macro_f1 > best_macro_f1:

        best_macro_f1 = val_macro_f1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "num_features":
                    X_train.shape[1],

                "num_classes":
                    NUM_CLASSES,

                "class_names":
                    CLASS_NAMES,

                "class_weights":
                    CLASS_WEIGHTS,

                "best_val_macro_f1":
                    best_macro_f1
            },
            MODEL_FILE
        )

        print(
            f"  → Best V2 model saved "
            f"(Macro F1: {best_macro_f1:.4f})"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("V2 TRAINING COMPLETE")
print("=" * 80)

print(
    f"\nBest Validation Macro F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"\nModel saved to:\n"
    f"{MODEL_FILE.resolve()}"
)

print(
    "\nNext step: evaluate the saved V2 model "
    "on the test set."
)