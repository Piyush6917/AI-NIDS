import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("data/final/features")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

TRAIN_SAMPLES = 100_000
BATCH_SIZE = 256

EPOCHS = 10
LEARNING_RATE = 1e-4

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2

DROPOUT = 0.1

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


print("=" * 80)
print("AI-NIDS - TRANSFORMER BINARY CLASSIFIER")
print("=" * 80)

print(f"\nDevice: {DEVICE}")
print(f"Training samples: {TRAIN_SAMPLES:,}")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading prepared data...")

X_train = np.load(
    FEATURE_DIR / "X_train.npy"
)

y_train = np.load(
    FEATURE_DIR / "y_train_binary.npy"
)

X_val = np.load(
    FEATURE_DIR / "X_validation.npy"
)

y_val = np.load(
    FEATURE_DIR / "y_validation_binary.npy"
)

X_test = np.load(
    FEATURE_DIR / "X_test.npy"
)

y_test = np.load(
    FEATURE_DIR / "y_test_binary.npy"
)

print(f"Full training data : {X_train.shape}")
print(f"Validation data     : {X_val.shape}")
print(f"Test data           : {X_test.shape}")


# ============================================================
# BALANCED TRAINING SUBSET
# ============================================================

print("\n" + "=" * 80)
print("CREATING BALANCED TRAINING SUBSET")
print("=" * 80)

rng = np.random.default_rng(SEED)

benign_indices = np.where(
    y_train == 0
)[0]

attack_indices = np.where(
    y_train == 1
)[0]

# Half of requested samples from each class
samples_per_class = TRAIN_SAMPLES // 2

selected_benign = rng.choice(
    benign_indices,
    size=samples_per_class,
    replace=False
)

selected_attack = rng.choice(
    attack_indices,
    size=samples_per_class,
    replace=False
)

selected_indices = np.concatenate(
    [
        selected_benign,
        selected_attack
    ]
)

rng.shuffle(selected_indices)

X_train_small = X_train[
    selected_indices
]

y_train_small = y_train[
    selected_indices
]

print(
    f"Selected training samples: "
    f"{len(y_train_small):,}"
)

print(
    f"BENIGN: "
    f"{np.sum(y_train_small == 0):,}"
)

print(
    f"ATTACK: "
    f"{np.sum(y_train_small == 1):,}"
)


# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_small,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_small,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)


# ============================================================
# DATASET / DATALOADER
# ============================================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ============================================================
# TRANSFORMER MODEL
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

        self.num_features = num_features
        self.d_model = d_model

        # Each scalar feature → embedding vector
        self.feature_embedding = nn.Linear(
            1,
            d_model
        )

        # Learnable feature identity embeddings
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

        # x shape:
        # [batch, 61]

        x = x.unsqueeze(-1)

        # [batch, 61, 1]
        # →
        # [batch, 61, d_model]

        x = self.feature_embedding(x)

        x = x + self.feature_position

        x = self.transformer(x)

        # Mean pooling across feature tokens
        x = x.mean(dim=1)

        x = self.norm(x)

        output = self.classifier(x)

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = NetworkTransformer(
    num_features=X_train.shape[1],
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


print("\n" + "=" * 80)
print("MODEL")
print("=" * 80)

print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(train_loader)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_predictions = []
    val_probabilities = []

    with torch.no_grad():

        # Process validation in batches
        for start in range(
            0,
            len(X_val_tensor),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(X_val_tensor)
            )

            batch_X = X_val_tensor[
                start:end
            ].to(DEVICE)

            outputs = model(
                batch_X
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

    val_predictions = np.array(
        val_predictions
    )

    val_probabilities = np.array(
        val_probabilities
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    val_auc = roc_auc_score(
        y_val,
        val_probabilities
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {average_loss:.4f} | "
        f"Val F1: {val_f1:.4f} | "
        f"Val Recall: {val_recall:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST EVALUATION")
print("=" * 80)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

model.eval()

test_predictions = []
test_probabilities = []

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

        batch_X = X_test_tensor[
            start:end
        ].to(DEVICE)

        outputs = model(
            batch_X
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        test_predictions.extend(
            predictions.cpu().numpy()
        )

        test_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

test_predictions = np.array(
    test_predictions
)

test_probabilities = np.array(
    test_probabilities
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    test_probabilities
)


print(
    f"\nAccuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print("\n                Predicted")
print("              BENIGN  ATTACK")

print(
    f"Actual BENIGN "
    f"{cm[0][0]:>9,}"
    f"{cm[0][1]:>9,}"
)

print(
    f"Actual ATTACK "
    f"{cm[1][0]:>9,}"
    f"{cm[1][1]:>9,}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "BENIGN",
            "ATTACK"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    MODEL_DIR /
    "transformer_binary_cpu.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_features": X_train.shape[1],
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS
    },
    model_file
)

print("\n" + "=" * 80)
print("MODEL SAVED")
print("=" * 80)

print(
    model_file.resolve()
)

print("\nTransformer training complete.")