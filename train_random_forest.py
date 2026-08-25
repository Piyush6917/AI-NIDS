import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = Path("data/final/features")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("AI-NIDS - RANDOM FOREST BASELINE")
print("=" * 80)

print("\nLoading training data...")

X_train = np.load(
    FEATURE_DIR / "X_train.npy"
)

y_train = np.load(
    FEATURE_DIR / "y_train_binary.npy"
)

print(f"Training shape: {X_train.shape}")

print("\nLoading validation data...")

X_val = np.load(
    FEATURE_DIR / "X_validation.npy"
)

y_val = np.load(
    FEATURE_DIR / "y_validation_binary.npy"
)

print(f"Validation shape: {X_val.shape}")

print("\nLoading test data...")

X_test = np.load(
    FEATURE_DIR / "X_test.npy"
)

y_test = np.load(
    FEATURE_DIR / "y_test_binary.npy"
)

print(f"Test shape: {X_test.shape}")


# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 80)

unique, counts = np.unique(
    y_train,
    return_counts=True
)

for label, count in zip(unique, counts):

    name = "BENIGN" if label == 0 else "ATTACK"

    print(
        f"{name:<10} : {count:,}"
    )


# ============================================================
# BALANCED TRAINING SAMPLE
# ============================================================

print("\n" + "=" * 80)
print("CREATING BALANCED TRAINING SAMPLE")
print("=" * 80)

rng = np.random.default_rng(42)

benign_indices = np.where(
    y_train == 0
)[0]

attack_indices = np.where(
    y_train == 1
)[0]

print(
    f"Available BENIGN : {len(benign_indices):,}"
)

print(
    f"Available ATTACK : {len(attack_indices):,}"
)


# Use all attack samples
# and an equal number of benign samples

sample_size = min(
    len(benign_indices),
    len(attack_indices)
)

selected_benign = rng.choice(
    benign_indices,
    size=sample_size,
    replace=False
)

selected_attack = rng.choice(
    attack_indices,
    size=sample_size,
    replace=False
)

selected_indices = np.concatenate(
    [
        selected_benign,
        selected_attack
    ]
)

# Shuffle
rng.shuffle(selected_indices)

X_train_balanced = X_train[
    selected_indices
]

y_train_balanced = y_train[
    selected_indices
]

print(
    f"\nBalanced training size: "
    f"{len(y_train_balanced):,}"
)

print(
    f"BENIGN : "
    f"{np.sum(y_train_balanced == 0):,}"
)

print(
    f"ATTACK : "
    f"{np.sum(y_train_balanced == 1):,}"
)


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

print("\nTraining started...")

model.fit(
    X_train_balanced,
    y_train_balanced
)

print("Training completed.")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)

val_predictions = model.predict(
    X_val
)

val_probabilities = model.predict_proba(
    X_val
)[:, 1]

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_recall = recall_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    zero_division=0
)

val_auc = roc_auc_score(
    y_val,
    val_probabilities
)

print(
    f"Accuracy  : {val_accuracy:.4f}"
)

print(
    f"Precision : {val_precision:.4f}"
)

print(
    f"Recall    : {val_recall:.4f}"
)

print(
    f"F1-score  : {val_f1:.4f}"
)

print(
    f"ROC-AUC   : {val_auc:.4f}"
)


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 80)
print("TEST RESULTS")
print("=" * 80)

test_predictions = model.predict(
    X_test
)

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1-score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\n                Predicted")
print("              BENIGN  ATTACK")
print(
    f"Actual BENIGN  {cm[0][0]:>8,}  {cm[0][1]:>8,}"
)
print(
    f"Actual ATTACK  {cm[1][0]:>8,}  {cm[1][1]:>8,}"
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
    "random_forest_binary.joblib"
)

joblib.dump(
    model,
    model_file
)

print("\n" + "=" * 80)
print("MODEL SAVED")
print("=" * 80)

print(
    model_file.resolve()
)

print("\nRandom Forest baseline complete.")