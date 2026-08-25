import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("data/final/features")
OUTPUT_DIR = Path("data/final/multiclass")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We exclude "Other" because it has:
# Training    : 11
# Validation  : 0
# Test        : 0
#
# It cannot be meaningfully evaluated as a supervised class.

CLASS_NAMES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration"
]

CLASS_TO_ID = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

print("=" * 80)
print("AI-NIDS - MULTI-CLASS LABEL PREPARATION")
print("=" * 80)

print("\nClass mapping:")

for name, idx in CLASS_TO_ID.items():
    print(f"  {idx} -> {name}")


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

splits = [
    ("train", "X_train.npy", "y_train_family.npy"),
    ("validation", "X_validation.npy", "y_validation_family.npy"),
    ("test", "X_test.npy", "y_test_family.npy")
]

for split_name, x_file, y_file in splits:

    print("\n" + "=" * 80)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 80)

    X = np.load(
        FEATURE_DIR / x_file
    )

    y = np.load(
        FEATURE_DIR / y_file,
        allow_pickle=True
    )

    print(f"Original rows: {len(y):,}")

    original_counts = Counter(y.tolist())

    print("\nOriginal distribution:")

    for label, count in original_counts.items():
        print(f"  {str(label):15} {count:>10,}")


    # --------------------------------------------------------
    # REMOVE "Other"
    # --------------------------------------------------------

    valid_mask = np.isin(
        y,
        CLASS_NAMES
    )

    X_clean = X[valid_mask]
    y_clean = y[valid_mask]


    # --------------------------------------------------------
    # ENCODE LABELS
    # --------------------------------------------------------

    y_encoded = np.array(
        [
            CLASS_TO_ID[label]
            for label in y_clean
        ],
        dtype=np.int64
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / f"X_{split_name}.npy",
        X_clean
    )

    np.save(
        OUTPUT_DIR / f"y_{split_name}.npy",
        y_encoded
    )


    print(
        f"\nRows after removing 'Other': "
        f"{len(y_encoded):,}"
    )

    print("\nEncoded distribution:")

    encoded_counts = Counter(
        y_encoded.tolist()
    )

    for class_id in range(len(CLASS_NAMES)):

        print(
            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:15} "
            f"{encoded_counts.get(class_id, 0):>10,}"
        )


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

mapping_file = OUTPUT_DIR / "class_mapping.txt"

with open(
    mapping_file,
    "w",
    encoding="utf-8"
) as f:

    for idx, name in enumerate(CLASS_NAMES):
        f.write(
            f"{idx} = {name}\n"
        )


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING CLASS WEIGHTS")
print("=" * 80)

y_train = np.load(
    OUTPUT_DIR / "y_train.npy"
)

train_counts = np.bincount(
    y_train,
    minlength=len(CLASS_NAMES)
)

total = len(y_train)
num_classes = len(CLASS_NAMES)

# Balanced class weights:
#
# weight = total / (number_of_classes * class_count)

class_weights = (
    total /
    (num_classes * train_counts)
)

print("\nClass weights:")

for idx, weight in enumerate(class_weights):

    print(
        f"  {idx} - "
        f"{CLASS_NAMES[idx]:15} "
        f"count={train_counts[idx]:>10,} "
        f"weight={weight:.4f}"
    )


np.save(
    OUTPUT_DIR / "class_weights.npy",
    class_weights
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("MULTI-CLASS PREPARATION COMPLETE")
print("=" * 80)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nFiles created:")

print("  X_train.npy")
print("  y_train.npy")
print("  X_validation.npy")
print("  y_validation.npy")
print("  X_test.npy")
print("  y_test.npy")
print("  class_weights.npy")
print("  class_mapping.txt")

print("\nReady for multi-class Transformer training.")