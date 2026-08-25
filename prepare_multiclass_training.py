import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("data/final/multiclass")
OUTPUT_DIR = Path("data/final/multiclass_training")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Number of samples per class for the first experiment.
# We deliberately keep this moderate because the minority
# classes are extremely small.
TARGET_SAMPLES = {
    0: 50_000,   # BENIGN
    1: 50_000,   # DoS_DDoS
    2: 50_000,   # PortScan
    3: 7_387,    # BruteForce
    4: 1_705,    # WebAttack
    5: 1_558,    # Bot
    6: 27        # Infiltration
}

CLASS_NAMES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration"
]

rng = np.random.default_rng(SEED)

print("=" * 80)
print("AI-NIDS - CONTROLLED MULTI-CLASS TRAINING PREPARATION")
print("=" * 80)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

X = np.load(
    INPUT_DIR / "X_train.npy"
)

y = np.load(
    INPUT_DIR / "y_train.npy"
)

print(f"\nOriginal training rows: {len(y):,}")

print("\nOriginal distribution:")

for class_id, name in enumerate(CLASS_NAMES):

    count = np.sum(y == class_id)

    print(
        f"{class_id} - {name:15} "
        f"{count:>10,}"
    )


# ============================================================
# CONTROLLED SAMPLING
# ============================================================

print("\n" + "=" * 80)
print("CREATING CONTROLLED TRAINING SUBSET")
print("=" * 80)

selected_indices = []

for class_id, target in TARGET_SAMPLES.items():

    indices = np.where(
        y == class_id
    )[0]

    available = len(indices)

    if available <= target:

        chosen = indices

    else:

        chosen = rng.choice(
            indices,
            size=target,
            replace=False
        )

    selected_indices.extend(
        chosen.tolist()
    )

    print(
        f"{CLASS_NAMES[class_id]:15} "
        f"available={available:>10,} "
        f"selected={len(chosen):>10,}"
    )


# ============================================================
# SHUFFLE
# ============================================================

selected_indices = np.array(
    selected_indices,
    dtype=np.int64
)

rng.shuffle(
    selected_indices
)

X_selected = X[
    selected_indices
]

y_selected = y[
    selected_indices
]


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_selected
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_selected
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("FINAL TRAINING DISTRIBUTION")
print("=" * 80)

counts = np.bincount(
    y_selected,
    minlength=len(CLASS_NAMES)
)

for class_id, name in enumerate(CLASS_NAMES):

    percentage = (
        counts[class_id]
        / len(y_selected)
        * 100
    )

    print(
        f"{class_id} - "
        f"{name:15} "
        f"{counts[class_id]:>10,} "
        f"{percentage:>7.2f}%"
    )


print(
    f"\nTotal selected samples: "
    f"{len(y_selected):,}"
)


# ============================================================
# MODERATED CLASS WEIGHTS
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING MODERATED CLASS WEIGHTS")
print("=" * 80)

# We use inverse square-root frequency rather than
# raw inverse frequency.
#
# This prevents extremely rare classes from receiving
# enormous loss weights.

frequency = counts / len(y_selected)

weights = 1.0 / np.sqrt(
    frequency
)

# Normalize weights so their mean is approximately 1.
weights = (
    weights /
    weights.mean()
)

print("\nClass weights:")

for class_id, name in enumerate(CLASS_NAMES):

    print(
        f"{class_id} - "
        f"{name:15} "
        f"weight={weights[class_id]:.4f}"
    )


np.save(
    OUTPUT_DIR / "class_weights.npy",
    weights.astype(np.float32)
)


# ============================================================
# SAVE MAPPING
# ============================================================

with open(
    OUTPUT_DIR / "class_mapping.txt",
    "w",
    encoding="utf-8"
) as f:

    for class_id, name in enumerate(CLASS_NAMES):

        f.write(
            f"{class_id} = {name}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("CONTROLLED MULTI-CLASS PREPARATION COMPLETE")
print("=" * 80)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_DIR.resolve()}"
)

print("\nFiles created:")

print("  X_train.npy")
print("  y_train.npy")
print("  class_weights.npy")
print("  class_mapping.txt")

print("\nReady for multi-class Transformer training.")