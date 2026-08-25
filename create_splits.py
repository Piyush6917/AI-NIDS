import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path("data/final/final_dataset.csv")
SPLIT_DIR = Path("data/final/splits")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD ONLY TARGET COLUMNS
# ============================================================

print("=" * 80)
print("AI-NIDS - TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

print("\nReading target columns...")

labels = pd.read_csv(
    DATA_FILE,
    usecols=["Binary_Label", "Attack_Family"],
    low_memory=False
)

print(f"Total rows: {len(labels):,}")

# Row IDs
indices = np.arange(len(labels))

# ============================================================
# FIRST SPLIT
# 80% TRAIN
# 20% TEMP
# ============================================================

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=labels["Binary_Label"]
)

# ============================================================
# SECOND SPLIT
# 10% VALIDATION
# 10% TEST
# ============================================================

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=labels.iloc[temp_idx]["Binary_Label"]
)

# ============================================================
# SAVE INDICES
# ============================================================

np.save(
    SPLIT_DIR / "train_indices.npy",
    train_idx
)

np.save(
    SPLIT_DIR / "validation_indices.npy",
    val_idx
)

np.save(
    SPLIT_DIR / "test_indices.npy",
    test_idx
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("SPLIT SUMMARY")
print("=" * 80)

print(f"Training   : {len(train_idx):,}")
print(f"Validation : {len(val_idx):,}")
print(f"Test       : {len(test_idx):,}")

print(
    f"Total      : "
    f"{len(train_idx) + len(val_idx) + len(test_idx):,}"
)

# ============================================================
# BINARY DISTRIBUTION
# ============================================================

def show_distribution(name, idx):

    subset = labels.iloc[idx]

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    counts = subset["Binary_Label"].value_counts()

    total = len(subset)

    for label, count in counts.items():

        percentage = count / total * 100

        label_name = (
            "BENIGN"
            if label == 0
            else "ATTACK"
        )

        print(
            f"{label_name:<10}"
            f"{count:>12,}"
            f"{percentage:>8.2f}%"
        )


show_distribution("TRAIN", train_idx)
show_distribution("VALIDATION", val_idx)
show_distribution("TEST", test_idx)

print("\n" + "=" * 80)
print("SPLIT CREATION COMPLETE")
print("=" * 80)