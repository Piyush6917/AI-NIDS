import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path("data/final/final_dataset.csv")
SPLIT_DIR = Path("data/final/splits")
FEATURE_DIR = Path("data/final/features")

FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD SPLIT INDICES
# ============================================================

train_indices = np.load(
    SPLIT_DIR / "train_indices.npy"
)

val_indices = np.load(
    SPLIT_DIR / "validation_indices.npy"
)

test_indices = np.load(
    SPLIT_DIR / "test_indices.npy"
)

train_set = set(train_indices)
val_set = set(val_indices)
test_set = set(test_indices)

print("=" * 80)
print("AI-NIDS - FEATURE PREPARATION")
print("=" * 80)

print(f"\nTrain rows      : {len(train_indices):,}")
print(f"Validation rows : {len(val_indices):,}")
print(f"Test rows       : {len(test_indices):,}")

# ============================================================
# DETERMINE FEATURE COLUMNS
# ============================================================

print("\nReading dataset header...")

header = pd.read_csv(
    DATA_FILE,
    nrows=0
)

target_columns = [
    "Label",
    "Binary_Label",
    "Attack_Family"
]

feature_columns = [
    column
    for column in header.columns
    if column not in target_columns
]

print(f"\nNumber of input features: {len(feature_columns)}")

# ============================================================
# PASS 1
# FIT SCALER ONLY ON TRAINING DATA
# ============================================================

print("\n" + "=" * 80)
print("PASS 1 - FITTING SCALER ON TRAINING DATA ONLY")
print("=" * 80)

scaler = StandardScaler()

current_position = 0
train_rows_seen = 0

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=feature_columns,
    chunksize=100000,
    low_memory=False
):

    chunk_size = len(chunk)

    chunk_start = current_position
    chunk_end = current_position + chunk_size

    # Find training rows inside this chunk
    local_train_positions = [
        i
        for i in range(
            chunk_size
        )
        if (chunk_start + i) in train_set
    ]

    if local_train_positions:

        train_chunk = chunk.iloc[
            local_train_positions
        ].copy()

        # Convert infinite values to NaN
        train_chunk = train_chunk.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Safety fallback
        train_chunk = train_chunk.fillna(0)

        scaler.partial_fit(train_chunk)

        train_rows_seen += len(train_chunk)

    current_position = chunk_end

    print(
        f"\rTraining rows processed: "
        f"{train_rows_seen:,}",
        end=""
    )

print("\n\nScaler fitted successfully.")

# ============================================================
# SAVE SCALER
# ============================================================

scaler_file = FEATURE_DIR / "standard_scaler.joblib"

joblib.dump(
    scaler,
    scaler_file
)

print(f"Scaler saved to:")
print(scaler_file)

# ============================================================
# PASS 2
# TRANSFORM AND SAVE DATA
# ============================================================

print("\n" + "=" * 80)
print("PASS 2 - TRANSFORMING DATA")
print("=" * 80)

# We will save each split as numpy arrays.

train_features = []
val_features = []
test_features = []

train_binary = []
val_binary = []
test_binary = []

train_family = []
val_family = []
test_family = []

# Read features + targets
use_columns = (
    feature_columns
    + [
        "Binary_Label",
        "Attack_Family"
    ]
)

current_position = 0

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=use_columns,
    chunksize=100000,
    low_memory=False
):

    chunk_size = len(chunk)

    chunk_start = current_position
    chunk_end = current_position + chunk_size

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = chunk[feature_columns].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    X_scaled = scaler.transform(X)

    # --------------------------------------------------------
    # Determine rows belonging to each split
    # --------------------------------------------------------

    train_positions = []
    val_positions = []
    test_positions = []

    for i in range(chunk_size):

        global_index = chunk_start + i

        if global_index in train_set:
            train_positions.append(i)

        elif global_index in val_set:
            val_positions.append(i)

        elif global_index in test_set:
            test_positions.append(i)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if train_positions:

        train_features.append(
            X_scaled[train_positions].astype(
                np.float32
            )
        )

        train_binary.append(
            chunk["Binary_Label"]
            .iloc[train_positions]
            .to_numpy(dtype=np.int8)
        )

        train_family.append(
            chunk["Attack_Family"]
            .iloc[train_positions]
            .astype(str)
            .to_numpy()
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if val_positions:

        val_features.append(
            X_scaled[val_positions].astype(
                np.float32
            )
        )

        val_binary.append(
            chunk["Binary_Label"]
            .iloc[val_positions]
            .to_numpy(dtype=np.int8)
        )

        val_family.append(
            chunk["Attack_Family"]
            .iloc[val_positions]
            .astype(str)
            .to_numpy()
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    if test_positions:

        test_features.append(
            X_scaled[test_positions].astype(
                np.float32
            )
        )

        test_binary.append(
            chunk["Binary_Label"]
            .iloc[test_positions]
            .to_numpy(dtype=np.int8)
        )

        test_family.append(
            chunk["Attack_Family"]
            .iloc[test_positions]
            .astype(str)
            .to_numpy()
        )

    current_position = chunk_end

    print(
        f"\rRows processed: "
        f"{current_position:,}",
        end=""
    )

print("\n")

# ============================================================
# COMBINE ARRAYS
# ============================================================

print("Combining arrays...")

X_train = np.concatenate(train_features)
X_val = np.concatenate(val_features)
X_test = np.concatenate(test_features)

y_train_binary = np.concatenate(train_binary)
y_val_binary = np.concatenate(val_binary)
y_test_binary = np.concatenate(test_binary)

y_train_family = np.concatenate(train_family)
y_val_family = np.concatenate(val_family)
y_test_family = np.concatenate(test_family)

# ============================================================
# SAVE
# ============================================================

np.save(
    FEATURE_DIR / "X_train.npy",
    X_train
)

np.save(
    FEATURE_DIR / "X_validation.npy",
    X_val
)

np.save(
    FEATURE_DIR / "X_test.npy",
    X_test
)

np.save(
    FEATURE_DIR / "y_train_binary.npy",
    y_train_binary
)

np.save(
    FEATURE_DIR / "y_validation_binary.npy",
    y_val_binary
)

np.save(
    FEATURE_DIR / "y_test_binary.npy",
    y_test_binary
)

np.save(
    FEATURE_DIR / "y_train_family.npy",
    y_train_family
)

np.save(
    FEATURE_DIR / "y_validation_family.npy",
    y_val_family
)

np.save(
    FEATURE_DIR / "y_test_family.npy",
    y_test_family
)

# ============================================================
# SUMMARY
# ============================================================

print("=" * 80)
print("FEATURE PREPARATION COMPLETE")
print("=" * 80)

print(f"\nX_train shape      : {X_train.shape}")
print(f"X_validation shape: {X_val.shape}")
print(f"X_test shape      : {X_test.shape}")

print("\nBinary labels:")
print(
    f"Train      : {y_train_binary.shape}"
)
print(
    f"Validation : {y_val_binary.shape}"
)
print(
    f"Test       : {y_test_binary.shape}"
)

print("\nFiles saved in:")
print(FEATURE_DIR.resolve())

print("=" * 80)