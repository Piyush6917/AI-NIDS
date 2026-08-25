import numpy as np

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

FEATURE_DIR = BASE_DIR / "data" / "final" / "features"
MULTICLASS_DIR = BASE_DIR / "data" / "final" / "multiclass"

# These should already exist from the cascade analysis
binary_probs = np.load(
    FEATURE_DIR / "binary_test_probabilities.npy"
)

y_family = np.load(
    MULTICLASS_DIR / "y_test.npy"
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

threshold = 0.80

print("=" * 70)
print("AI-NIDS - BINARY GATE BY ATTACK FAMILY")
print("=" * 70)

for class_id, class_name in enumerate(CLASS_NAMES):

    indices = np.where(
        y_family == class_id
    )[0]

    if len(indices) == 0:
        continue

    probabilities = binary_probs[
        indices
    ]

    detected = np.sum(
        probabilities >= threshold
    )

    missed = np.sum(
        probabilities < threshold
    )

    recall = (
        detected / len(indices)
    )

    print(
        f"\n{class_name}"
    )

    print(
        f"Total    : {len(indices):,}"
    )

    print(
        f"Detected : {detected:,}"
    )

    print(
        f"Rejected : {missed:,}"
    )

    print(
        f"Gate recall: {recall:.4f}"
    )