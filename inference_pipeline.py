import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
FEATURE_DIR = BASE_DIR / "data" / "final" / "features"
MULTICLASS_DIR = BASE_DIR / "data" / "final" / "multiclass"

BINARY_MODEL = MODEL_DIR / "transformer_binary_gpu.pth"
MULTICLASS_MODEL = MODEL_DIR / "transformer_multiclass_gpu_v2.pth"

SCALER_FILE = FEATURE_DIR / "standard_scaler.joblib"

X_TEST_FILE = FEATURE_DIR / "X_test.npy"
Y_TEST_BINARY_FILE = FEATURE_DIR / "y_test_binary.npy"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BINARY_THRESHOLD = 0.80

CLASS_NAMES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration"
]


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class NetworkTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        num_classes,
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

            nn.Dropout(
                dropout
            ),

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

        x = x.mean(
            dim=1
        )

        x = self.norm(x)

        return self.classifier(x)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("AI-NIDS - REAL FLOW INFERENCE TEST")
print("=" * 70)

print(
    f"\nDevice: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


print("\nLoading StandardScaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("Scaler loaded.")


# ------------------------------------------------------------
# Binary model
# ------------------------------------------------------------

print("\nLoading binary Transformer...")

binary_model = NetworkTransformer(
    num_features=61,
    num_classes=2
).to(DEVICE)

binary_checkpoint = torch.load(
    BINARY_MODEL,
    map_location=DEVICE,
    weights_only=False
)

binary_model.load_state_dict(
    binary_checkpoint["model_state_dict"]
)

binary_model.eval()

print("Binary Transformer loaded.")


# ------------------------------------------------------------
# Multi-class model
# ------------------------------------------------------------

print("\nLoading multi-class Transformer V2...")

multiclass_model = NetworkTransformer(
    num_features=61,
    num_classes=7
).to(DEVICE)

multiclass_checkpoint = torch.load(
    MULTICLASS_MODEL,
    map_location=DEVICE,
    weights_only=False
)

multiclass_model.load_state_dict(
    multiclass_checkpoint["model_state_dict"]
)

multiclass_model.eval()

print("Multi-class Transformer V2 loaded.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

X_test = np.load(
    X_TEST_FILE
).astype(np.float32)

y_binary = np.load(
    Y_TEST_BINARY_FILE
).astype(np.int64)

print(
    f"Test samples: {len(X_test):,}"
)

print(
    f"Features: {X_test.shape[1]}"
)


# ============================================================
# INFERENCE FUNCTION
# ============================================================

def predict_flow(features):

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if features.ndim == 1:
        features = features.reshape(
            1,
            -1
        )

    if features.shape[1] != 61:

        raise ValueError(
            f"Expected 61 features, "
            f"got {features.shape[1]}"
        )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # X_test.npy is ALREADY standardized.
    #
    # Therefore we do NOT apply scaler.transform()
    # to these test rows.
    #
    # For future raw network-flow input, scaling must
    # happen before model inference.
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        features
    ).to(
        DEVICE
    )

    with torch.no_grad():

        # ====================================================
        # BINARY MODEL
        # ====================================================

        binary_logits = binary_model(
            tensor
        )

        binary_probabilities = torch.softmax(
            binary_logits,
            dim=1
        )

        attack_probability = (
            binary_probabilities[:, 1]
            .item()
        )

        if attack_probability >= BINARY_THRESHOLD:

            binary_prediction = "ATTACK"

        else:

            binary_prediction = "BENIGN"


        # ====================================================
        # MULTI-CLASS MODEL
        # ====================================================

        multiclass_logits = multiclass_model(
            tensor
        )

        multiclass_probabilities = torch.softmax(
            multiclass_logits,
            dim=1
        )

        multiclass_prediction = torch.argmax(
            multiclass_probabilities,
            dim=1
        ).item()

        family_confidence = (
            multiclass_probabilities[
                0,
                multiclass_prediction
            ].item()
        )

        family_name = CLASS_NAMES[
            multiclass_prediction
        ]


    # ========================================================
    # FINAL RESULT
    # ========================================================

    if binary_prediction == "BENIGN":

        final_family = "BENIGN"

        severity = "NONE"

    else:

        final_family = family_name

        if final_family in [
            "DoS_DDoS",
            "WebAttack",
            "Bot",
            "Infiltration"
        ]:

            severity = "HIGH"

        elif final_family in [
            "PortScan",
            "BruteForce"
        ]:

            severity = "MEDIUM"

        else:

            severity = "LOW"


    return {
        "binary_prediction":
            binary_prediction,

        "attack_probability":
            attack_probability,

        "attack_threshold":
            BINARY_THRESHOLD,

        "attack_family":
            final_family,

        "family_confidence":
            family_confidence,

        "severity":
            severity
    }


# ============================================================
# TEST BENIGN ROW
# ============================================================

print("\n" + "=" * 70)
print("TEST 1 - KNOWN BENIGN FLOW")
print("=" * 70)

benign_indices = np.where(
    y_binary == 0
)[0]

benign_index = benign_indices[0]

benign_result = predict_flow(
    X_test[benign_index]
)

print(
    f"\nActual label       : BENIGN"
)

print(
    f"Predicted          : "
    f"{benign_result['binary_prediction']}"
)

print(
    f"Attack probability : "
    f"{benign_result['attack_probability']:.4f}"
)

print(
    f"Attack threshold   : "
    f"{benign_result['attack_threshold']:.2f}"
)

print(
    f"Attack family      : "
    f"{benign_result['attack_family']}"
)

print(
    f"Family confidence  : "
    f"{benign_result['family_confidence']:.4f}"
)

print(
    f"Severity           : "
    f"{benign_result['severity']}"
)


# ============================================================
# TEST ATTACK ROW
# ============================================================

print("\n" + "=" * 70)
print("TEST 2 - KNOWN ATTACK FLOW")
print("=" * 70)

attack_indices = np.where(
    y_binary == 1
)[0]

attack_index = attack_indices[0]

attack_result = predict_flow(
    X_test[attack_index]
)

print(
    f"\nActual label       : ATTACK"
)

print(
    f"Predicted          : "
    f"{attack_result['binary_prediction']}"
)

print(
    f"Attack probability : "
    f"{attack_result['attack_probability']:.4f}"
)

print(
    f"Attack threshold   : "
    f"{attack_result['attack_threshold']:.2f}"
)

print(
    f"Attack family      : "
    f"{attack_result['attack_family']}"
)

print(
    f"Family confidence  : "
    f"{attack_result['family_confidence']:.4f}"
)

print(
    f"Severity           : "
    f"{attack_result['severity']}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("INFERENCE TEST COMPLETE")
print("=" * 70)


# ============================================================
# TEST ALL 7 CLASSES
# ============================================================

print("\n" + "=" * 70)
print("7-CLASS INFERENCE VALIDATION")
print("=" * 70)

# Load actual multi-class labels
Y_FAMILY_FILE = MULTICLASS_DIR / "y_test.npy"

y_family = np.load(
    Y_FAMILY_FILE
).astype(np.int64)

print("\nTesting one sample from each class...\n")

for class_id, class_name in enumerate(CLASS_NAMES):

    indices = np.where(
        y_family == class_id
    )[0]

    if len(indices) == 0:
        print(
            f"{class_name:<15} → NO SAMPLE"
        )
        continue

    index = indices[0]

    result = predict_flow(
        X_test[index]
    )

    predicted = result["attack_family"]

    confidence = result["family_confidence"]

    binary = result["binary_prediction"]

    status = (
        "✓ CORRECT"
        if predicted == class_name
        else "✗ INCORRECT"
    )

    print(
        f"{class_name:<15} → "
        f"{predicted:<15} | "
        f"Binary: {binary:<7} | "
        f"Confidence: {confidence:.4f} | "
        f"{status}"
    )


print("\n" + "=" * 70)
print("7-CLASS VALIDATION COMPLETE")
print("=" * 70)

# ============================================================
# DIRECT MULTI-CLASS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DIRECT MULTI-CLASS VALIDATION")
print("=" * 70)

print(
    "\nBinary model is being bypassed."
)

for class_id, class_name in enumerate(CLASS_NAMES):

    indices = np.where(
        y_family == class_id
    )[0]

    if len(indices) == 0:
        print(
            f"{class_name:<15} → NO SAMPLE"
        )
        continue

    index = indices[0]

    features = X_test[index]

    tensor = torch.from_numpy(
        features
    ).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = multiclass_model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_id = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_id
        ].item()

    predicted_name = CLASS_NAMES[
        predicted_id
    ]

    status = (
        "✓ CORRECT"
        if predicted_id == class_id
        else "✗ INCORRECT"
    )

    print(
        f"{class_name:<15} → "
        f"{predicted_name:<15} | "
        f"Confidence: {confidence:.4f} | "
        f"{status}"
    )

print("\n" + "=" * 70)
print("DIRECT MULTI-CLASS VALIDATION COMPLETE")
print("=" * 70)
y_family = np.load(
    MULTICLASS_DIR / "y_test.npy"
).astype(np.int64)