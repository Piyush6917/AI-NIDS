import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("AI-NIDS - CORRELATION & REDUNDANCY ANALYSIS")
print("=" * 80)

samples = []

for file in csv_files:

    print(f"Loading: {file.name}")

    df = pd.read_csv(
        file,
        nrows=20000,
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    samples.append(df)

data = pd.concat(samples, ignore_index=True)

# Remove label
X = data.drop(columns=["Label"], errors="ignore")

# Keep numeric columns
X = X.select_dtypes(include=np.number)

# Replace infinity
X = X.replace([np.inf, -np.inf], np.nan)

# Calculate correlation
correlation = X.corr()

# Find highly correlated pairs
threshold = 0.98

pairs = []

columns = correlation.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        corr_value = correlation.iloc[i, j]

        if pd.notna(corr_value) and abs(corr_value) >= threshold:

            pairs.append(
                (
                    columns[i],
                    columns[j],
                    corr_value
                )
            )

print("\n" + "=" * 80)
print(f"HIGHLY CORRELATED FEATURE PAIRS (|correlation| >= {threshold})")
print("=" * 80)

if not pairs:

    print("No highly correlated pairs found.")

else:

    pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True
    )

    for feature1, feature2, corr in pairs:

        print(
            f"{feature1:<40} "
            f"{feature2:<40} "
            f"{corr:.4f}"
        )

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS COMPLETE")
print("=" * 80)