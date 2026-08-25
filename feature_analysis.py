import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("AI-NIDS - FEATURE ANALYSIS")
print("=" * 80)

# We'll inspect a representative sample from each file
samples = []

for file in csv_files:

    print(f"\nLoading sample: {file.name}")

    df = pd.read_csv(
        file,
        nrows=20000,
        low_memory=False
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    samples.append(df)

# Combine samples
data = pd.concat(samples, ignore_index=True)

print("\n" + "=" * 80)
print("DATASET SAMPLE INFORMATION")
print("=" * 80)

print(f"Sample rows      : {len(data):,}")
print(f"Features + label : {len(data.columns)}")

print("\n" + "=" * 80)
print("DATA TYPES")
print("=" * 80)

print(data.dtypes.to_string())

print("\n" + "=" * 80)
print("UNIQUE VALUES PER COLUMN")
print("=" * 80)

for column in data.columns:

    unique_count = data[column].nunique(dropna=False)

    print(f"{column:<45} {unique_count:,}")

print("\n" + "=" * 80)
print("CONSTANT / NEAR-CONSTANT FEATURES")
print("=" * 80)

for column in data.columns:

    counts = data[column].value_counts(
        normalize=True,
        dropna=False
    )

    if len(counts) > 0:

        most_common_percentage = counts.iloc[0] * 100

        if most_common_percentage >= 99.9:

            print(
                f"{column:<45} "
                f"{most_common_percentage:.4f}% same value"
            )

print("\n" + "=" * 80)
print("NUMERIC FEATURE SUMMARY")
print("=" * 80)

numeric_columns = data.select_dtypes(
    include=np.number
).columns

summary = data[numeric_columns].describe().T

print(
    summary[
        [
            "count",
            "mean",
            "std",
            "min",
            "max"
        ]
    ].to_string()
)

print("\n" + "=" * 80)
print("FEATURE ANALYSIS COMPLETE")
print("=" * 80)