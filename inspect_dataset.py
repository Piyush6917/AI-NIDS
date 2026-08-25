import pandas as pd
from pathlib import Path
import numpy as np

# Dataset folder
DATA_DIR = Path("data/raw")

# Find all CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("AI-NIDS - CIC-IDS2017 DATASET INSPECTION")
print("=" * 70)

print(f"\nNumber of CSV files found: {len(csv_files)}")

for file in csv_files:
    print(f" - {file.name}")

print("\n" + "=" * 70)

# Inspect each file
for file in csv_files:

    print(f"\nFILE: {file.name}")
    print("-" * 70)

    # Read only first 10,000 rows
    df = pd.read_csv(
        file,
        nrows=10000,
        low_memory=False
    )

    print(f"Sample rows loaded : {len(df)}")
    print(f"Number of columns  : {len(df.columns)}")

    print("\nColumns:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i:3}. {column}")

    # Find label column
    label_columns = [
        col for col in df.columns
        if col.strip().lower() == "label"
    ]

    if label_columns:
        label_col = label_columns[0]

        print("\nLabel distribution in first 10,000 rows:")
        print(df[label_col].value_counts(dropna=False))

    # Missing values
    missing = df.isnull().sum()

    missing_columns = missing[missing > 0]

    print("\nMissing values:")
    if len(missing_columns) == 0:
        print("No missing values found in sample.")
    else:
        print(missing_columns)

    # Infinite values
    numeric_df = df.select_dtypes(include=np.number)

    infinite_count = np.isinf(numeric_df).sum().sum()

    print(f"\nInfinite numeric values: {infinite_count}")

    # Duplicate rows
    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows in sample: {duplicate_count}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)