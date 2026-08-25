import pandas as pd
from pathlib import Path

FILE = Path("data/final/final_dataset.csv")

print("=" * 80)
print("AI-NIDS - FINAL DATASET CHECK")
print("=" * 80)

# Read only a sample
df = pd.read_csv(
    FILE,
    nrows=10000,
    low_memory=False
)

print(f"\nColumns: {len(df.columns)}")

print("\nColumn names:")
for i, column in enumerate(df.columns, 1):
    print(f"{i:3}. {column}")

print("\nSample shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nBinary label distribution in sample:")
print(df["Binary_Label"].value_counts())

print("\nAttack family distribution in sample:")
print(df["Attack_Family"].value_counts())

print("\n" + "=" * 80)
print("FINAL DATASET CHECK COMPLETE")
print("=" * 80)