import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 70)
print("AI-NIDS - FULL DATASET LABEL ANALYSIS")
print("=" * 70)

overall_labels = {}

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    label_counts = {}

    # Read dataset in chunks
    for chunk in pd.read_csv(
        file,
        usecols=lambda column: column.strip().lower() == "label",
        chunksize=100000,
        low_memory=False
    ):

        # Remove spaces from column name
        chunk.columns = chunk.columns.str.strip()

        labels = chunk["Label"].astype(str).str.strip()

        counts = labels.value_counts()

        for label, count in counts.items():

            label_counts[label] = (
                label_counts.get(label, 0) + count
            )

            overall_labels[label] = (
                overall_labels.get(label, 0) + count
            )

    print("\nLabels in this file:")

    for label, count in sorted(
        label_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{label:<30} {count:,}")


print("\n" + "=" * 70)
print("OVERALL DATASET LABEL DISTRIBUTION")
print("=" * 70)

total_rows = sum(overall_labels.values())

for label, count in sorted(
    overall_labels.items(),
    key=lambda x: x[1],
    reverse=True
):

    percentage = (count / total_rows) * 100

    print(
        f"{label:<30} "
        f"{count:>12,} "
        f"{percentage:>8.2f}%"
    )

print("\n" + "=" * 70)
print(f"TOTAL ROWS: {total_rows:,}")
print("=" * 70)