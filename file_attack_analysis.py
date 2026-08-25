import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 90)
print("AI-NIDS - ATTACK DISTRIBUTION BY FILE")
print("=" * 90)

overall = {}

for file in csv_files:

    print("\n" + "=" * 90)
    print(f"FILE: {file.name}")
    print("=" * 90)

    file_counts = {}

    for chunk in pd.read_csv(
        file,
        usecols=lambda col: col.strip().lower() == "label",
        chunksize=100000,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        labels = chunk["Label"].astype(str).str.strip()

        counts = labels.value_counts()

        for label, count in counts.items():

            file_counts[label] = (
                file_counts.get(label, 0) + count
            )

            overall[label] = (
                overall.get(label, 0) + count
            )

    total = sum(file_counts.values())

    for label, count in sorted(
        file_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        percentage = count / total * 100

        print(
            f"{label:<35}"
            f"{count:>12,}"
            f"{percentage:>8.2f}%"
        )


print("\n" + "=" * 90)
print("ATTACK TYPES ACROSS DATASET")
print("=" * 90)

for label, count in sorted(
    overall.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{label:<35}{count:>12,}")

print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)