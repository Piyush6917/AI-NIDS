import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

csv_files = list(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("AI-NIDS - CROSS-FILE DUPLICATE ANALYSIS")
print("=" * 80)

# Store hashes of rows from each file
file_hashes = {}

for file in csv_files:

    print(f"\nProcessing: {file.name}")

    hashes = set()

    for chunk in pd.read_csv(
        file,
        chunksize=100000,
        low_memory=False
    ):

        chunk.columns = chunk.columns.str.strip()

        # Remove label so we compare network-flow features
        features = chunk.drop(
            columns=["Label"],
            errors="ignore"
        )

        # Create row hashes
        row_hashes = pd.util.hash_pandas_object(
            features,
            index=False
        )

        hashes.update(row_hashes)

    file_hashes[file.name] = hashes

    print(f"Unique feature hashes: {len(hashes):,}")


print("\n" + "=" * 80)
print("CROSS-FILE OVERLAPS")
print("=" * 80)

names = list(file_hashes.keys())

total_cross_file_overlap = 0

for i in range(len(names)):

    for j in range(i + 1, len(names)):

        file1 = names[i]
        file2 = names[j]

        overlap = (
            file_hashes[file1]
            & file_hashes[file2]
        )

        if overlap:

            print(
                f"\n{file1}"
                f"\n  ↕"
                f"\n{file2}"
                f"\nCommon feature rows: {len(overlap):,}"
            )

            total_cross_file_overlap += len(overlap)


print("\n" + "=" * 80)
print(
    f"Total reported cross-file overlaps: "
    f"{total_cross_file_overlap:,}"
)
print("=" * 80)