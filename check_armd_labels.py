from pathlib import Path
import pandas as pd

DATA_DIR = Path("backend/dataset/armd_ecuh")
FILE = DATA_DIR / "00_microbiology_cultures_cohort.csv"

if not FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {FILE.resolve()}")

print("Scanning complete culture dataset in chunks...")

label_counts = {}
organism_counts = {}
antibiotic_counts = {}
rows = 0
missing = 0

for chunk in pd.read_csv(
    FILE,
    usecols=["susceptibility", "organism", "antibiotic"],
    chunksize=200_000,
    low_memory=False,
):
    rows += len(chunk)
    missing += int(chunk["susceptibility"].isna().sum())

    for label, count in chunk["susceptibility"].fillna("<MISSING>").value_counts().items():
        label_counts[str(label)] = label_counts.get(str(label), 0) + int(count)

    for value, count in chunk["organism"].fillna("<MISSING>").value_counts().items():
        organism_counts[str(value)] = organism_counts.get(str(value), 0) + int(count)

    for value, count in chunk["antibiotic"].fillna("<MISSING>").value_counts().items():
        antibiotic_counts[str(value)] = antibiotic_counts.get(str(value), 0) + int(count)

print("\nTotal culture rows:", rows)
print("Missing susceptibility labels:", missing)

print("\n=== SUSCEPTIBILITY LABEL COUNTS ===")
for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"{label}: {count}")

print("\n=== TOP 20 ORGANISMS ===")
for name, count in sorted(organism_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"{name}: {count}")

print("\n=== TOP 20 ANTIBIOTICS ===")
for name, count in sorted(antibiotic_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"{name}: {count}")

print("\nLabel scan complete.")