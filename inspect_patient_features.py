from pathlib import Path
import pandas as pd

folder = Path("backend/dataset/armd_ecuh")

files = [
    "00_microbiology_cultures_cohort.csv",
    "05_microbiology_cultures_demographics.csv",
    "06_microbiology_cultures_labs.csv",
    "013_microbiology_cultures_vitals.csv",
    "012_microbiology_cultures_ward_info.csv",
    "07_microbiology_cultures_microbial_resistance.csv",
]

for name in files:
    path = folder / name
    print("\n" + "=" * 70)
    print(name)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    sample = pd.read_csv(path, nrows=2, dtype=str)
    print("Columns:")
    print(sample.columns.tolist())
    print("Sample:")
    print(sample.head(1).to_string(index=False))
