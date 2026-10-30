from pathlib import Path
import pandas as pd

DATA_DIR = Path("backend/dataset/armd_ecuh")
FILES = [
    "00_microbiology_cultures_cohort.csv",
    "05_microbiology_cultures_demographics.csv",
    "06_microbiology_cultures_labs.csv",
    "07_microbiology_cultures_microbial_resistance.csv",
    "013_microbiology_cultures_vitals.csv",
    "02_microbiology_cultures_antibiotic_class_exposure.csv",
    "03_microbiology_cultures_antibiotic_subtype_exposure.csv",
]

for filename in FILES:
    path = DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    try:
        # Read only 3 rows; do not load the entire large CSV.
        df = pd.read_csv(path, nrows=3, low_memory=False)

        print("\nColumns:")
        for column in df.columns:
            print(f"  - {column}")

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nSample rows:")
        print(df.head(2).to_string(index=False))

    except Exception as exc:
        print(f"Could not inspect file: {exc}")