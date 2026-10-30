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

    print("\n" + "=" * 70)
    print("FILE:", filename)
    print("=" * 70)

    if not path.exists():
        print("NOT FOUND:", path)
        continue

    try:
        # Read only selected columns and sample rows for initial inspection.
        if filename == "00_microbiology_cultures_cohort.csv":
            columns = [
                "anon_id",
                "pat_enc_csn_id_coded",
                "order_proc_id_coded",
                "organism",
                "antibiotic",
                "susceptibility",
                "was_positive",
            ]
            df = pd.read_csv(path, usecols=columns, nrows=100_000)

            print("Rows inspected:", len(df))
            print("Unique organisms:", df["organism"].nunique(dropna=True))
            print("Unique antibiotics:", df["antibiotic"].nunique(dropna=True))
            print("\nSusceptibility labels:")
            print(df["susceptibility"].value_counts(dropna=False).head(20))
            print("\nCulture positivity:")
            print(df["was_positive"].value_counts(dropna=False))

            print("\nMissing values:")
            print(df.isna().sum().to_string())

            print("\nDuplicate rows in sample:", df.duplicated().sum())

        elif filename == "07_microbiology_cultures_microbial_resistance.csv":
            columns = [
                "anon_id",
                "pat_enc_csn_id_coded",
                "order_proc_id_coded",
                "organism",
                "antibiotic",
                "resistant_time_to_culturetime",
            ]
            df = pd.read_csv(path, usecols=columns, nrows=100_000)

            print("Rows inspected:", len(df))
            print("\nResistance time values:")
            print(df["resistant_time_to_culturetime"].describe())
            print("\nMissing values:")
            print(df.isna().sum().to_string())
            print("\nSample rows:")
            print(df.head(5).to_string(index=False))

        else:
            df = pd.read_csv(path, nrows=5, low_memory=False)
            print("Columns:", df.columns.tolist())
            print("Sample rows:")
            print(df.head(2).to_string(index=False))

    except Exception as exc:
        print("ERROR:", exc)

print("\nInitial dataset inspection complete.")
print("Note: counts above are sample counts, not full-dataset totals.")