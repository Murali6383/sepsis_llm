from pathlib import Path
import pandas as pd

root = Path("backend/dataset/armd_ecuh")
preview = root / "processed" / "antibiotic_patient_join_preview.csv"

key_cols = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
]

files = [
    "00_microbiology_cultures_cohort.csv",
    "05_microbiology_cultures_demographics.csv",
    "06_microbiology_cultures_labs.csv",
    "013_microbiology_cultures_vitals.csv",
]

print("=" * 65)
print("SOURCE KEY QUALITY CHECK")
print("=" * 65)

for filename in files:
    path = root / filename
    if not path.exists():
        print(f"\nMISSING FILE: {filename}")
        continue

    header = pd.read_csv(path, nrows=0)
    keys = [c for c in key_cols if c in header.columns]

    df = pd.read_csv(
        path,
        usecols=keys,
        dtype=str,
        low_memory=False,
    )

    for col in keys:
        df[col] = df[col].replace(
            ["Null", "null", "NULL", ""], pd.NA
        )

    print(f"\n{filename}")
    print("Rows:", f"{len(df):,}")
    print("Missing key rows:", int(df[keys].isna().any(axis=1).sum()))
    print("Duplicate key rows:", int(df.duplicated(keys).sum()))
    print("Unique key combinations:", f"{len(df.drop_duplicates(keys)):,}")

print("\n" + "=" * 65)
print("PREVIEW FEATURE AVAILABILITY")
print("=" * 65)

df = pd.read_csv(
    preview,
    usecols=[
        "susceptibility",
        "organism",
        "antibiotic",
        "demo_age",
        "lab_Period_Day",
        "lab_first_cr",
        "lab_last_cr",
        "lab_first_wbc",
        "lab_last_wbc",
        "vital_first_heartrate",
        "vital_last_heartrate",
    ],
    dtype=str,
    low_memory=False,
)

for col in df.columns:
    df[col] = df[col].replace(
        ["Null", "null", "NULL", ""], pd.NA
    )

for col in df.columns:
    if col != "susceptibility":
        missing = df[col].isna().mean() * 100
        print(f"{col}: missing {missing:.1f}%")

print("\nPeriod_Day sample values:")
print(df["lab_Period_Day"].value_counts(dropna=False).head(15))

print("\nAntibiotic examples:")
print(sorted(df["antibiotic"].dropna().unique())[:20])

print("\nCheck complete.")
