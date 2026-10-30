from pathlib import Path
import pandas as pd

root = Path("backend/dataset/armd_ecuh")

for filename in [
    "06_microbiology_cultures_labs.csv",
    "013_microbiology_cultures_vitals.csv",
]:
    path = root / filename
    print("\n" + "=" * 65)
    print(filename)

    df = pd.read_csv(
        path,
        nrows=20000,
        dtype=str,
        low_memory=False,
    )

    keys = [
        c for c in [
            "anon_id",
            "pat_enc_csn_id_coded",
            "order_proc_id_coded",
        ]
        if c in df.columns
    ]

    print("Sample rows:", len(df))
    print("Duplicate keys in first 20,000 rows:", int(df.duplicated(keys).sum()))

    duplicated = df[df.duplicated(keys, keep=False)]
    if not duplicated.empty:
        print("\nExample duplicate records:")
        print(duplicated.head(6).to_string(index=False))

print("\nInspection finished. Original files unchanged.")
