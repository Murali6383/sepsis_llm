from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "backend" / "dataset" / "armd_ecuh"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

cohort_file = DATA / "00_microbiology_cultures_cohort.csv"

# Only fields required for susceptibility prediction.
cohort_cols = [
    "anon_id", "pat_enc_csn_id_coded", "order_proc_id_coded",
    "order_time_jittered", "ordering_mode", "culture_description",
    "was_positive", "organism", "antibiotic", "susceptibility",
]

cohort = pd.read_csv(
    cohort_file,
    usecols=cohort_cols,
    dtype=str,
    low_memory=False,
)

def clean_nulls(df):
    for col in df.columns:
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].replace(
                ["Null", "null", "NULL", ""], pd.NA
            )
    return df

cohort = clean_nulls(cohort)

# Remove records without usable supervised labels.
cohort["susceptibility"] = cohort["susceptibility"].str.strip().str.title()
cohort = cohort[
    cohort["susceptibility"].isin(
        ["Susceptible", "Resistant", "Intermediate"]
    )
    & cohort["organism"].notna()
    & cohort["antibiotic"].notna()
    & cohort["anon_id"].notna()
].copy()

# Merge each source on the culture/order identifiers.
# Do not use resistant_time_to_culturetime as a target.
join_keys = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
]

def merge_source(base, filename, prefix, keep_cols=None):
    path = DATA / filename
    if not path.exists():
        print(f"Skipping missing file: {filename}")
        return base

    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)

    keys = [k for k in join_keys if k in available]
    if not keys:
        print(f"Skipping {filename}: no shared join keys")
        return base

    if keep_cols is None:
        keep_cols = [
            c for c in available
            if c not in join_keys and c != "order_time_jittered"
        ]

    selected = list(dict.fromkeys(keys + [
        c for c in keep_cols if c in available
    ]))

    source = pd.read_csv(
        path,
        usecols=selected,
        dtype=str,
        low_memory=False,
    )
    source = clean_nulls(source)

    # A many-to-one merge is required for these summary feature tables.
    # If duplicate keys exist, collapse them rather than multiplying rows.
    source = source.drop_duplicates(subset=keys, keep="first")

    rename = {
        c: f"{prefix}_{c}"
        for c in source.columns
        if c not in keys
    }
    source = source.rename(columns=rename)

    return base.merge(source, on=keys, how="left", validate="many_to_one")

print("Starting cohort:", len(cohort))

cohort = merge_source(
    cohort,
    "05_microbiology_cultures_demographics.csv",
    "demo",
)
cohort = merge_source(
    cohort,
    "06_microbiology_cultures_labs.csv",
    "lab",
)
cohort = merge_source(
    cohort,
    "013_microbiology_cultures_vitals.csv",
    "vital",
)

# Keep the output manageable and avoid using future information
# until the temporal meaning of Period_Day is verified.
# These summaries are retained for inspection only at this stage.
output_file = OUT / "antibiotic_patient_join_preview.csv"
cohort.head(50000).to_csv(output_file, index=False)

print("\nPreview saved:", output_file)
print("Rows in cohort:", len(cohort))
print("Columns in preview:", len(cohort.columns))
print("Labels:\n", cohort["susceptibility"].value_counts())
print("\nPreview columns:\n", cohort.columns.tolist())
print(
    "\nThis is a join preview only, NOT a validated training dataset. "
    "Check duplicate keys, timestamp availability, and feature timing "
    "before model training."
)
