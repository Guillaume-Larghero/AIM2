import os
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.join(
    os.path.dirname(__file__),
    "physionet.org", "files", "mimic-cxr-jpg", "2.1.0",
)
OUT = os.path.join(os.path.dirname(__file__), "mimic_cxr_manifest.csv")

LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

# --------------------------------------------------------------------------- #
# Load raw CSVs
# --------------------------------------------------------------------------- #
print("Loading CSVs...")
split_df  = pd.read_csv(os.path.join(ROOT, "mimic-cxr-2.0.0-split.csv.gz"))
meta_df   = pd.read_csv(os.path.join(ROOT, "mimic-cxr-2.0.0-metadata.csv.gz"),
                        usecols=["dicom_id", "ViewPosition"])
labels_df = pd.read_csv(os.path.join(ROOT, "mimic-cxr-2.0.0-chexpert.csv.gz"))

print(f"  Total DICOMs in split file : {len(split_df):,}")

# --------------------------------------------------------------------------- #
# Filter to AP / PA
# --------------------------------------------------------------------------- #
df = split_df.merge(meta_df, on="dicom_id", how="left")
df = df[df["ViewPosition"].isin({"AP", "PA"})].copy()
print(f"  After AP/PA filter         : {len(df):,} DICOMs")

# --------------------------------------------------------------------------- #
# Deduplicate: one DICOM per study
#   PA preferred over AP; within same ViewPosition, sort by dicom_id
# --------------------------------------------------------------------------- #
VIEW_ORDER = {"PA": 0, "AP": 1}
df["_view_rank"] = df["ViewPosition"].map(VIEW_ORDER)
df = df.sort_values(["study_id", "_view_rank", "dicom_id"])
df = df.drop_duplicates(subset="study_id", keep="first").drop(columns="_view_rank")
print(f"  After dedup (1 per study)  : {len(df):,} studies")

# --------------------------------------------------------------------------- #
# Join pathology labels
# --------------------------------------------------------------------------- #
df = df.merge(labels_df, on=["subject_id", "study_id"], how="left")

# --------------------------------------------------------------------------- #
# Reorder columns and save
# --------------------------------------------------------------------------- #
df = df[["split", "subject_id", "study_id", "dicom_id", "ViewPosition"] + LABEL_COLS]
df.to_csv(OUT, index=False)
print(f"\nManifest saved to: {OUT}")

# --------------------------------------------------------------------------- #
# Summary stats
# --------------------------------------------------------------------------- #
print("\n--- Split counts ---")
print(df["split"].value_counts().to_string())

print("\n--- ViewPosition counts ---")
print(df["ViewPosition"].value_counts().to_string())

print("\n--- Label prevalence (% positive, ignoring NaN/-1) ---")
for col in LABEL_COLS:
    pos = (df[col] == 1).sum()
    total = df[col].notna().sum()
    print(f"  {col:<30} {100 * pos / total:.1f}%  ({pos:,} / {total:,})")
