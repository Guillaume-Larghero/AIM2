import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "Pleural Effusion", "Pneumonia", "Pneumothorax", "Pleural Other",
    "Support Devices", "No Finding"
]

# Matches section headers like "FINDINGS:", "IMPRESSION:", etc.
_SECTION_RE = re.compile(r"^\s*([A-Z][A-Z /]+):\s*", re.MULTILINE)


def _parse_report(text):
    """Return a dict of {section_name: section_text} for a raw report string."""
    headers = [(m.group(1).strip(), m.start(), m.end())
               for m in _SECTION_RE.finditer(text)]
    if not headers:
        return {"FULL": text.strip()}
    sections = {}
    for i, (name, _, content_start) in enumerate(headers):
        content_end = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        sections[name] = text[content_start:content_end].strip()
    return sections


class MIMICCXRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        report_dir=None,
        split="train",
        labeler="chexpert",
        transform=None,
        uncertain_as=0,
        report_section="impression",  # 'impression', 'findings', or 'full'
    ):
        """
        root_dir:        path to the MIMIC-CXR-JPG root (contains 'files/' and the CSV files)
        report_dir:      path to the MIMIC-CXR root containing the .txt reports; defaults to
                         root_dir (set this if JPG and reports live in different trees)
        split:           one of 'train', 'validate', 'test'
        labeler:         'chexpert' or 'negbio'
        transform:       torchvision transforms; defaults to resize + normalize
        uncertain_as:    how to handle -1.0 labels — 0 (negative) or 1 (positive)
        report_section:  which section of the report to return:
                           'impression' → IMPRESSION (falls back to FINDINGS, then full text)
                           'findings'   → FINDINGS (falls back to IMPRESSION, then full text)
                           'full'       → entire raw report text
        """
        self.root_dir = root_dir
        self.report_dir = report_dir or root_dir
        self.uncertain_as = uncertain_as
        self.report_section = report_section.upper()

        split_df = pd.read_csv(os.path.join(
            root_dir, "mimic-cxr-2.0.0-split.csv.gz"))
        split_df = split_df[split_df["split"] == split].reset_index(drop=True)

        label_file = f"mimic-cxr-2.0.0-{labeler}.csv.gz"
        labels_df = pd.read_csv(os.path.join(root_dir, label_file))

        self.df = split_df.merge(
            labels_df, on=["subject_id", "study_id"], how="left")

        meta_df = pd.read_csv(os.path.join(
            root_dir, "mimic-cxr-2.0.0-metadata.csv.gz"))
        self.df = self.df.merge(
            meta_df[["dicom_id", "ViewPosition"]], on="dicom_id", how="left")

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])

    # ------------------------------------------------------------------
    def _report_path(self, subject_id, study_id):
        pid = str(subject_id)
        return os.path.join(
            self.report_dir, "files",
            f"p{pid[:2]}", f"p{pid}",
            f"s{study_id}.txt"
        )

    def _load_report(self, subject_id, study_id):
        path = self._report_path(subject_id, study_id)
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        if self.report_section == "FULL":
            return raw.strip()
        sections = _parse_report(raw)
        # preferred → fallback order
        if self.report_section == "IMPRESSION":
            order = ["IMPRESSION", "FINDINGS"]
        else:
            order = ["FINDINGS", "IMPRESSION"]
        for key in order:
            if key in sections:
                return sections[key]
        # last resort: return everything
        return raw.strip()

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = str(int(row["subject_id"]))
        sid = int(row["study_id"])

        img_path = os.path.join(
            self.root_dir, "files",
            f"p{pid[:2]}", f"p{pid}",
            f"s{sid}", f"{row['dicom_id']}.jpg"
        )
        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        label_vals = self.df[LABELS].iloc[idx].fillna(0).values.astype(float)
        label_vals[label_vals == -1.0] = self.uncertain_as

        report_text = self._load_report(int(pid), sid)

        return {
            "image":      image,
            "labels":     label_vals,
            "report":     report_text,
            "subject_id": int(pid),
            "study_id":   sid,
            "dicom_id":   row["dicom_id"],
            "view":       row.get("ViewPosition", ""),
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ROOT = "/path/to/mimic-cxr-jpg"
    # If the .txt reports are in a separate MIMIC-CXR checkout, set report_dir:
    # REPORT_DIR = "/path/to/mimic-cxr"

    train_ds = MIMICCXRDataset(
        ROOT, split="train", report_section="impression")
    val_ds = MIMICCXRDataset(ROOT, split="validate")
    test_ds = MIMICCXRDataset(ROOT, split="test")

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                        num_workers=4, pin_memory=True)
    batch = next(iter(loader))

    print("Image shape :", batch["image"].shape)    # [8, 1, 224, 224]
    print("Labels shape:", batch["labels"].shape)   # [8, 14]
    print("Report sample:", batch["report"][0][:120])
