"""PyTorch Dataset for MIMIC-CXR image-report pairs.

Changes from original:
  1. image_size parameter — used for blank-image fallback (was hardcoded 224).
  2. use_findings_only parameter (default True):
       - Filters to has_findings=True rows at init time.
       - _prepare_text() returns FINDINGS section only.
     Rationale: MAIRA-2 generates FINDINGS only. Using FINDINGS on both sides
     makes the GT↔generated MedCLIP embedding comparison fair.
  3. Diagnostic prints: sample reports at init, tensor shape + token counts
     on first few __getitem__ calls (3 per dataloader worker).
"""

import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, List
from transformers import AutoTokenizer


class MIMICCXRDataset(Dataset):
    """MIMIC-CXR Dataset returning image, tokenized report, and CheXpert labels."""

    def __init__(
        self,
        csv_path:         str,
        split:            str,
        tokenizer_name:   str,
        image_size:       int  = 512,
        max_length:       int  = 512,
        image_transform:  Optional[callable] = None,
        use_findings_only: bool = True,
        combine_sections: bool = False,
        section_separator: str = " [SEP] ",
        text_fallback:    str  = "No findings reported.",
        chexpert_labels:  Optional[List[str]] = None,
    ):
        self.split             = split
        self.image_size        = image_size
        self.max_length        = max_length
        self.image_transform   = image_transform
        self.use_findings_only = use_findings_only
        self.combine_sections  = combine_sections
        self.section_separator = section_separator
        self.text_fallback     = text_fallback
        self._item_check_count = 0
        self._ITEM_CHECK_LIMIT = 3   # prints per worker process (NUM_WORKERS=2 → 6 total)

        print(f"\n{'='*60}")
        print(f"[MIMICCXRDataset] split='{split}' | {csv_path}")

        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        print(f"  Rows in split '{split}':    {len(df):>8,}")

        # ── TEXT FILTER ────────────────────────────────────────────────────────
        if self.use_findings_only:
            before = len(df)
            mask = df['has_findings'].astype(str).str.lower().isin(['true', '1', '1.0'])
            df = df[mask].reset_index(drop=True)
            print(f"  has_findings=True:         {len(df):>8,}  "
                  f"(dropped {before - len(df):,})")
        else:
            before = len(df)
            mask = (
                df['has_findings'].astype(str).str.lower().isin(['true', '1', '1.0']) |
                df['has_impression'].astype(str).str.lower().isin(['true', '1', '1.0'])
            )
            df = df[mask].reset_index(drop=True)
            print(f"  has_any_text=True:         {len(df):>8,}  "
                  f"(dropped {before - len(df):,})")

        for col in ['findings', 'impression', 'indication']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
            else:
                df[col] = ''

        self.data = df

        print(f"  Text mode:   {'FINDINGS only' if self.use_findings_only else 'FINDINGS + IMPRESSION'}")
        print(f"  Image size:  {self.image_size}×{self.image_size}")
        print(f"  Max tokens:  {self.max_length}")
        print(f"  Final size:  {len(self.data):>8,}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.chexpert_cols = chexpert_labels or [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        # ── SAMPLE REPORT PREVIEW ─────────────────────────────────────────────
        print(f"\n  Sample FINDINGS (first 3 rows of '{split}'):")
        for i in range(min(3, len(self.data))):
            row = self.data.iloc[i]
            text = self._prepare_text(row)
            preview = text[:120] + ('...' if len(text) > 120 else '')
            print(f"    [{i}] study={row['study_id']} | \"{preview}\"")
        print(f"{'='*60}\n")

        self._print_label_distribution()

    def _print_label_distribution(self):
        print(f"[MIMICCXRDataset] CheXpert label distribution ({self.split}):")
        for col in self.chexpert_cols:
            if col not in self.data.columns:
                continue
            total     = len(self.data)
            positive  = (self.data[col] == 1.0).sum()
            negative  = (self.data[col] == 0.0).sum()
            uncertain = (self.data[col] == -1.0).sum()
            missing   = self.data[col].isna().sum()
            print(f"  {col:30s}: Pos={positive:5,} Neg={negative:5,} "
                  f"Unc={uncertain:4,} Missing={missing:4,} ({missing/total*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row  = self.data.iloc[idx]
        image = self._load_image(row['image_path'])
        text  = self._prepare_text(row)

        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        chexpert_labels = self._get_chexpert_labels(row)

        # ── DIAGNOSTIC: first _ITEM_CHECK_LIMIT items per worker ─────────────
        if self._item_check_count < self._ITEM_CHECK_LIMIT:
            real_tokens = text_encoded['attention_mask'].squeeze().sum().item()
            print(f"[Dataset.__getitem__] sample {self._item_check_count} | "
                  f"split={self.split} | study_id={row['study_id']}")
            print(f"  image tensor: {tuple(image.shape)}  "
                  f"(min={image.min():.3f}, max={image.max():.3f})")
            print(f"  real tokens:  {int(real_tokens)} / {self.max_length}")
            print(f"  findings:     \"{text[:100]}{'...' if len(text)>100 else ''}\"")
            self._item_check_count += 1
        # ─────────────────────────────────────────────────────────────────────

        return {
            'image':           image,
            'input_ids':       text_encoded['input_ids'].squeeze(0),
            'attention_mask':  text_encoded['attention_mask'].squeeze(0),
            'chexpert_labels': chexpert_labels,
            'metadata': {
                'study_id':   row['study_id'],
                'subject_id': row['subject_id'],
                'dicom_id':   row['dicom_id'],
            },
        }

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and transform image; return a blank square on error."""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform is not None:
                return self.image_transform(image)
            return transforms.ToTensor()(image)
        except Exception as e:
            print(f"[Dataset] WARNING: could not load {image_path}: {e}. Using blank.")
            blank = Image.new('RGB', (self.image_size, self.image_size), color='black')
            if self.image_transform is not None:
                return self.image_transform(blank)
            return torch.zeros(3, self.image_size, self.image_size)

    def _prepare_text(self, row: pd.Series) -> str:
        """Return the text string to embed.

        use_findings_only=True (default):
            FINDINGS only — matches what MAIRA-2 generates.
        use_findings_only=False (legacy):
            FINDINGS + IMPRESSION combined.
        """
        if self.use_findings_only:
            findings = row['findings'].strip() if row['findings'] else ''
            return ' '.join((findings if findings else self.text_fallback).split())

        if self.combine_sections:
            findings   = row['findings'].strip()   if row['findings']   else ''
            impression = row['impression'].strip() if row['impression'] else ''
            if findings and impression:
                text = findings + self.section_separator + impression
            elif findings:
                text = findings
            elif impression:
                text = impression
            else:
                text = self.text_fallback
        else:
            text = row['findings'].strip() if row['findings'] else self.text_fallback

        return ' '.join(text.split())

    def _get_chexpert_labels(self, row: pd.Series) -> torch.Tensor:
        """1.0=positive, 0.0=negative, NaN=masked (uncertain -1.0 → NaN)."""
        labels = []
        for col in self.chexpert_cols:
            if col not in row.index:
                labels.append(float('nan'))
                continue
            val = row[col]
            if pd.isna(val) or val == -1.0:
                labels.append(float('nan'))
            elif val == 1.0:
                labels.append(1.0)
            elif val == 0.0:
                labels.append(0.0)
            else:
                labels.append(float('nan'))
        return torch.tensor(labels, dtype=torch.float32)


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Collate batch.

    __getitem__ returns key 'image' (singular).
    collate_fn stacks under 'images' (plural) — matching trainer.py's batch['images'].
    Do not rename either side.
    """
    return {
        'images':          torch.stack([item['image']           for item in batch]),
        'input_ids':       torch.stack([item['input_ids']       for item in batch]),
        'attention_mask':  torch.stack([item['attention_mask']  for item in batch]),
        'chexpert_labels': torch.stack([item['chexpert_labels'] for item in batch]),
        'metadata':        [item['metadata'] for item in batch],
    }