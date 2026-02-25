"""PyTorch Dataset for MIMIC-CXR image-report pairs."""

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
        csv_path: str,
        split: str,
        tokenizer_name: str,
        max_length: int = 512,
        image_transform: Optional[callable] = None,
        combine_sections: bool = True,
        section_separator: str = " [SEP] ",
        text_fallback: str = "No findings reported.",
        chexpert_labels: Optional[List[str]] = None,
    ):
        self.split = split
        self.max_length = max_length
        self.image_transform = image_transform
        self.combine_sections = combine_sections
        self.section_separator = section_separator
        self.text_fallback = text_fallback

        print(f"Loading {split} data from {csv_path}...")
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == split].reset_index(drop=True)
        print(f"Loaded {len(self.data)} {split} samples")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.chexpert_cols = chexpert_labels or [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        self._print_label_distribution()

    def _print_label_distribution(self):
        """Print CheXpert label distribution for this split."""
        print(f"\nCheXpert label distribution ({self.split}):")
        for col in self.chexpert_cols:
            if col not in self.data.columns:
                continue
            total = len(self.data)
            positive = (self.data[col] == 1.0).sum()
            negative = (self.data[col] == 0.0).sum()
            uncertain = (self.data[col] == -1.0).sum()
            missing = self.data[col].isna().sum()
            print(f"  {col:30s}: Pos={positive:4d} Neg={negative:4d} "
                  f"Unc={uncertain:4d} Missing={missing:4d} ({missing/total*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        image = self._load_image(row['image_path'])
        text = self._prepare_text(row)

        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        chexpert_labels = self._get_chexpert_labels(row)

        metadata = {
            'study_id': row['study_id'],
            'subject_id': row['subject_id'],
            'dicom_id': row['dicom_id'],
        }

        return {
            'image': image,
            'input_ids': text_encoded['input_ids'].squeeze(0),
            'attention_mask': text_encoded['attention_mask'].squeeze(0),
            'chexpert_labels': chexpert_labels,
            'metadata': metadata,
        }

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and transform an image, returning a black fallback on error."""
        try:
            image = Image.open(image_path).convert('RGB')
            if self.image_transform is not None:
                image = self.image_transform(image)
            else:
                image = transforms.ToTensor()(image)
            return image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            blank = Image.new('RGB', (224, 224), color='black')
            if self.image_transform is not None:
                return self.image_transform(blank)
            return torch.zeros(3, 224, 224)

    def _prepare_text(self, row: pd.Series) -> str:
        """Combine FINDINGS and IMPRESSION sections into a single string."""
        if self.combine_sections:
            findings = row['findings'] if pd.notna(row['findings']) else ""
            impression = row['impression'] if pd.notna(row['impression']) else ""

            if findings and impression:
                text = findings + self.section_separator + impression
            elif findings:
                text = findings
            elif impression:
                text = impression
            else:
                text = self.text_fallback
        else:
            text = row['findings'] if pd.notna(row['findings']) else self.text_fallback

        return ' '.join(text.strip().split())

    def _get_chexpert_labels(self, row: pd.Series) -> torch.Tensor:
        """Extract CheXpert labels: 1.0=positive, 0.0=negative, NaN=masked.

        Uncertain (-1.0) labels are treated as NaN (masked out in loss).
        """
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
    """Custom collate function for DataLoader."""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'chexpert_labels': torch.stack([item['chexpert_labels'] for item in batch]),
        'metadata': [item['metadata'] for item in batch],
    }