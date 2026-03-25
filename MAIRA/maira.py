import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from GENERATION.pipeline.generator import GeneratedReport

logger = logging.getLogger(__name__)

# ViewPosition values that correspond to frontal/lateral projections
_FRONTAL_VIEWS = {"PA", "AP", "pa", "ap"}
_LATERAL_VIEWS = {"LATERAL", "LL", "Lateral", "lateral", "ll"}

# MIMIC-CXR data paths (shared cluster location)
_DATA_CSV = "/n/groups/training/bmif203/AIM2/processed_data/processed_data.csv"


# ---------------------------------------------------------------------------
# Study-level data helpers
# ---------------------------------------------------------------------------

def load_mimic_study(
    study_id: str,
    data_csv: str = _DATA_CSV,
) -> Dict[str, Any]:
    """Return frontal/lateral image paths and metadata for a MIMIC study.

    Args:
        study_id: MIMIC study ID (with or without leading 's').
        data_csv: Path to the processed MIMIC CSV.

    Returns:
        Dict with keys:
            frontal_path, lateral_path (or None), indication,
            findings, impression, chexpert_labels.
    """
    df = pd.read_csv(data_csv, dtype={"study_id": str})

    # Normalise study_id — CSV stores numeric IDs without 's' prefix
    sid = str(study_id).lstrip("s")
    rows = df[df["study_id"] == sid]
    if rows.empty:
        raise ValueError(f"Study {study_id} not found in {data_csv}")

    frontal_row = rows[rows["ViewPosition"].isin(_FRONTAL_VIEWS)]
    lateral_row = rows[rows["ViewPosition"].isin(_LATERAL_VIEWS)]

    if frontal_row.empty:
        # Fall back to the first available row
        frontal_row = rows.iloc[[0]]
        logger.warning(
            f"No explicit frontal view for study {study_id}; "
            f"using first row (ViewPosition={rows.iloc[0]['ViewPosition']})"
        )

    row = frontal_row.iloc[0]

    chexpert_cols = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices",
    ]
    chexpert_labels = [
        float(row[c]) if pd.notna(row.get(c)) else 0.0
        for c in chexpert_cols
        if c in row.index
    ]

    return {
        "frontal_path": row["image_path"],
        "lateral_path": lateral_row.iloc[0]["image_path"] if not lateral_row.empty else None,
        "indication": row.get("indication") if pd.notna(row.get("indication")) else None,
        "findings": row.get("findings") if pd.notna(row.get("findings")) else None,
        "impression": row.get("impression") if pd.notna(row.get("impression")) else None,
        "chexpert_labels": chexpert_labels or None,
    }


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class MAIRAReportGenerator:
    """Drop-in replacement for ReportGenerator using MAIRA-2 for direct CXR inference.

    MAIRA-2 generates the **findings** section directly from the image.
    The impression field in the returned GeneratedReport is left empty
    because MAIRA-2 does not produce it; add a post-processing step if needed.

    Args:
        device: PyTorch device string. Defaults to 'cuda' if available.
        use_grounding: Whether to generate a grounded report (findings with
            bounding boxes). Defaults to False (plain narrative findings).
        model_name: HuggingFace model ID. Defaults to 'microsoft/maira-2'.
    """

    DEFAULT_MODEL = "microsoft/maira-2"

    def __init__(
        self,
        device: Optional[str] = None,
        use_grounding: bool = False,
        model_name: str = DEFAULT_MODEL,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_grounding = use_grounding
        self.model_name = model_name

        logger.info(f"Loading MAIRA-2 ({model_name}) on {self.device} …")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = self.model.eval().to(self.device)
        logger.info("MAIRA-2 ready.")

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def generate_report(
        self,
        image_path: str,
        study_id: Optional[str] = None,
        ground_truth_findings: Optional[str] = None,
        ground_truth_impression: Optional[str] = None,
        query_chexpert_labels: Optional[List[float]] = None,
        # MAIRA-2 optional context inputs
        lateral_image_path: Optional[str] = None,
        indication: Optional[str] = None,
        technique: Optional[str] = None,
        comparison: Optional[str] = None,
        prior_frontal_path: Optional[str] = None,
        prior_report: Optional[str] = None,
        # Ignored — kept for API compatibility with ReportGenerator
        return_detailed_info: bool = False,
    ) -> GeneratedReport:
        """Generate a findings report from a frontal chest X-ray using MAIRA-2.

        Args:
            image_path: Path to the frontal CXR (required).
            study_id: MIMIC study identifier; used only for labelling output.
            ground_truth_findings: GT findings text (stored for evaluation).
            ground_truth_impression: GT impression text (stored for evaluation).
            query_chexpert_labels: CheXpert label vector (stored, not used by model).
            lateral_image_path: Optional path to the lateral view.
            indication: Clinical indication text from the report header.
            technique: Technique/protocol section text.
            comparison: Comparison section text.
            prior_frontal_path: Optional path to a prior frontal CXR.
            prior_report: Prior study findings text (used with prior_frontal_path).
            return_detailed_info: Ignored; present for API compatibility.

        Returns:
            GeneratedReport with findings populated by MAIRA-2.
            The impression field is empty (MAIRA-2 does not generate it).
        """
        start_time = time.time()

        frontal_image = Image.open(image_path).convert("RGB")
        lateral_image = (
            Image.open(lateral_image_path).convert("RGB")
            if lateral_image_path
            else None
        )
        prior_frontal = (
            Image.open(prior_frontal_path).convert("RGB")
            if prior_frontal_path
            else None
        )

        processed = self.processor.format_and_preprocess_reporting_input(
            current_frontal=frontal_image,
            current_lateral=lateral_image,
            prior_frontal=prior_frontal,
            indication=indication,
            technique=technique,
            comparison=comparison,
            prior_report=prior_report,
            return_tensors="pt",
            get_grounding=self.use_grounding,
        )
        processed = {k: v.to(self.device) for k, v in processed.items()}

        max_new_tokens = 450 if self.use_grounding else 300

        with torch.no_grad():
            output_ids = self.model.generate(
                **processed,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        prompt_length = processed["input_ids"].shape[-1]
        decoded = self.processor.decode(
            output_ids[0][prompt_length:], skip_special_tokens=True
        )
        decoded = decoded.lstrip()  # MAIRA-2 outputs a leading space

        prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(decoded)

        if self.use_grounding:
            # prediction is List[Tuple[str, Optional[List[Tuple]]]]
            findings = " ".join(sentence for sentence, _ in prediction)
        else:
            findings = prediction if isinstance(prediction, str) else str(prediction)

        return GeneratedReport(
            study_id=study_id or Path(image_path).stem,
            query_image_path=image_path,
            findings=findings.strip(),
            impression="",  # MAIRA-2 generates findings only
            retrieved_study_ids=[],
            retrieval_scores=[],
            num_retrieved=0,
            gt_findings=ground_truth_findings,
            gt_impression=ground_truth_impression,
            gt_chexpert_labels=query_chexpert_labels,
            query_chexpert_labels=query_chexpert_labels,
            generation_time=time.time() - start_time,
            model_name=self.model_name,
        )

    def generate_report_for_study(
        self,
        study_id: str,
        data_csv: str = _DATA_CSV,
        include_lateral: bool = True,
        include_indication: bool = True,
    ) -> GeneratedReport:
        """Convenience wrapper: look up a MIMIC study and run generate_report.

        Automatically resolves frontal/lateral paths and clinical context from
        the processed data CSV.

        Args:
            study_id: MIMIC study ID.
            data_csv: Path to processed_data.csv.
            include_lateral: Pass lateral image to MAIRA-2 if available.
            include_indication: Pass indication text to MAIRA-2 if available.

        Returns:
            GeneratedReport.
        """
        study = load_mimic_study(study_id, data_csv=data_csv)
        return self.generate_report(
            image_path=study["frontal_path"],
            study_id=study_id,
            ground_truth_findings=study["findings"],
            ground_truth_impression=study["impression"],
            query_chexpert_labels=study["chexpert_labels"],
            lateral_image_path=study["lateral_path"] if include_lateral else None,
            indication=study["indication"] if include_indication else None,
        )

    def __repr__(self) -> str:
        return (
            f"MAIRAReportGenerator("
            f"model={self.model_name}, "
            f"grounding={self.use_grounding}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

def evaluate_maira_on_dataset(
    data_csv: str = _DATA_CSV,
    split: str = "test",
    n_samples: Optional[int] = None,
    include_lateral: bool = True,
    include_indication: bool = True,
    use_grounding: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
) -> List[GeneratedReport]:
    """Run MAIRA-2 report generation over the MIMIC test set.

    Loads the processed CSV, filters to the requested split, and calls
    MAIRAReportGenerator.generate_report_for_study() for each study.

    Args:
        data_csv: Path to processed_data.csv.
        split: Dataset split to evaluate ('train', 'test', or 'validate').
        n_samples: Subsample size; None evaluates the full split.
        include_lateral: Pass lateral image to MAIRA-2 when available.
        include_indication: Pass indication text to MAIRA-2 when available.
        use_grounding: Generate grounded (bounding-box) reports.
        device: PyTorch device string.
        seed: Random seed for subsampling.

    Returns:
        List of GeneratedReport objects, one per study.
    """
    df = pd.read_csv(data_csv, dtype={"study_id": str})
    df = df[df["split"] == split]
    df = df[df["has_findings"] == True]

    # One row per study: keep the frontal view (preferring PA over AP)
    frontal_mask = df["ViewPosition"].isin(_FRONTAL_VIEWS)
    df_frontal = df[frontal_mask].copy()
    if df_frontal.empty:
        df_frontal = df.copy()

    # Deduplicate to one row per study (keep PA > AP if both present)
    df_frontal["_view_rank"] = df_frontal["ViewPosition"].map(
        lambda v: 0 if v == "PA" else 1
    )
    df_frontal = (
        df_frontal.sort_values("_view_rank")
        .groupby("study_id", as_index=False)
        .first()
        .drop(columns=["_view_rank"])
    )

    if n_samples is not None and n_samples < len(df_frontal):
        df_frontal = df_frontal.sample(n=n_samples, random_state=seed)

    df_frontal = df_frontal.reset_index(drop=True)
    logger.info(
        f"evaluate_maira_on_dataset: {len(df_frontal)} studies "
        f"({split} split, grounding={use_grounding})"
    )

    generator = MAIRAReportGenerator(
        device=device,
        use_grounding=use_grounding,
    )

    results: List[GeneratedReport] = []
    for i, row in df_frontal.iterrows():
        sid = row["study_id"]
        try:
            report = generator.generate_report_for_study(
                study_id=sid,
                data_csv=data_csv,
                include_lateral=include_lateral,
                include_indication=include_indication,
            )
            results.append(report)
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(df_frontal)} studies")
        except Exception as exc:
            logger.error(f"  Failed on study {sid}: {exc}")

    logger.info(f"evaluate_maira_on_dataset: done. {len(results)} reports generated.")
    return results
