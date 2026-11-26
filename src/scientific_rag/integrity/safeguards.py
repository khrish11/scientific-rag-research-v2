"""Research integrity safeguards and leakage prevention."""

import logging
from collections import defaultdict

from pydantic import BaseModel, Field

from scientific_rag.data.schemas import QASPERDataset

logger = logging.getLogger(__name__)


class SplitOverlapReport(BaseModel):
    """Report on split overlap."""

    has_overlap: bool = Field(description="Whether any overlap was detected")
    overlapping_papers: set[str] = Field(
        default_factory=set, description="Set of paper IDs that appear in multiple splits"
    )
    overlap_details: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Details of which splits each overlapping paper appears in",
    )


class LeakageReport(BaseModel):
    """Report on potential data leakage."""

    gold_labels_in_input: bool = Field(
        default=False, description="Whether gold labels were found in input data"
    )
    test_data_in_training: bool = Field(
        default=False, description="Whether test data was found in training data"
    )
    answers_in_retrieval: bool = Field(
        default=False, description="Whether answer annotations were found in retrieval context"
    )
    details: list[str] = Field(
        default_factory=list, description="Detailed description of any leakage issues"
    )


class DataSplitValidator:
    """Validator for dataset split isolation."""

    def __init__(self) -> None:
        """Initialize validator."""
        self.split_datasets: dict[str, QASPERDataset] = {}

    def add_split(self, split: str, dataset: QASPERDataset) -> None:
        """Add a dataset split for validation.

        Args:
            split: Split name (train, validation, test).
            dataset: Dataset for this split.
        """
        self.split_datasets[split] = dataset
        logger.info(f"Added {split} split with {len(dataset.papers)} papers")

    def check_paper_overlap(self) -> SplitOverlapReport:
        """Check for paper overlap across splits.

        Returns:
            SplitOverlapReport with overlap details.
        """
        logger.info("Checking for paper overlap across splits")

        # Collect paper IDs per split
        papers_by_split: dict[str, set[str]] = {}
        for split, dataset in self.split_datasets.items():
            paper_ids = {paper.id for paper in dataset.papers}
            papers_by_split[split] = paper_ids
            logger.info(f"Split {split}: {len(paper_ids)} unique papers")

        # Check for overlaps
        all_papers: dict[str, list[str]] = defaultdict(list)
        for split, paper_ids in papers_by_split.items():
            for paper_id in paper_ids:
                all_papers[paper_id].append(split)

        overlapping_papers = {
            paper_id for paper_id, splits in all_papers.items() if len(splits) > 1
        }

        overlap_details = {
            paper_id: splits for paper_id, splits in all_papers.items() if len(splits) > 1
        }

        has_overlap = len(overlapping_papers) > 0

        if has_overlap:
            logger.warning(f"Found {len(overlapping_papers)} overlapping papers across splits")
        else:
            logger.info("No paper overlap detected across splits")

        return SplitOverlapReport(
            has_overlap=has_overlap,
            overlapping_papers=overlapping_papers,
            overlap_details=overlap_details,
        )

    def validate_split_isolation(self) -> bool:
        """Validate that splits are properly isolated.

        Returns:
            True if splits are isolated, False otherwise.
        """
        report = self.check_paper_overlap()

        if report.has_overlap:
            logger.error("Split isolation validation FAILED")
            for paper_id, splits in report.overlap_details.items():
                logger.error(f"Paper {paper_id} appears in splits: {splits}")
            return False

        logger.info("Split isolation validation PASSED")
        return True


class LeakageDetector:
    """Detector for data leakage in experimental pipeline."""

    def __init__(self) -> None:
        """Initialize leakage detector."""
        self.issues: list[str] = []

    def check_gold_labels_in_input(
        self,
        input_data: dict[str, object],
        gold_label_fields: set[str],
    ) -> bool:
        """Check if gold labels are present in input data.

        Args:
            input_data: Input data dictionary.
            gold_label_fields: Set of field names that should not be in input.

        Returns:
            True if gold labels found in input (leakage detected).
        """
        found_fields = gold_label_fields & set(input_data.keys())

        if found_fields:
            self.issues.append(f"Gold label fields found in input data: {found_fields}")
            logger.warning(f"Gold label leakage detected: {found_fields}")
            return True

        return False

    def check_test_in_training(
        self,
        train_papers: set[str],
        test_papers: set[str],
    ) -> bool:
        """Check if test papers appear in training data.

        Args:
            train_papers: Set of training paper IDs.
            test_papers: Set of test paper IDs.

        Returns:
            True if overlap found (leakage detected).
        """
        overlap = train_papers & test_papers

        if overlap:
            self.issues.append(f"Test papers found in training data: {len(overlap)} papers")
            logger.warning(f"Test-train leakage detected: {len(overlap)} overlapping papers")
            return True

        return False

    def check_answers_in_retrieval_context(
        self,
        retrieval_context: str,
        gold_answer: str,
    ) -> bool:
        """Check if gold answer appears in retrieval context.

        Args:
            retrieval_context: The retrieved context text.
            gold_answer: The gold answer text.

        Returns:
            True if answer found in context (potential leakage).
        """
        # Simple containment check
        if gold_answer.lower() in retrieval_context.lower():
            self.issues.append("Gold answer found in retrieval context")
            logger.warning("Answer leakage detected in retrieval context")
            return True

        return False

    def get_report(self) -> LeakageReport:
        """Get leakage report.

        Returns:
            LeakageReport with all detected issues.
        """
        return LeakageReport(
            gold_labels_in_input="gold label" in " ".join(self.issues).lower(),
            test_data_in_training="test" in " ".join(self.issues).lower()
            and "training" in " ".join(self.issues).lower(),
            answers_in_retrieval="answer" in " ".join(self.issues).lower()
            and "retrieval" in " ".join(self.issues).lower(),
            details=self.issues,
        )

    def reset(self) -> None:
        """Reset detector state."""
        self.issues = []


def validate_split_isolation(
    train_dataset: QASPERDataset,
    validation_dataset: QASPERDataset,
    test_dataset: QASPERDataset,
) -> bool:
    """Validate isolation between dataset splits.

    Args:
        train_dataset: Training dataset.
        validation_dataset: Validation dataset.
        test_dataset: Test dataset.

    Returns:
        True if splits are isolated, False otherwise.
    """
    validator = DataSplitValidator()
    validator.add_split("train", train_dataset)
    validator.add_split("validation", validation_dataset)
    validator.add_split("test", test_dataset)

    return validator.validate_split_isolation()


def detect_leakage(
    input_data: dict[str, object],
    gold_label_fields: set[str] | None = None,
) -> LeakageReport:
    """Detect data leakage in experimental inputs.

    Args:
        input_data: Input data to check.
        gold_label_fields: Fields that should not be present.

    Returns:
        LeakageReport with detected issues.
    """
    detector = LeakageDetector()
    gold_fields = gold_label_fields or {
        "answer",
        "gold_answer",
        "evidence",
        "gold_evidence",
        "label",
        "gold_label",
    }

    detector.check_gold_labels_in_input(input_data, gold_fields)

    return detector.get_report()
