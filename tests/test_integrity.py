"""Tests for research integrity safeguards."""

import pytest

from scientific_rag.data.schemas import FullText, FullTextSection, Paper, QASPERDataset
from scientific_rag.integrity.safeguards import (
    DataSplitValidator,
    LeakageDetector,
    LeakageReport,
    detect_leakage,
    validate_split_isolation,
)


@pytest.fixture
def sample_train_dataset() -> QASPERDataset:
    """Create a sample training dataset."""
    papers = [
        Paper(
            id="paper1",
            title="Train Paper 1",
            abstract="Abstract",
            full_text=FullText(
                sections=[FullTextSection(section_name="Intro", paragraphs=["Text"])]
            ),
            qas=[],
            figures_and_tables=[],
        ),
        Paper(
            id="paper2",
            title="Train Paper 2",
            abstract="Abstract",
            full_text=FullText(
                sections=[FullTextSection(section_name="Intro", paragraphs=["Text"])]
            ),
            qas=[],
            figures_and_tables=[],
        ),
    ]
    return QASPERDataset(papers=papers, split="train", version="0.3.0", source_url="http://test")


@pytest.fixture
def sample_validation_dataset() -> QASPERDataset:
    """Create a sample validation dataset."""
    papers = [
        Paper(
            id="paper4",
            title="Validation Paper",
            abstract="Abstract",
            full_text=FullText(
                sections=[FullTextSection(section_name="Intro", paragraphs=["Text"])]
            ),
            qas=[],
            figures_and_tables=[],
        ),
    ]
    return QASPERDataset(
        papers=papers, split="validation", version="0.3.0", source_url="http://test"
    )


@pytest.fixture
def sample_test_dataset() -> QASPERDataset:
    """Create a sample test dataset."""
    papers = [
        Paper(
            id="paper3",
            title="Test Paper 1",
            abstract="Abstract",
            full_text=FullText(
                sections=[FullTextSection(section_name="Intro", paragraphs=["Text"])]
            ),
            qas=[],
            figures_and_tables=[],
        ),
    ]
    return QASPERDataset(papers=papers, split="test", version="0.3.0", source_url="http://test")


@pytest.fixture
def overlapping_test_dataset() -> QASPERDataset:
    """Create a test dataset with overlapping papers."""
    papers = [
        Paper(
            id="paper1",  # Overlaps with train
            title="Overlap Paper",
            abstract="Abstract",
            full_text=FullText(
                sections=[FullTextSection(section_name="Intro", paragraphs=["Text"])]
            ),
            qas=[],
            figures_and_tables=[],
        ),
    ]
    return QASPERDataset(papers=papers, split="test", version="0.3.0", source_url="http://test")


def test_data_split_validator_initialization() -> None:
    """Test DataSplitValidator initialization."""
    validator = DataSplitValidator()
    assert validator.split_datasets == {}


def test_add_split(sample_train_dataset: QASPERDataset) -> None:
    """Test adding a split to validator."""
    validator = DataSplitValidator()
    validator.add_split("train", sample_train_dataset)
    assert "train" in validator.split_datasets
    assert len(validator.split_datasets["train"].papers) == 2


def test_check_paper_overlap_no_overlap(
    sample_train_dataset: QASPERDataset, sample_test_dataset: QASPERDataset
) -> None:
    """Test paper overlap detection with no overlap."""
    validator = DataSplitValidator()
    validator.add_split("train", sample_train_dataset)
    validator.add_split("test", sample_test_dataset)

    report = validator.check_paper_overlap()

    assert report.has_overlap is False
    assert len(report.overlapping_papers) == 0


def test_check_paper_overlap_with_overlap(
    sample_train_dataset: QASPERDataset, overlapping_test_dataset: QASPERDataset
) -> None:
    """Test paper overlap detection with overlap."""
    validator = DataSplitValidator()
    validator.add_split("train", sample_train_dataset)
    validator.add_split("test", overlapping_test_dataset)

    report = validator.check_paper_overlap()

    assert report.has_overlap is True
    assert "paper1" in report.overlapping_papers
    assert report.overlap_details["paper1"] == ["train", "test"]


def test_validate_split_isolation_success(
    sample_train_dataset: QASPERDataset,
    sample_validation_dataset: QASPERDataset,
    sample_test_dataset: QASPERDataset,
) -> None:
    """Test split isolation validation with no overlap."""
    result = validate_split_isolation(
        sample_train_dataset, sample_validation_dataset, sample_test_dataset
    )
    assert result is True


def test_validate_split_isolation_failure(
    sample_train_dataset: QASPERDataset,
    sample_validation_dataset: QASPERDataset,
    overlapping_test_dataset: QASPERDataset,
) -> None:
    """Test split isolation validation with overlap."""
    result = validate_split_isolation(
        sample_train_dataset, sample_validation_dataset, overlapping_test_dataset
    )
    assert result is False


def test_leakage_detector_initialization() -> None:
    """Test LeakageDetector initialization."""
    detector = LeakageDetector()
    assert detector.issues == []


def test_check_gold_labels_in_input() -> None:
    """Test detection of gold labels in input."""
    detector = LeakageDetector()

    # No leakage
    detector.check_gold_labels_in_input({"query": "test"}, {"answer", "evidence"})
    assert len(detector.issues) == 0

    # Leakage detected
    detector.check_gold_labels_in_input({"query": "test", "answer": "secret"}, {"answer"})
    assert len(detector.issues) == 1


def test_check_test_in_training() -> None:
    """Test detection of test data in training."""
    detector = LeakageDetector()

    # No overlap
    detector.check_test_in_training({"paper1", "paper2"}, {"paper3", "paper4"})
    assert len(detector.issues) == 0

    # Overlap detected
    detector.check_test_in_training({"paper1", "paper2"}, {"paper2", "paper3"})
    assert len(detector.issues) == 1


def test_check_answers_in_retrieval_context() -> None:
    """Test detection of answers in retrieval context."""
    detector = LeakageDetector()

    # No leakage
    detector.check_answers_in_retrieval_context("This is context about X", "The answer is Y")
    assert len(detector.issues) == 0

    # Leakage detected
    detector.check_answers_in_retrieval_context("The answer is 42", "42")
    assert len(detector.issues) == 1


def test_get_leakage_report() -> None:
    """Test getting leakage report."""
    detector = LeakageDetector()
    detector.issues.append("Test issue")

    report = detector.get_report()
    assert isinstance(report, LeakageReport)
    assert len(report.details) == 1


def test_detector_reset() -> None:
    """Test resetting detector state."""
    detector = LeakageDetector()
    detector.issues.append("Issue 1")

    detector.reset()
    assert len(detector.issues) == 0


def test_detect_leakage_function() -> None:
    """Test detect_leakage convenience function."""
    # No leakage
    report = detect_leakage({"query": "test"})
    assert len(report.details) == 0

    # Leakage
    report = detect_leakage({"query": "test", "answer": "secret"})
    assert len(report.details) > 0
