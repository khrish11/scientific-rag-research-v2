"""Tests for dataset analysis and statistics."""

import pytest

from scientific_rag.analysis.stats import (
    AnswerTypeStats,
    DatasetStatistics,
    LengthStats,
    compute_length_stats,
    compute_statistics,
)
from scientific_rag.data.schemas import (
    AnswerAnnotation,
    AnswerData,
    FullText,
    FullTextSection,
    Paper,
    QASPERDataset,
    QuestionAnswer,
)


@pytest.fixture
def sample_dataset() -> QASPERDataset:
    """Create a sample dataset for testing."""
    papers = [
        Paper(
            id="paper1",
            title="Test Paper 1",
            abstract="Abstract 1",
            full_text=FullText(
                sections=[
                    FullTextSection(
                        section_name="Introduction",
                        paragraphs=["Paragraph 1 text.", "Paragraph 2 text."],
                    ),
                ]
            ),
            qas=[
                QuestionAnswer(
                    question_id="q1",
                    question="What is the method?",
                    question_type="free-form",
                    answers=[
                        AnswerAnnotation(
                            answer=AnswerData(
                                unanswerable=False,
                                extractive_spans=[],
                                yes_no=None,
                                free_form_answer="Answer 1",
                                evidence=["Paragraph 1 text."],
                                highlighted_evidence=[],
                            ),
                            annotation_id="ann1",
                            worker_id="worker1",
                        )
                    ],
                ),
                QuestionAnswer(
                    question_id="q2",
                    question="Is this yes/no?",
                    question_type="yes/no",
                    answers=[
                        AnswerAnnotation(
                            answer=AnswerData(
                                unanswerable=False,
                                extractive_spans=[],
                                yes_no="yes",
                                free_form_answer="",
                                evidence=[],
                                highlighted_evidence=[],
                            ),
                            annotation_id="ann2",
                            worker_id="worker1",
                        )
                    ],
                ),
            ],
            figures_and_tables=[],
        ),
        Paper(
            id="paper2",
            title="Test Paper 2",
            abstract="Abstract 2",
            full_text=FullText(
                sections=[
                    FullTextSection(
                        section_name="Method",
                        paragraphs=["Method paragraph."],
                    ),
                ]
            ),
            qas=[
                QuestionAnswer(
                    question_id="q3",
                    question="What is the result?",
                    question_type="span",
                    answers=[
                        AnswerAnnotation(
                            answer=AnswerData(
                                unanswerable=True,
                                extractive_spans=[],
                                yes_no=None,
                                free_form_answer="",
                                evidence=[],
                                highlighted_evidence=[],
                            ),
                            annotation_id="ann3",
                            worker_id="worker1",
                        )
                    ],
                ),
            ],
            figures_and_tables=[],
        ),
    ]

    return QASPERDataset(papers=papers, split="test", version="0.3.0", source_url="http://test")


def test_compute_length_stats_empty() -> None:
    """Test compute_length_stats with empty list."""
    stats = compute_length_stats([])
    assert stats.mean == 0.0
    assert stats.median == 0.0
    assert stats.min == 0
    assert stats.max == 0
    assert stats.std == 0.0


def test_compute_length_stats_single() -> None:
    """Test compute_length_stats with single value."""
    stats = compute_length_stats([100])
    assert stats.mean == 100.0
    assert stats.median == 100.0
    assert stats.min == 100
    assert stats.max == 100
    assert stats.std == 0.0


def test_compute_length_stats_multiple() -> None:
    """Test compute_length_stats with multiple values."""
    stats = compute_length_stats([10, 20, 30, 40, 50])
    assert stats.mean == 30.0
    assert stats.median == 30.0
    assert stats.min == 10
    assert stats.max == 50
    assert stats.std > 0


def test_compute_statistics(sample_dataset: QASPERDataset) -> None:
    """Test compute_statistics function."""
    stats = compute_statistics(sample_dataset)

    assert stats.split == "test"
    assert stats.num_papers == 2
    assert stats.num_questions == 3
    assert stats.num_annotations == 3
    assert stats.answer_type_distribution.free_form == 1
    assert stats.answer_type_distribution.yes_no == 1
    assert stats.answer_type_distribution.unanswerable == 1
    assert stats.num_unanswerable == 1
    assert stats.num_questions_with_evidence == 1


def test_dataset_statistics_schema() -> None:
    """Test DatasetStatistics schema."""
    stats = DatasetStatistics(
        split="train",
        num_papers=100,
        num_questions=500,
        num_annotations=600,
        answer_type_distribution=AnswerTypeStats(yes_no=100, span=200, free_form=200),
        num_unanswerable=10,
        num_questions_with_evidence=400,
        mean_evidence_per_question=1.5,
        paragraph_length=LengthStats(mean=100, median=90, min=10, max=500, std=50),
        document_length=LengthStats(mean=1000, median=900, min=100, max=5000, std=500),
        num_multi_evidence_questions=50,
    )

    assert stats.split == "train"
    assert stats.num_papers == 100
    assert stats.num_questions == 500
