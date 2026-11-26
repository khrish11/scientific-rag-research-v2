"""Dataset analysis and statistics generation."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scientific_rag.data.schemas import QASPERDataset

logger = logging.getLogger(__name__)


class AnswerTypeStats(BaseModel):
    """Statistics for answer types."""

    yes_no: int = Field(default=0, description="Number of yes/no questions")
    span: int = Field(default=0, description="Number of span questions")
    free_form: int = Field(default=0, description="Number of free-form questions")
    multiple_choice: int = Field(default=0, description="Number of multiple-choice questions")
    unanswerable: int = Field(default=0, description="Number of unanswerable questions")
    unknown: int = Field(default=0, description="Number of questions with unknown type")


class LengthStats(BaseModel):
    """Statistics for text lengths."""

    mean: float = Field(description="Mean length")
    median: float = Field(description="Median length")
    min: int = Field(description="Minimum length")
    max: int = Field(description="Maximum length")
    std: float = Field(description="Standard deviation")


class DatasetStatistics(BaseModel):
    """Comprehensive dataset statistics."""

    split: str = Field(description="Dataset split")
    num_papers: int = Field(description="Number of papers")
    num_questions: int = Field(description="Number of questions")
    num_annotations: int = Field(description="Total number of answer annotations")
    answer_type_distribution: AnswerTypeStats = Field(description="Distribution of answer types")
    num_unanswerable: int = Field(description="Number of unanswerable questions")
    num_questions_with_evidence: int = Field(
        description="Number of questions with evidence annotations"
    )
    mean_evidence_per_question: float = Field(
        description="Average number of evidence passages per question"
    )
    paragraph_length: LengthStats = Field(description="Paragraph length statistics")
    document_length: LengthStats = Field(description="Document length statistics")
    num_multi_evidence_questions: int = Field(
        description="Number of questions requiring multiple evidence passages"
    )


def compute_length_stats(lengths: list[int]) -> LengthStats:
    """Compute length statistics.

    Args:
        lengths: List of lengths.

    Returns:
        LengthStats object.
    """
    if not lengths:
        return LengthStats(mean=0.0, median=0.0, min=0, max=0, std=0.0)

    import statistics

    mean = statistics.mean(lengths)
    median = statistics.median(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    try:
        std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    except statistics.StatisticsError:
        std = 0.0

    return LengthStats(mean=mean, median=median, min=min_len, max=max_len, std=std)


def compute_statistics(dataset: QASPERDataset) -> DatasetStatistics:
    """Compute comprehensive statistics for a dataset split.

    Args:
        dataset: QASPERDataset object.

    Returns:
        DatasetStatistics object.
    """
    logger.info(f"Computing statistics for {dataset.split} split")

    papers = dataset.papers

    # Basic counts
    num_papers = len(papers)
    num_questions = sum(len(paper.qas) for paper in papers)
    num_annotations = sum(len(qa.answers) for paper in papers for qa in paper.qas)

    # Answer type distribution
    answer_type_counter: Counter[str] = Counter()
    unanswerable_count = 0
    questions_with_evidence = 0
    evidence_counts = []

    for paper in papers:
        for qa in paper.qas:
            # Determine question type from first answer
            if qa.answers:
                question_type = qa.answers[0].answer.answer_type
                answer_type_counter[question_type] += 1
            else:
                answer_type_counter["unknown"] += 1

            # Check for unanswerable (any answer marked as unanswerable)
            for answer_ann in qa.answers:
                if answer_ann.answer.unanswerable:
                    unanswerable_count += 1
                    break  # Count each question once

            # Check for evidence
            has_evidence = False
            question_evidence_count = 0
            for answer_ann in qa.answers:
                # Evidence is in answer data
                evidence_list = answer_ann.answer.evidence
                highlighted_list = answer_ann.answer.highlighted_evidence
                evidence_count = len(evidence_list) + len(highlighted_list)

                if evidence_count > 0:
                    has_evidence = True
                    question_evidence_count += evidence_count

            if has_evidence:
                questions_with_evidence += 1
                evidence_counts.append(question_evidence_count)

    # Build answer type stats
    answer_type_stats = AnswerTypeStats(
        yes_no=answer_type_counter.get("yes/no", 0),
        span=answer_type_counter.get("span", 0),
        free_form=answer_type_counter.get("free-form", 0),
        multiple_choice=answer_type_counter.get("multiple-choice", 0),
        unanswerable=answer_type_counter.get("unanswerable", 0),
        unknown=answer_type_counter.get("unknown", 0),
    )

    # Evidence statistics
    mean_evidence = sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0.0
    num_multi_evidence = sum(1 for count in evidence_counts if count > 1)

    # Length statistics
    paragraph_lengths = []
    document_lengths = []

    for paper in papers:
        doc_length = 0
        for section in paper.full_text.sections:
            for paragraph in section.paragraphs:
                paragraph_lengths.append(len(paragraph))
                doc_length += len(paragraph)
        document_lengths.append(doc_length)

    paragraph_length_stats = compute_length_stats(paragraph_lengths)
    document_length_stats = compute_length_stats(document_lengths)

    stats = DatasetStatistics(
        split=dataset.split,
        num_papers=num_papers,
        num_questions=num_questions,
        num_annotations=num_annotations,
        answer_type_distribution=answer_type_stats,
        num_unanswerable=unanswerable_count,
        num_questions_with_evidence=questions_with_evidence,
        mean_evidence_per_question=mean_evidence,
        paragraph_length=paragraph_length_stats,
        document_length=document_length_stats,
        num_multi_evidence_questions=num_multi_evidence,
    )

    logger.info(f"Computed statistics: {num_papers} papers, {num_questions} questions")

    return stats


def export_statistics_json(stats: DatasetStatistics, output_path: Path) -> None:
    """Export statistics to JSON.

    Args:
        stats: DatasetStatistics object.
        output_path: Path to save JSON file.
    """
    logger.info(f"Exporting statistics to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats.model_dump(mode="json"), f, indent=2)

    logger.info(f"Successfully exported statistics to {output_path}")


def export_statistics_csv(stats: DatasetStatistics, output_path: Path) -> None:
    """Export statistics to CSV.

    Args:
        stats: DatasetStatistics object.
        output_path: Path to save CSV file.
    """
    logger.info(f"Exporting statistics to {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import csv

    # Flatten statistics for CSV
    data: dict[str, Any] = {
        "split": stats.split,
        "num_papers": stats.num_papers,
        "num_questions": stats.num_questions,
        "num_annotations": stats.num_annotations,
        "answer_type_yes_no": stats.answer_type_distribution.yes_no,
        "answer_type_span": stats.answer_type_distribution.span,
        "answer_type_free_form": stats.answer_type_distribution.free_form,
        "answer_type_multiple_choice": stats.answer_type_distribution.multiple_choice,
        "answer_type_unanswerable": stats.answer_type_distribution.unanswerable,
        "answer_type_unknown": stats.answer_type_distribution.unknown,
        "num_unanswerable": stats.num_unanswerable,
        "num_questions_with_evidence": stats.num_questions_with_evidence,
        "mean_evidence_per_question": stats.mean_evidence_per_question,
        "paragraph_length_mean": stats.paragraph_length.mean,
        "paragraph_length_median": stats.paragraph_length.median,
        "paragraph_length_min": stats.paragraph_length.min,
        "paragraph_length_max": stats.paragraph_length.max,
        "paragraph_length_std": stats.paragraph_length.std,
        "document_length_mean": stats.document_length.mean,
        "document_length_median": stats.document_length.median,
        "document_length_min": stats.document_length.min,
        "document_length_max": stats.document_length.max,
        "document_length_std": stats.document_length.std,
        "num_multi_evidence_questions": stats.num_multi_evidence_questions,
    }

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    logger.info(f"Successfully exported statistics to {output_path}")
