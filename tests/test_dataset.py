"""Tests for dataset loading and schemas."""

import json
from pathlib import Path

import pytest

from scientific_rag.data.dataset import (
    DatasetValidationError,
    load_json,
    parse_paper,
)
from scientific_rag.data.schemas import (
    AnswerAnnotation,
    AnswerData,
    FullText,
    FullTextSection,
    Paper,
    QuestionAnswer,
)


@pytest.fixture
def sample_paper_data() -> dict:
    """Sample paper data for testing."""
    return {
        "title": "Test Paper",
        "abstract": "This is a test abstract.",
        "full_text": [
            {
                "section_name": "Introduction",
                "paragraphs": ["This is the first paragraph.", "This is the second."],
            },
            {
                "section_name": "Method",
                "paragraphs": ["Method paragraph one."],
            },
        ],
        "qas": [
            {
                "question_id": "q1",
                "question": "What is the method?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": [],
                            "yes_no": None,
                            "free_form_answer": "The method is described in section 2.",
                            "evidence": ["Method paragraph one."],
                            "highlighted_evidence": [],
                        },
                        "annotation_id": "ann1",
                        "worker_id": "worker1",
                    }
                ],
            }
        ],
        "figures_and_tables": [{"caption": "Figure 1: Test figure", "type": "figure"}],
    }


def test_answer_data_schema() -> None:
    """Test AnswerData schema."""
    answer = AnswerData(
        unanswerable=False,
        extractive_spans=["span1", "span2"],
        yes_no=None,
        free_form_answer="",
        evidence=["evidence1"],
        highlighted_evidence=[],
    )
    assert answer.unanswerable is False
    assert len(answer.extractive_spans) == 2
    assert answer.answer_type == "span"
    assert answer.answer_text == "span1 span2"


def test_answer_annotation_schema() -> None:
    """Test AnswerAnnotation schema."""
    answer_data = AnswerData(
        unanswerable=False,
        extractive_spans=[],
        yes_no="yes",
        free_form_answer="",
        evidence=[],
        highlighted_evidence=[],
    )
    annotation = AnswerAnnotation(
        answer=answer_data,
        annotation_id="ann1",
        worker_id="worker1",
    )
    assert annotation.annotation_id == "ann1"
    assert annotation.worker_id == "worker1"
    assert annotation.answer.answer_type == "yes/no"


def test_question_answer_schema() -> None:
    """Test QuestionAnswer schema."""
    answer_data = AnswerData(
        unanswerable=False,
        extractive_spans=[],
        yes_no="yes",
        free_form_answer="",
        evidence=[],
        highlighted_evidence=[],
    )
    qa = QuestionAnswer(
        question_id="q1",
        question="Is this yes/no?",
        question_type="yes/no",
        answers=[
            AnswerAnnotation(
                answer=answer_data,
                annotation_id="ann1",
                worker_id="worker1",
            )
        ],
    )
    assert qa.question_id == "q1"
    assert qa.question_text == "Is this yes/no?"
    assert qa.question_type == "yes/no"
    assert len(qa.answers) == 1


def test_full_text_section_schema() -> None:
    """Test FullTextSection schema."""
    section = FullTextSection(
        section_name="Introduction",
        paragraphs=["Para 1", "Para 2"],
    )
    assert section.section_name == "Introduction"
    assert len(section.paragraphs) == 2


def test_full_text_schema() -> None:
    """Test FullText schema."""
    full_text = FullText(
        sections=[
            FullTextSection(section_name="Intro", paragraphs=["P1"]),
            FullTextSection(section_name="Method", paragraphs=["P2", "P3"]),
        ]
    )
    assert len(full_text.sections) == 2
    assert full_text.section_names == ["Intro", "Method"]
    assert full_text.all_paragraphs == [["P1"], ["P2", "P3"]]


def test_paper_schema(sample_paper_data: dict) -> None:
    """Test Paper schema."""
    paper = Paper(
        id="paper1",
        title=sample_paper_data["title"],
        abstract=sample_paper_data["abstract"],
        full_text=FullText(sections=[FullTextSection(**s) for s in sample_paper_data["full_text"]]),
        qas=sample_paper_data["qas"],  # type: ignore[arg-type]
        figures_and_tables=sample_paper_data["figures_and_tables"],  # type: ignore[arg-type]
    )
    assert paper.id == "paper1"
    assert paper.title == "Test Paper"
    assert len(paper.qas) == 1


def test_parse_paper(sample_paper_data: dict) -> None:
    """Test parse_paper function."""
    paper = parse_paper("paper1", sample_paper_data)
    assert paper.id == "paper1"
    assert paper.title == "Test Paper"
    assert len(paper.qas) == 1
    assert len(paper.figures_and_tables) == 1


def test_parse_paper_missing_fields() -> None:
    """Test parse_paper with missing fields."""
    incomplete_data = {"title": "Test"}

    # Should not raise exception for missing optional fields
    paper = parse_paper("paper1", incomplete_data)
    assert paper.id == "paper1"
    assert paper.title == "Test"
    assert paper.abstract == ""


def test_load_json(tmp_path: Path) -> None:
    """Test load_json function."""
    test_file = tmp_path / "test.json"
    test_data = {"key": "value"}

    with test_file.open("w") as f:
        json.dump(test_data, f)

    loaded = load_json(test_file)
    assert loaded == test_data


def test_load_json_invalid(tmp_path: Path) -> None:
    """Test load_json with invalid JSON."""
    test_file = tmp_path / "invalid.json"

    with test_file.open("w") as f:
        f.write("invalid json")

    with pytest.raises(DatasetValidationError):
        load_json(test_file)


def test_parse_paper_invalid_structure() -> None:
    """Test parse_paper with invalid structure that causes validation error."""
    # This tests the error handling in parse_paper
    invalid_data = {
        "title": "Test",
        "full_text": "not a list",  # Invalid type
    }

    with pytest.raises(DatasetValidationError):
        parse_paper("paper1", invalid_data)
