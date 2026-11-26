"""QASPER dataset download and ingestion pipeline."""

import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import ValidationError

from scientific_rag.config import Config
from scientific_rag.data.schemas import (
    AnswerAnnotation,
    AnswerData,
    FigureOrTable,
    FullText,
    FullTextSection,
    Paper,
    QASPERDataset,
    QuestionAnswer,
)

logger = logging.getLogger(__name__)

# Official dataset URLs
QASPER_VERSION = "0.3.0"
TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

TRAIN_FILE = "qasper-train-v0.3.json"
DEV_FILE = "qasper-dev-v0.3.json"
TEST_FILE = "qasper-test-v0.3.json"


class DatasetDownloadError(Exception):
    """Error during dataset download."""

    pass


class DatasetValidationError(Exception):
    """Error during dataset validation."""

    pass


def download_file(url: str, target_path: Path) -> None:
    """Download a file from URL to target path.

    Args:
        url: URL to download from.
        target_path: Local path to save the file.

    Raises:
        DatasetDownloadError: If download fails.
    """
    logger.info(f"Downloading {url} to {target_path}")

    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as f:
            f.write(response.content)

        logger.info(f"Successfully downloaded {target_path}")

    except requests.RequestException as e:
        raise DatasetDownloadError(f"Failed to download {url}: {e}") from e


def extract_tarball(tarball_path: Path, extract_dir: Path) -> None:
    """Extract a tarball to the specified directory.

    Args:
        tarball_path: Path to the tarball.
        extract_dir: Directory to extract to.
    """
    logger.info(f"Extracting {tarball_path} to {extract_dir}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    logger.info(f"Successfully extracted {tarball_path}")


def load_json(file_path: Path) -> Any:
    """Load a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON data.

    Raises:
        DatasetValidationError: If JSON parsing fails.
    """
    logger.info(f"Loading JSON from {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {file_path}")
        return data

    except json.JSONDecodeError as e:
        raise DatasetValidationError(f"Failed to parse JSON from {file_path}: {e}") from e


def parse_paper(paper_id: str, paper_data: dict[str, Any]) -> Paper:
    """Parse a single paper from raw data.

    Args:
        paper_id: Paper identifier.
        paper_data: Raw paper data from JSON.

    Returns:
        Parsed Paper object.

    Raises:
        DatasetValidationError: If parsing fails.
    """
    try:
        # Parse full text (list of sections)
        full_text_data = paper_data.get("full_text", [])
        if not isinstance(full_text_data, list):
            raise DatasetValidationError(f"full_text must be a list, got {type(full_text_data)}")

        sections = []
        for section_data in full_text_data:
            if not isinstance(section_data, dict):
                logger.warning(f"Skipping non-dict section: {type(section_data)}")
                continue

            section = FullTextSection(
                section_name=section_data.get("section_name") or "",
                paragraphs=section_data.get("paragraphs", []),
            )
            sections.append(section)

        full_text = FullText(sections=sections)

        # Parse question-answer pairs
        qas_data = paper_data.get("qas", [])
        qas = []
        for qa_data in qas_data:
            # Parse answers
            answers = []
            for answer_data in qa_data.get("answers", []):
                if not isinstance(answer_data, dict):
                    logger.warning(f"Skipping non-dict answer: {type(answer_data)}")
                    continue

                answer_info = answer_data.get("answer", {})
                if not isinstance(answer_info, dict):
                    logger.warning(f"Skipping non-dict answer info: {type(answer_info)}")
                    continue

                answer = AnswerData(
                    unanswerable=answer_info.get("unanswerable", False),
                    extractive_spans=answer_info.get("extractive_spans", []),
                    yes_no=answer_info.get("yes_no"),
                    free_form_answer=answer_info.get("free_form_answer", ""),
                    evidence=answer_info.get("evidence", []),
                    highlighted_evidence=answer_info.get("highlighted_evidence", []),
                )

                answer_annotation = AnswerAnnotation(
                    answer=answer,
                    annotation_id=answer_data.get("annotation_id", ""),
                    worker_id=answer_data.get("worker_id", ""),
                )
                answers.append(answer_annotation)

            # Infer question type from first answer
            question_type = None
            if answers:
                question_type = answers[0].answer.answer_type

            qa = QuestionAnswer(
                question_id=qa_data.get("question_id", ""),
                question=qa_data.get("question", ""),
                question_type=question_type,
                answers=answers,
            )
            qas.append(qa)

        # Parse figures and tables
        figures_and_tables_data = paper_data.get("figures_and_tables", [])
        figures_and_tables = [
            FigureOrTable(
                caption=ft.get("caption", ""),
                type=ft.get("type", ""),
            )
            for ft in figures_and_tables_data
            if isinstance(ft, dict)
        ]

        paper = Paper(
            id=paper_id,
            title=paper_data.get("title", ""),
            abstract=paper_data.get("abstract", ""),
            full_text=full_text,
            qas=qas,
            figures_and_tables=figures_and_tables,
        )

        return paper

    except (ValidationError, AttributeError, TypeError) as e:
        raise DatasetValidationError(f"Failed to parse paper {paper_id}: {e}") from e


def load_split(
    split: Literal["train", "validation", "test"],
    raw_dir: Path,
) -> QASPERDataset:
    """Load a specific dataset split.

    Args:
        split: Dataset split to load.
        raw_dir: Directory containing raw dataset files.

    Returns:
        Parsed QASPERDataset object.

    Raises:
        DatasetValidationError: If loading fails.
    """
    logger.info(f"Loading {split} split from {raw_dir}")

    # Determine file name
    if split == "train":
        file_name = TRAIN_FILE
        source_url = TRAIN_DEV_URL
    elif split == "validation":
        file_name = DEV_FILE
        source_url = TRAIN_DEV_URL
    elif split == "test":
        file_name = TEST_FILE
        source_url = TEST_URL
    else:
        raise ValueError(f"Unknown split: {split}")

    file_path = raw_dir / file_name

    if not file_path.exists():
        raise DatasetValidationError(
            f"Dataset file not found: {file_path}. "
            f"Please download the dataset first using download_dataset()."
        )

    # Load JSON
    data = load_json(file_path)

    # Parse papers
    papers = []
    parse_errors = 0
    for paper_id, paper_data in data.items():
        try:
            paper = parse_paper(paper_id, paper_data)
            papers.append(paper)
        except DatasetValidationError as e:
            logger.warning(f"Failed to parse paper {paper_id}: {e}")
            parse_errors += 1
            # Continue processing other papers

    logger.info(f"Loaded {len(papers)} papers from {split} split ({parse_errors} parse errors)")

    return QASPERDataset(
        papers=papers,
        split=split,
        version=QASPER_VERSION,
        source_url=source_url,
    )


def download_dataset(
    split: Literal["train", "validation", "test"] | None = None,
    raw_dir: Path | None = None,
) -> None:
    """Download the QASPER dataset.

    Args:
        split: Specific split to download. If None, downloads all splits.
        raw_dir: Directory to save raw data. If None, uses default from config.

    Raises:
        DatasetDownloadError: If download fails.
    """
    config = Config()
    target_dir = raw_dir or config.data.raw_dir
    target_dir = target_dir.absolute()

    logger.info(f"Downloading QASPER dataset to {target_dir}")

    # Download train/dev (together)
    if split is None or split in ["train", "validation"]:
        tarball_path = target_dir / "qasper-train-dev-v0.3.tgz"
        if not tarball_path.exists():
            download_file(TRAIN_DEV_URL, tarball_path)
            extract_tarball(tarball_path, target_dir)
        else:
            logger.info(f"Train/dev tarball already exists: {tarball_path}")

    # Download test (separate)
    if split is None or split == "test":
        tarball_path = target_dir / "qasper-test-and-evaluator-v0.3.tgz"
        if not tarball_path.exists():
            download_file(TEST_URL, tarball_path)
            extract_tarball(tarball_path, target_dir)
        else:
            logger.info(f"Test tarball already exists: {tarball_path}")

    logger.info("Dataset download complete")


def load_dataset(
    split: Literal["train", "validation", "test"],
    raw_dir: Path | None = None,
) -> QASPERDataset:
    """Load a dataset split (downloading if necessary).

    Args:
        split: Dataset split to load.
        raw_dir: Directory containing raw data. If None, uses default from config.

    Returns:
        Parsed QASPERDataset object.
    """
    config = Config()
    target_dir = raw_dir or config.data.raw_dir
    target_dir = target_dir.absolute()

    # Download if necessary
    try:
        return load_split(split, target_dir)
    except DatasetValidationError:
        logger.info(f"Dataset files not found, downloading {split} split")
        download_dataset(split, target_dir)
        return load_split(split, target_dir)
