"""Data schemas for QASPER dataset."""

from pydantic import BaseModel, Field


class AnswerData(BaseModel):
    """Structured answer data."""

    unanswerable: bool = Field(description="Whether the question is unanswerable")
    extractive_spans: list[str] = Field(
        default_factory=list,
        description="Extractive spans from the text",
    )
    yes_no: str | bool | None = Field(
        default=None,
        description="Yes/no answer if applicable (can be string 'yes'/'no' or boolean)",
    )
    free_form_answer: str = Field(
        default="",
        description="Free-form answer text",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence text passages",
    )
    highlighted_evidence: list[str] = Field(
        default_factory=list,
        description="Highlighted evidence passages",
    )

    @property
    def answer_type(self) -> str:
        """Determine the answer type."""
        if self.unanswerable:
            return "unanswerable"
        if self.yes_no is not None:
            return "yes/no"
        if self.extractive_spans:
            return "span"
        if self.free_form_answer:
            return "free-form"
        return "unknown"

    @property
    def answer_text(self) -> str:
        """Get the answer text."""
        if self.yes_no is not None:
            if isinstance(self.yes_no, bool):
                return "yes" if self.yes_no else "no"
            return str(self.yes_no)
        if self.extractive_spans:
            return " ".join(self.extractive_spans)
        if self.free_form_answer:
            return self.free_form_answer
        return ""


class AnswerAnnotation(BaseModel):
    """Answer annotation from a single annotator."""

    answer: AnswerData = Field(description="Answer data")
    annotation_id: str = Field(description="Annotation identifier")
    worker_id: str = Field(description="Worker/annotator identifier")


class QuestionAnswer(BaseModel):
    """A question-answer pair with annotations."""

    question_id: str = Field(description="Unique question identifier")
    question_text: str = Field(
        default="",
        alias="question",
        description="The question text",
    )
    question_type: str | None = Field(
        default=None,
        description="Type of answer (inferred from answers)",
    )
    answers: list[AnswerAnnotation] = Field(description="List of annotated answers")

    model_config = {"populate_by_name": True}


class SectionText(BaseModel):
    """A section of the paper with its paragraphs."""

    section_name: str = Field(description="Name of the section")
    paragraphs: list[str] = Field(description="Paragraphs in this section")


class FullTextSection(BaseModel):
    """A single section in the paper."""

    section_name: str | None = Field(
        default=None,
        description="Name of the section (can be None)",
    )
    paragraphs: list[str] = Field(description="Paragraphs in this section")


class FullText(BaseModel):
    """Structured full text of a paper (list of sections)."""

    sections: list[FullTextSection] = Field(description="List of sections")

    @property
    def section_names(self) -> list[str]:
        """Get list of section names."""
        return [s.section_name or "" for s in self.sections]

    @property
    def all_paragraphs(self) -> list[list[str]]:
        """Get list of paragraph lists per section."""
        return [s.paragraphs for s in self.sections]


class FigureOrTable(BaseModel):
    """A figure or table with caption."""

    caption: str = Field(description="Caption text")
    type: str = Field(description="Type: 'figure' or 'table'")


class Paper(BaseModel):
    """A single research paper in the dataset."""

    id: str = Field(description="Unique paper identifier")
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract")
    full_text: FullText = Field(description="Structured full text")
    qas: list[QuestionAnswer] = Field(description="Question-answer pairs")
    figures_and_tables: list[FigureOrTable] = Field(
        default_factory=list,
        description="Figures and tables with captions",
    )


class QASPERDataset(BaseModel):
    """The complete QASPER dataset."""

    papers: list[Paper] = Field(description="List of papers in the dataset")
    split: str = Field(description="Dataset split (train, validation, test)")
    version: str = Field(description="Dataset version")
    source_url: str = Field(description="URL where dataset was downloaded")
