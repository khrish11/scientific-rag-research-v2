"""Evidence mapping from annotations to retrievable passages.

Phase 1.1 corrections:
- Detect ambiguous matches (multiple candidate paragraphs) instead of
  silently returning the first match.
- Preserve annotation_id linkage for each evidence string.
- Flag non-textual evidence (FLOAT SELECTED markers, figure/table-only
  references) instead of treating it as ordinary paragraph text.
- Strict mode: exact + normalized containment only; fuzzy matching is an
  explicit diagnostic mode and never auto-accepted for gold labels.
"""

import logging
import re
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field

from scientific_rag.data.schemas import Paper, QuestionAnswer

logger = logging.getLogger(__name__)

# Markers indicating non-paragraph / non-textual evidence in QASPER.
FLOAT_SELECTED_RE = re.compile(r"FLOAT\s*SELECTED|FLOATSELECTED", re.IGNORECASE)
NON_TEXTUAL_RE = re.compile(
    r"\b(fig(?:ure)?s?\.?|tables?\.?|tabref\d+|figref\d+|bibr\d+)\b",
    re.IGNORECASE,
)


class PassageIdentifier(BaseModel):
    """Unique identifier for a passage."""

    paper_id: str = Field(description="Paper identifier")
    section_name: str = Field(default="", description="Section name")
    paragraph_index: int = Field(description="Paragraph index within section")
    passage_type: Literal["paragraph", "chunk"] = Field(
        default="paragraph", description="Type of passage"
    )
    chunk_index: int | None = Field(
        default=None, description="Chunk index (if passage_type is 'chunk')"
    )

    def __str__(self) -> str:
        if self.passage_type == "paragraph":
            return f"{self.paper_id}::{self.section_name}::{self.paragraph_index}"
        return (
            f"{self.paper_id}::{self.section_name}::{self.paragraph_index}::chunk{self.chunk_index}"
        )


class EvidenceMapping(BaseModel):
    """A single evidence string mapped to candidate passages."""

    evidence_text: str = Field(description="Original evidence string")
    annotation_id: str = Field(description="Annotation identifier")
    worker_id: str = Field(description="Annotator worker identifier")
    question_id: str = Field(description="Question identifier")
    candidate_passages: list[PassageIdentifier] = Field(
        description="All matching passages (may be >1 when ambiguous)"
    )
    match_type: Literal["exact", "normalized", "fuzzy", "none"] = Field(
        description="Strongest unambiguous match type"
    )
    is_ambiguous: bool = Field(
        default=False, description="True if multiple passages matched equally"
    )
    is_non_textual: bool = Field(
        default=False,
        description="True if evidence references figure/table/FLOAT SELECTED content",
    )
    passage_texts: list[str] = Field(default_factory=list, description="Text of candidate passages")


class MappingFailure(BaseModel):
    """Record of failed evidence mapping."""

    evidence_text: str = Field(description="Evidence text that could not be mapped")
    annotation_id: str = Field(default="", description="Annotation identifier")
    paper_id: str = Field(description="Paper identifier")
    question_id: str = Field(default="", description="Question identifier")
    reason: str = Field(description="Reason for mapping failure")


class EvidenceMappingResult(BaseModel):
    """Result of evidence mapping for a paper."""

    paper_id: str = Field(description="Paper identifier")
    total_evidence_count: int = Field(description="Total evidence strings")
    mapped_unique: int = Field(description="Evidence with exactly one candidate")
    mapped_ambiguous: int = Field(description="Evidence with multiple candidates")
    unmatched: int = Field(description="Evidence with no candidate")
    non_textual: int = Field(description="Evidence flagged non-textual")
    exact_matches: int = Field(default=0)
    normalized_matches: int = Field(default=0)
    fuzzy_matches: int = Field(default=0)
    mappings: list[EvidenceMapping] = Field(default_factory=list)
    failures: list[MappingFailure] = Field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize text for matching.

    Operations (documented, applied in order):
    1. Lowercase.
    2. Collapse all whitespace runs to a single space.
    3. Strip leading/trailing whitespace.
    4. Remove characters that are not word chars, whitespace, or the
       punctuation set ``.,;:!?-``.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"[^\w\s\.,;:!?-]", "", text)
    return text


def is_non_textual_evidence(text: str) -> bool:
    """Return True if the evidence string is non-textual / figure-table content.

    Rules: FLOAT SELECTED markers, or a short string that is essentially only
    a figure/table pointer (e.g. "Figure 3", "Table 2 results"). Long paragraph
    excerpts that merely mention a figure/table remain textual.
    """
    if FLOAT_SELECTED_RE.search(text):
        return True
    stripped = text.strip()
    return bool(NON_TEXTUAL_RE.match(stripped)) and len(stripped) < 200


def _iter_paragraphs(paper: Paper) -> Iterator[tuple[str, int, str]]:
    """Yield (section_name, paragraph_index, paragraph_text)."""
    for section in paper.full_text.sections:
        for idx, para in enumerate(section.paragraphs):
            yield section.section_name or "", idx, para


def map_evidence_string(
    paper: Paper,
    evidence_text: str,
    *,
    use_normalization: bool = True,
    use_fuzzy: bool = False,
) -> tuple[list[tuple[PassageIdentifier, str]], Literal["exact", "normalized", "fuzzy", "none"]]:
    """Find ALL candidate passages for an evidence string.

    Returns (candidates, match_type) where match_type is the strongest
    match level that produced candidates: 'exact', 'normalized', 'fuzzy',
    or 'none'. Ambiguity is preserved by returning every candidate.
    """
    # Pass 1: exact match
    exact = [
        (
            PassageIdentifier(
                paper_id=paper.id, section_name=s, paragraph_index=i, passage_type="paragraph"
            ),
            p,
        )
        for s, i, p in _iter_paragraphs(paper)
        if p == evidence_text
    ]
    if exact:
        return exact, "exact"

    norm_ev = normalize_text(evidence_text)

    # Pass 2: normalized equality, then normalized containment
    if use_normalization:
        norm_eq = [
            (
                PassageIdentifier(
                    paper_id=paper.id, section_name=s, paragraph_index=i, passage_type="paragraph"
                ),
                p,
            )
            for s, i, p in _iter_paragraphs(paper)
            if normalize_text(p) == norm_ev
        ]
        if norm_eq:
            return norm_eq, "normalized"

        norm_contain = [
            (
                PassageIdentifier(
                    paper_id=paper.id, section_name=s, paragraph_index=i, passage_type="paragraph"
                ),
                p,
            )
            for s, i, p in _iter_paragraphs(paper)
            if norm_ev and (norm_ev in normalize_text(p))
        ]
        if norm_contain:
            return norm_contain, "normalized"

    # Pass 3: fuzzy containment (diagnostic mode only)
    if use_fuzzy and norm_ev:
        fuzzy = [
            (
                PassageIdentifier(
                    paper_id=paper.id, section_name=s, paragraph_index=i, passage_type="paragraph"
                ),
                p,
            )
            for s, i, p in _iter_paragraphs(paper)
            if norm_ev in normalize_text(p) or normalize_text(p) in norm_ev
        ]
        if fuzzy:
            return fuzzy, "fuzzy"

    return [], "none"


def map_paper_evidence(
    paper: Paper,
    *,
    use_normalization: bool = True,
    use_fuzzy: bool = False,
) -> EvidenceMappingResult:
    """Map every evidence string in a paper, preserving annotation linkage."""
    result = EvidenceMappingResult(
        paper_id=paper.id,
        total_evidence_count=0,
        mapped_unique=0,
        mapped_ambiguous=0,
        unmatched=0,
        non_textual=0,
    )

    for qa in paper.qas:
        for ann in qa.answers:
            ad = ann.answer
            for ev_text in ad.evidence + ad.highlighted_evidence:
                result.total_evidence_count += 1
                non_text = is_non_textual_evidence(ev_text)
                if non_text:
                    result.non_textual += 1

                cands, mtype = map_evidence_string(
                    paper,
                    ev_text,
                    use_normalization=use_normalization,
                    use_fuzzy=use_fuzzy,
                )

                match_type: Literal["exact", "normalized", "fuzzy", "none"] = (
                    mtype if cands else "none"
                )
                mapping = EvidenceMapping(
                    evidence_text=ev_text,
                    annotation_id=ann.annotation_id,
                    worker_id=ann.worker_id,
                    question_id=qa.question_id,
                    candidate_passages=[c[0] for c in cands],
                    match_type=match_type,
                    is_ambiguous=len(cands) > 1,
                    is_non_textual=non_text,
                    passage_texts=[c[1] for c in cands],
                )
                result.mappings.append(mapping)

                if not cands:
                    result.unmatched += 1
                    result.failures.append(
                        MappingFailure(
                            evidence_text=ev_text,
                            annotation_id=ann.annotation_id,
                            paper_id=paper.id,
                            question_id=qa.question_id,
                            reason="non_textual" if non_text else "no_candidate",
                        )
                    )
                elif len(cands) > 1:
                    result.mapped_ambiguous += 1
                else:
                    result.mapped_unique += 1

                if mtype == "exact":
                    result.exact_matches += 1
                elif mtype == "normalized":
                    result.normalized_matches += 1
                elif mtype == "fuzzy":
                    result.fuzzy_matches += 1

    return result


def inspect_failed_mappings(
    result: EvidenceMappingResult,
    paper_id: str | None = None,
) -> list[MappingFailure]:
    """Get failed mappings for manual inspection."""
    failures = (
        result.failures
        if paper_id is None
        else [f for f in result.failures if f.paper_id == paper_id]
    )
    return failures


# ---------------------------------------------------------------------------
# Alternative evidence-set completeness predicate
# ---------------------------------------------------------------------------


def annotation_evidence_sets(qa: QuestionAnswer) -> list[set[str]]:
    """Return the list of distinct evidence sets, one per answer annotation.

    Each set contains the evidence strings belonging to that annotation.
    Alternative annotations are NOT merged.
    """
    sets = []
    for ann in qa.answers:
        ad = ann.answer
        ev = frozenset(ad.evidence + ad.highlighted_evidence)
        sets.append(set(ev))
    return sets


def evidence_set_complete(
    retrieved_passage_texts: set[str],
    evidence_set: set[str],
) -> bool:
    """Check whether retrieved passages cover an annotated evidence set.

    An evidence set is 'complete' only if EVERY evidence string in it is
    representable in the retrieved text (normalized containment). Empty or
    unmapped evidence strings can never be complete.

    NOTE: This measures annotated-evidence coverage, NOT semantic answer
    sufficiency.
    """
    if not evidence_set:
        return False  # empty evidence set is never 'complete'

    norm_retrieved = [normalize_text(t) for t in retrieved_passage_texts]
    for ev in evidence_set:
        norm_ev = normalize_text(ev)
        if not norm_ev:
            return False
        if not any(norm_ev in rt or rt in norm_ev for rt in norm_retrieved):
            return False
    return True


def question_has_complete_evidence_set(
    qa: QuestionAnswer,
    retrieved_passage_texts: set[str],
) -> bool:
    """True if retrieved text contains at least one complete annotated
    evidence set for this question."""
    for ev_set in annotation_evidence_sets(qa):
        if ev_set and evidence_set_complete(retrieved_passage_texts, ev_set):
            return True
    return False
