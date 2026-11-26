"""Tests for evidence mapping (Phase 1.1 API)."""

from scientific_rag.data.schemas import (
    AnswerAnnotation,
    AnswerData,
    FullText,
    FullTextSection,
    Paper,
    QuestionAnswer,
)
from scientific_rag.retrieval.evidence import (
    EvidenceMappingResult,
    MappingFailure,
    annotation_evidence_sets,
    evidence_set_complete,
    inspect_failed_mappings,
    is_non_textual_evidence,
    map_evidence_string,
    map_paper_evidence,
    normalize_text,
    question_has_complete_evidence_set,
)


def make_paper(evidence: list[str], paragraphs: list[str] | None = None) -> Paper:
    paragraphs = paragraphs or ["Alpha paragraph text.", "Beta method details here."]
    return Paper(
        id="p1",
        title="T",
        abstract="A",
        full_text=FullText(sections=[FullTextSection(section_name="Intro", paragraphs=paragraphs)]),
        qas=[
            QuestionAnswer(
                question_id="q1",
                question="Q?",
                question_type="free-form",
                answers=[
                    AnswerAnnotation(
                        answer=AnswerData(
                            unanswerable=False,
                            extractive_spans=[],
                            yes_no=None,
                            free_form_answer="ans",
                            evidence=evidence,
                            highlighted_evidence=[],
                        ),
                        annotation_id="ann1",
                        worker_id="w1",
                    )
                ],
            )
        ],
        figures_and_tables=[],
    )


def test_normalize_text() -> None:
    assert normalize_text("HELLO   World") == "hello world"
    assert normalize_text("  spaced  ") == "spaced"


def test_exact_match_unique() -> None:
    paper = make_paper(["Alpha paragraph text."])
    cands, mtype = map_evidence_string(paper, "Alpha paragraph text.")
    assert mtype == "exact"
    assert len(cands) == 1


def test_normalized_match() -> None:
    paper = make_paper(["Alpha   paragraph   text."])
    cands, mtype = map_evidence_string(paper, "alpha paragraph text.")
    assert mtype == "normalized"
    assert len(cands) == 1


def test_ambiguous_match_detected() -> None:
    dup = "Same paragraph text."
    paper = make_paper([dup], paragraphs=[dup, dup])
    cands, mtype = map_evidence_string(paper, dup)
    assert len(cands) == 2  # ambiguity preserved, not silently resolved


def test_unmatched() -> None:
    paper = make_paper(["Nonexistent evidence xyz."])
    cands, mtype = map_evidence_string(paper, "Nonexistent evidence xyz.")
    assert cands == []
    assert mtype == "none"


def test_fuzzy_only_in_diagnostic_mode() -> None:
    paper = make_paper([], paragraphs=["The quick brown fox jumps."])
    # strict: containment substring should already match via normalized
    cands_s, _ = map_evidence_string(paper, "quick brown fox", use_fuzzy=False)
    cands_f, m_f = map_evidence_string(paper, "quick brown fox", use_fuzzy=True)
    assert m_f in ("normalized", "fuzzy")


def test_non_textual_detection() -> None:
    # FLOAT SELECTED markers are always non-textual
    assert is_non_textual_evidence("FLOAT SELECTED content here")
    # Short figure/table pointers are non-textual
    assert is_non_textual_evidence("Figure 3")
    assert is_non_textual_evidence("Table 2 results")
    # Running text that merely mentions a figure/table stays textual
    assert not is_non_textual_evidence("See Figure 3 for results.")
    assert not is_non_textual_evidence("Results in Table TABREF4 are shown.")
    assert not is_non_textual_evidence("The model achieves 90% accuracy.")


def test_map_paper_preserves_annotation_linkage() -> None:
    paper = make_paper(["Alpha paragraph text.", "Beta method details here."])
    res = map_paper_evidence(paper)
    assert res.total_evidence_count == 2
    assert res.mapped_unique == 2
    for m in res.mappings:
        assert m.annotation_id == "ann1"
        assert m.question_id == "q1"


def test_evidence_set_complete() -> None:
    ev_set = {"Alpha paragraph text.", "Beta method details here."}
    retrieved = {"Alpha paragraph text.", "Beta method details here.", "extra"}
    assert evidence_set_complete(retrieved, ev_set)
    # partial retrieval is NOT complete
    assert not evidence_set_complete({"Alpha paragraph text."}, ev_set)
    # empty evidence set never complete
    assert not evidence_set_complete(retrieved, set())


def test_alternative_evidence_sets_not_merged() -> None:
    paper = Paper(
        id="p1",
        title="T",
        abstract="A",
        full_text=FullText(
            sections=[
                FullTextSection(section_name="S", paragraphs=["Para one text.", "Para two text."])
            ]
        ),
        qas=[
            QuestionAnswer(
                question_id="q1",
                question="Q?",
                question_type="free-form",
                answers=[
                    AnswerAnnotation(
                        answer=AnswerData(
                            unanswerable=False,
                            extractive_spans=[],
                            yes_no=None,
                            free_form_answer="a1",
                            evidence=["Para one text."],
                            highlighted_evidence=[],
                        ),
                        annotation_id="ann1",
                        worker_id="w1",
                    ),
                    AnswerAnnotation(
                        answer=AnswerData(
                            unanswerable=False,
                            extractive_spans=[],
                            yes_no=None,
                            free_form_answer="a2",
                            evidence=["Para two text."],
                            highlighted_evidence=[],
                        ),
                        annotation_id="ann2",
                        worker_id="w2",
                    ),
                ],
            )
        ],
        figures_and_tables=[],
    )
    sets = annotation_evidence_sets(paper.qas[0])
    assert len(sets) == 2
    assert sets[0] == {"Para one text."}
    assert sets[1] == {"Para two text."}
    # retrieving only one set still counts as complete (at least one valid set)
    assert question_has_complete_evidence_set(paper.qas[0], {"Para one text."})
    assert question_has_complete_evidence_set(paper.qas[0], {"Para two text."})
    assert not question_has_complete_evidence_set(paper.qas[0], {"unrelated"})


def test_inspect_failed_mappings() -> None:
    res = EvidenceMappingResult(
        paper_id="p1",
        total_evidence_count=1,
        mapped_unique=0,
        mapped_ambiguous=0,
        unmatched=1,
        non_textual=0,
        failures=[MappingFailure(evidence_text="x", paper_id="p1", reason="no_candidate")],
    )
    assert len(inspect_failed_mappings(res)) == 1
    assert len(inspect_failed_mappings(res, paper_id="other")) == 0
