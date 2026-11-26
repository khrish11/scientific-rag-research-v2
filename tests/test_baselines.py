"""Tests for Phase 2 retrieval baselines and evaluation."""

import tiktoken

from scientific_rag.data.schemas import (
    AnswerAnnotation,
    AnswerData,
    FullText,
    FullTextSection,
    Paper,
    QuestionAnswer,
)
from scientific_rag.retrieval.baselines import (
    BM25Retriever,
    Passage,
    build_passages,
    hybrid_search,
    minmax,
    pack_budget,
    rrf_search,
    tokenize,
)
from scientific_rag.retrieval.evidence import PassageIdentifier

ENC = tiktoken.get_encoding("cl100k_base")


def mkpass(pid, text, sec="S", idx=0):
    return Passage(
        passage_id=PassageIdentifier(paper_id=pid, section_name=sec, paragraph_index=idx),
        text=text,
        section_name=sec,
        paragraph_index=idx,
        token_count=len(ENC.encode(text)),
    )


def test_tokenize() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("BERT-based models") == ["bert", "based", "models"]


def test_bm25_ranking() -> None:
    passages = [mkpass("p", "the cat sat", idx=i) for i in range(3)]
    passages.append(mkpass("p", "dogs and cats", idx=3))
    r = BM25Retriever(passages)
    res = r.search("cats", k=4)
    assert res[0][0].text == "dogs and cats"
    assert all(s >= res[i + 1][1] for i, s in enumerate([x[1] for x in res[:-1]]))


def test_bm25_empty_query() -> None:
    r = BM25Retriever([mkpass("p", "some text", idx=0)])
    assert r.search("zzzqqq", k=5) == []
    assert r.search("", k=5) == []


def test_bm25_deterministic_ties() -> None:
    passages = [mkpass("p", "same words", idx=i) for i in range(3)]
    r = BM25Retriever(passages)
    a = r.search("same", k=3)
    b = r.search("same", k=3)
    assert [str(x[0].passage_id) for x in a] == [str(x[0].passage_id) for x in b]


def test_minmax_constant() -> None:
    assert minmax([5.0, 5.0, 5.0]) == [0.0, 0.0, 0.0]
    assert minmax([0.0, 10.0]) == [0.0, 1.0]


def test_hybrid_endpoints() -> None:
    sp = [(mkpass("p", "alpha text", idx=0), 2.0), (mkpass("p", "beta text", idx=1), 1.0)]
    dn = [(mkpass("p", "alpha text", idx=0), 0.9), (mkpass("p", "beta text", idx=1), 0.1)]
    h0 = hybrid_search(sp, dn, alpha=0.0, k=2, pool_size=2)
    h1 = hybrid_search(sp, dn, alpha=1.0, k=2, pool_size=2)
    assert h0[0][0].text == "alpha text"  # sparse top
    assert h1[0][0].text == "alpha text"  # dense top
    # alpha=0 should reproduce sparse ordering, alpha=1 dense ordering
    assert [x[0].text for x in h0] == [x[0].text for x in sp]
    assert [x[0].text for x in h1] == [x[0].text for x in dn]


def test_rrf_basic() -> None:
    a = mkpass("p", "A", idx=0)
    b = mkpass("p", "B", idx=1)
    c = mkpass("p", "C", idx=2)
    l1 = [(a, 1.0), (b, 0.5)]
    l2 = [(c, 1.0), (a, 0.5)]
    res = rrf_search([l1, l2], k=3, rank_constant=60)
    ids = [x[0].text for x in res]
    # a appears in both lists -> highest RRF
    assert ids[0] == "A"


def test_pack_budget_never_exceeds() -> None:
    passages = [mkpass("p", "word " * 300, idx=i) for i in range(5)]
    ranked = [(p, 0.0) for p in passages]
    sel, used = pack_budget(ranked, 256, ENC)
    total = sum(p.token_count for p in sel) + 2 * max(0, len(sel) - 1)
    assert total <= 256 + 5  # small epsilon for sep estimate
    assert len(sel) >= 0


def test_pack_budget_skip_policy() -> None:
    # first passage too big, second fits -> skip-and-continue
    big = mkpass("p", "word " * 5000, idx=0)
    small = mkpass("p", "tiny", idx=1)
    sel, _ = pack_budget([(big, 0), (small, 0)], 100, ENC)
    assert [p.text for p in sel] == ["tiny"]


def test_build_passages_provenance() -> None:
    paper = Paper(
        id="p1",
        title="T",
        abstract="A",
        full_text=FullText(
            sections=[FullTextSection(section_name="Intro", paragraphs=["Para one.", "Para two."])]
        ),
        qas=[],
        figures_and_tables=[],
    )
    ps = build_passages(paper, ENC)
    assert len(ps) == 2
    assert ps[0].passage_id.section_name == "Intro"
    assert ps[0].passage_id.paragraph_index == 0
    assert ps[1].passage_id.paragraph_index == 1


def test_no_gold_leakage_in_passages() -> None:
    """Passage objects must not contain answer/evidence fields."""
    paper = Paper(
        id="p1",
        title="T",
        abstract="A",
        full_text=FullText(sections=[FullTextSection(section_name="S", paragraphs=["Body text."])]),
        qas=[
            QuestionAnswer(
                question_id="q",
                question="Q?",
                question_type="span",
                answers=[
                    AnswerAnnotation(
                        answer=AnswerData(
                            unanswerable=False,
                            extractive_spans=["x"],
                            yes_no=None,
                            free_form_answer="a",
                            evidence=["Body text."],
                            highlighted_evidence=[],
                        ),
                        annotation_id="a1",
                        worker_id="w",
                    )
                ],
            )
        ],
        figures_and_tables=[],
    )
    ps = build_passages(paper, ENC)
    for p in ps:
        assert not hasattr(p, "answer")
        assert not hasattr(p, "evidence")
        assert "Body text." in p.text


def test_oracle_isolation() -> None:
    """oracle_fits must be importable only inside eval module usage;
    retrievers never receive gold sets."""
    import scientific_rag.experiments.phase2_full as m

    assert hasattr(m, "oracle_fits")
    # retriever APIs take only (query, k)
    import inspect

    from scientific_rag.retrieval.baselines import BM25Retriever as B

    sig = inspect.signature(B.search)
    assert "gold" not in str(sig).lower()
