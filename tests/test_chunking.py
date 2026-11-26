"""Tests for text chunking."""

import pytest

from scientific_rag.data.schemas import FullText, FullTextSection, Paper
from scientific_rag.retrieval.chunking import Chunk, Chunker, ChunkingConfig, ChunkingResult


@pytest.fixture
def sample_paper() -> Paper:
    """Create a sample paper for testing."""
    return Paper(
        id="paper1",
        title="Test Paper",
        abstract="Test abstract",
        full_text=FullText(
            sections=[
                FullTextSection(
                    section_name="Introduction",
                    paragraphs=["This is the first paragraph."],
                ),
                FullTextSection(
                    section_name="Method",
                    paragraphs=["This is the method paragraph."],
                ),
            ]
        ),
        qas=[],
        figures_and_tables=[],
    )


def test_chunking_config_defaults() -> None:
    """Test ChunkingConfig default values."""
    config = ChunkingConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.tokenizer_name == "cl100k_base"


def test_chunking_config_validation() -> None:
    """Test ChunkingConfig validation."""
    # Valid
    config = ChunkingConfig(chunk_size=256, chunk_overlap=50)
    assert config.chunk_size == 256

    # Invalid: overlap >= size
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_chunker_initialization() -> None:
    """Test Chunker initialization."""
    config = ChunkingConfig(chunk_size=256, chunk_overlap=25)
    chunker = Chunker(config)
    assert chunker.config.chunk_size == 256
    assert chunker.tokenizer is not None


def test_count_tokens() -> None:
    """Test token counting."""
    config = ChunkingConfig()
    chunker = Chunker(config)

    count = chunker.count_tokens("Hello, world!")
    assert count > 0


def test_chunk_paragraph_short() -> None:
    """Test chunking a short paragraph (fits in one chunk)."""
    config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
    chunker = Chunker(config)

    paragraph = "This is a short paragraph."
    chunks = chunker.chunk_paragraph(paragraph, "paper1", "Introduction", 0)

    assert len(chunks) == 1
    assert chunks[0].text == paragraph
    assert chunks[0].token_count < config.chunk_size


def test_chunk_paragraph_level(sample_paper: Paper) -> None:
    """Test paragraph-level chunking (no splitting)."""
    config = ChunkingConfig()
    chunker = Chunker(config)

    result = chunker.chunk_paragraph_level(sample_paper)

    assert result.paper_id == "paper1"
    assert result.total_chunks == 2  # Two paragraphs
    assert result.total_tokens > 0

    # Verify chunk identifiers
    for chunk in result.chunks:
        assert chunk.chunk_id.passage_type == "paragraph"
        assert chunk.chunk_id.chunk_index is None


def test_chunk_paper(sample_paper: Paper) -> None:
    """Test chunking an entire paper."""
    config = ChunkingConfig(chunk_size=512, chunk_overlap=50)
    chunker = Chunker(config)

    result = chunker.chunk_paper(sample_paper)

    assert result.paper_id == "paper1"
    assert result.total_chunks > 0
    assert result.total_tokens > 0
    assert result.tokenizer_name == config.tokenizer_name


def test_chunk_provenance(sample_paper: Paper) -> None:
    """Test that chunks maintain source provenance."""
    config = ChunkingConfig()
    chunker = Chunker(config)

    result = chunker.chunk_paragraph_level(sample_paper)

    for chunk in result.chunks:
        assert chunk.source_section in ["Introduction", "Method"]
        assert chunk.source_paragraph_index >= 0
        assert chunk.chunk_id.paper_id == "paper1"


def test_chunking_result_schema() -> None:
    """Test ChunkingResult schema."""
    result = ChunkingResult(
        paper_id="paper1",
        chunks=[],
        total_chunks=0,
        total_tokens=0,
        tokenizer_name="cl100k_base",
    )

    assert result.paper_id == "paper1"
    assert result.total_chunks == 0


def test_chunk_schema() -> None:
    """Test Chunk schema."""
    from scientific_rag.retrieval.evidence import PassageIdentifier

    chunk = Chunk(
        chunk_id=PassageIdentifier(
            paper_id="paper1",
            section_name="Intro",
            paragraph_index=0,
            passage_type="paragraph",
        ),
        text="Test text",
        token_count=10,
        source_paragraph_index=0,
        source_section="Intro",
    )

    assert chunk.text == "Test text"
    assert chunk.token_count == 10
