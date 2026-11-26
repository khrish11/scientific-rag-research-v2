"""Token-aware text chunking with deterministic segmentation."""

import logging

import pydantic
import tiktoken
from pydantic import BaseModel, Field, field_validator

from scientific_rag.data.schemas import Paper
from scientific_rag.retrieval.evidence import PassageIdentifier

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """A text chunk."""

    chunk_id: PassageIdentifier = Field(description="Unique chunk identifier")
    text: str = Field(description="Chunk text")
    token_count: int = Field(description="Number of tokens in chunk")
    source_paragraph_index: int = Field(description="Index of source paragraph within section")
    source_section: str = Field(description="Source section name")


class ChunkingResult(BaseModel):
    """Result of chunking a paper."""

    paper_id: str = Field(description="Paper identifier")
    chunks: list[Chunk] = Field(description="All chunks from the paper")
    total_chunks: int = Field(description="Total number of chunks")
    total_tokens: int = Field(description="Total tokens across all chunks")
    tokenizer_name: str = Field(description="Tokenizer used")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    chunk_size: int = Field(default=512, description="Maximum tokens per chunk", ge=1)
    chunk_overlap: int = Field(default=50, description="Overlap tokens between chunks", ge=0)
    tokenizer_name: str = Field(
        default="cl100k_base",
        description="Tokenizer encoding name (tiktoken)",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: pydantic.ValidationInfo) -> int:
        """Ensure overlap is less than chunk size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class Chunker:
    """Token-aware text chunker."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize chunker with configuration.

        Args:
            config: Chunking configuration.
        """
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.tokenizer_name)
        logger.info(
            f"Initialized chunker with tokenizer={config.tokenizer_name}, "
            f"chunk_size={config.chunk_size}, overlap={config.chunk_overlap}"
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text))

    def chunk_paragraph(
        self,
        paragraph: str,
        paper_id: str,
        section_name: str,
        paragraph_index: int,
    ) -> list[Chunk]:
        """Chunk a single paragraph.

        Args:
            paragraph: Paragraph text.
            paper_id: Paper identifier.
            section_name: Section name.
            paragraph_index: Paragraph index within section.

        Returns:
            List of chunks.
        """
        chunks: list[Chunk] = []

        # If paragraph fits in one chunk
        token_count = self.count_tokens(paragraph)
        if token_count <= self.config.chunk_size:
            chunk_id = PassageIdentifier(
                paper_id=paper_id,
                section_name=section_name,
                paragraph_index=paragraph_index,
                passage_type="chunk",
                chunk_index=0,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=paragraph,
                    token_count=token_count,
                    source_paragraph_index=paragraph_index,
                    source_section=section_name,
                )
            )
            return chunks

        # Split paragraph into chunks
        tokens = self.tokenizer.encode(paragraph)
        chunks_text = []

        for i in range(0, len(tokens), self.config.chunk_size - self.config.chunk_overlap):
            chunk_tokens = tokens[i : i + self.config.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks_text.append(chunk_text)

        # Create chunk objects
        for chunk_idx, chunk_text in enumerate(chunks_text):
            chunk_token_count = self.count_tokens(chunk_text)
            chunk_id = PassageIdentifier(
                paper_id=paper_id,
                section_name=section_name,
                paragraph_index=paragraph_index,
                passage_type="chunk",
                chunk_index=chunk_idx,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    token_count=chunk_token_count,
                    source_paragraph_index=paragraph_index,
                    source_section=section_name,
                )
            )

        return chunks

    def chunk_paper(self, paper: Paper) -> ChunkingResult:
        """Chunk all paragraphs in a paper.

        Args:
            paper: Paper object.

        Returns:
            ChunkingResult with all chunks.
        """
        logger.info(f"Chunking paper {paper.id}")

        all_chunks: list[Chunk] = []
        total_tokens = 0

        for section in paper.full_text.sections:
            for para_idx, paragraph in enumerate(section.paragraphs):
                chunks = self.chunk_paragraph(
                    paragraph, paper.id, section.section_name or "", para_idx
                )
                all_chunks.extend(chunks)

        total_tokens = sum(chunk.token_count for chunk in all_chunks)

        result = ChunkingResult(
            paper_id=paper.id,
            chunks=all_chunks,
            total_chunks=len(all_chunks),
            total_tokens=total_tokens,
            tokenizer_name=self.config.tokenizer_name,
        )

        logger.info(
            f"Chunked paper {paper.id}: {result.total_chunks} chunks, {total_tokens} total tokens"
        )

        return result

    def chunk_paragraph_level(self, paper: Paper) -> ChunkingResult:
        """Create paragraph-level chunks (no splitting).

        This provides a reference condition where each paragraph is a single chunk.

        Args:
            paper: Paper object.

        Returns:
            ChunkingResult with paragraph-level chunks.
        """
        logger.info(f"Creating paragraph-level chunks for paper {paper.id}")

        all_chunks: list[Chunk] = []
        total_tokens = 0

        for section in paper.full_text.sections:
            for para_idx, paragraph in enumerate(section.paragraphs):
                token_count = self.count_tokens(paragraph)
                chunk_id = PassageIdentifier(
                    paper_id=paper.id,
                    section_name=section.section_name or "",
                    paragraph_index=para_idx,
                    passage_type="paragraph",
                    chunk_index=None,
                )
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=paragraph,
                        token_count=token_count,
                        source_paragraph_index=para_idx,
                        source_section=section.section_name or "",
                    )
                )

        total_tokens = sum(chunk.token_count for chunk in all_chunks)

        result = ChunkingResult(
            paper_id=paper.id,
            chunks=all_chunks,
            total_chunks=len(all_chunks),
            total_tokens=total_tokens,
            tokenizer_name=self.config.tokenizer_name,
        )

        logger.info(f"Created {result.total_chunks} paragraph-level chunks for paper {paper.id}")

        return result
