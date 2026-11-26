"""Retrieval and evidence mapping utilities."""

from scientific_rag.retrieval.chunking import Chunk, Chunker, ChunkingConfig, ChunkingResult
from scientific_rag.retrieval.evidence import (
    EvidenceMapping,
    EvidenceMappingResult,
    MappingFailure,
    PassageIdentifier,
    annotation_evidence_sets,
    evidence_set_complete,
    inspect_failed_mappings,
    is_non_textual_evidence,
    map_evidence_string,
    map_paper_evidence,
    normalize_text,
    question_has_complete_evidence_set,
)

__all__ = [
    "Chunk",
    "Chunker",
    "ChunkingConfig",
    "ChunkingResult",
    "EvidenceMapping",
    "EvidenceMappingResult",
    "MappingFailure",
    "PassageIdentifier",
    "annotation_evidence_sets",
    "evidence_set_complete",
    "inspect_failed_mappings",
    "is_non_textual_evidence",
    "map_evidence_string",
    "map_paper_evidence",
    "normalize_text",
    "question_has_complete_evidence_set",
]
