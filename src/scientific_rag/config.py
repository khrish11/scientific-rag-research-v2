"""Configuration management for scientific RAG research."""

import logging
from pathlib import Path
from typing import Literal

import pydantic
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    log_file: Path | None = Field(
        default=None,
        description="Optional log file path",
    )

    model_config = SettingsConfigDict(env_prefix="LOG_")


class DataConfig(BaseSettings):
    """Data storage configuration."""

    raw_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory for raw dataset files",
    )
    processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed dataset files",
    )
    results_dir: Path = Field(
        default=Path("results"),
        description="Directory for experiment results",
    )

    @field_validator("raw_dir", "processed_dir", "results_dir")
    @classmethod
    def ensure_path(cls, v: Path) -> Path:
        """Ensure path is absolute."""
        return v.absolute()

    model_config = SettingsConfigDict(env_prefix="DATA_")


class ChunkingConfig(BaseSettings):
    """Text chunking configuration."""

    chunk_size: int = Field(
        default=512,
        description="Maximum tokens per chunk",
        ge=1,
    )
    chunk_overlap: int = Field(
        default=50,
        description="Number of overlapping tokens between chunks",
        ge=0,
    )
    tokenizer_name: str = Field(
        default="cl100k_base",
        description="Tokenizer identifier (tiktoken encoding name)",
    )

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: pydantic.ValidationInfo) -> int:
        """Ensure overlap is less than chunk size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")


class ExperimentConfig(BaseSettings):
    """Experiment configuration."""

    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    split: Literal["train", "validation", "test"] = Field(
        default="validation",
        description="Dataset split to use",
    )

    model_config = SettingsConfigDict(env_prefix="EXP_")


class Config(BaseSettings):
    """Main configuration class."""

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )


def setup_logging(config: LoggingConfig) -> None:
    """Configure structured logging.

    Args:
        config: Logging configuration.
    """
    logging.basicConfig(
        level=getattr(logging, config.level),
        format=config.format,
        handlers=[logging.StreamHandler()],
    )

    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(logging.Formatter(config.format))
        logging.getLogger().addHandler(file_handler)
