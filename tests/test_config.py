"""Tests for configuration management."""

import pytest
from pydantic import ValidationError

from scientific_rag.config import (
    ChunkingConfig,
    Config,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    setup_logging,
)


def test_logging_config_defaults() -> None:
    """Test LoggingConfig default values."""
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert config.log_file is None


def test_logging_config_custom() -> None:
    """Test LoggingConfig with custom values."""
    config = LoggingConfig(level="DEBUG", log_file="test.log")
    assert config.level == "DEBUG"
    assert config.log_file.name == "test.log"


def test_data_config_defaults() -> None:
    """Test DataConfig default values."""
    config = DataConfig()
    assert config.raw_dir.name == "raw"
    assert config.processed_dir.name == "processed"
    assert config.results_dir.name == "results"


def test_chunking_config_defaults() -> None:
    """Test ChunkingConfig default values."""
    config = ChunkingConfig()
    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.tokenizer_name == "cl100k_base"


def test_chunking_config_validation() -> None:
    """Test ChunkingConfig validation."""
    # Valid configuration
    config = ChunkingConfig(chunk_size=256, chunk_overlap=50)
    assert config.chunk_size == 256
    assert config.chunk_overlap == 50

    # Invalid: overlap >= chunk_size
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)


def test_experiment_config_defaults() -> None:
    """Test ExperimentConfig default values."""
    config = ExperimentConfig()
    assert config.random_seed == 42
    assert config.split == "validation"


def test_config_defaults() -> None:
    """Test Config default values."""
    config = Config()
    assert isinstance(config.logging, LoggingConfig)
    assert isinstance(config.data, DataConfig)
    assert isinstance(config.chunking, ChunkingConfig)
    assert isinstance(config.experiment, ExperimentConfig)


def test_setup_logging() -> None:
    """Test logging setup."""
    config = LoggingConfig(level="DEBUG")
    setup_logging(config)

    # If no exception is raised, the test passes
    # In a real test, we could verify logging handlers
