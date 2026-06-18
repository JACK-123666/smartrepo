"""Pytest configuration for benchmark tests."""

from pathlib import Path
import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Provide a temporary workspace directory for benchmark tests."""
    return tmp_path
