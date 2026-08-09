from pathlib import Path

import pytest

from src.pipeline.orchestrator import build_all
from src.utils.config import PROCESSED_DIR


@pytest.fixture(scope="session")
def processed_dir() -> Path:
    """Build the feature store once for the whole integration session
    (fast tier: MINDsmall + ebnerd_demo). If any loader/validator regresses,
    this raises here and every dependent test fails loudly."""
    build_all()
    return PROCESSED_DIR
