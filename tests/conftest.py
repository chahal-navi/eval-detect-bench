"""Shared test configuration and fixtures.

Test gates (--runslow, RUN_SLOW_TESTS and so on), markers and shared fixtures
come from the pytest-inspect-evals plugin, which loads automatically.
"""

import logging
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    logging.basicConfig(level=logging.INFO)


@pytest.fixture(autouse=True)
def _eval_logs_in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write each test's eval logs to its own temporary directory, not ./logs."""
    monkeypatch.setenv("INSPECT_LOG_DIR", str(tmp_path))
