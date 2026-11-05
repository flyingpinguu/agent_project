"""Shared test fixtures for the reference solution."""

from pathlib import Path

import pytest

from solution.agent_project.agent import OperationsAgent
from solution.agent_project.config import Settings


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    """Keep local credentials and .env files out of every regression test."""

    monkeypatch.setattr("pydantic_ai.models.ALLOW_MODEL_REQUESTS", False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "dry-run")
    for provider in ("GROQ", "MISTRAL", "OPENROUTER", "GOOGLE", "OPENAI"):
        monkeypatch.delenv(f"{provider}_API_KEY", raising=False)


@pytest.fixture()
def project_root() -> Path:
    return Path(__file__).parents[2]


@pytest.fixture()
def settings(project_root: Path) -> Settings:
    return Settings.from_env(project_root=project_root)


@pytest.fixture()
def agent(settings: Settings) -> OperationsAgent:
    return OperationsAgent(settings)
