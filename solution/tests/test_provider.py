"""Provider setup uses native adapters and fails closed without credentials."""

from dataclasses import replace

import pytest
from pydantic_ai import models

from solution.agent_project.agent import OperationsAgent
from solution.agent_project.config import Settings
from solution.agent_project.provider import PROVIDERS, model_from_env, model_settings


@pytest.mark.parametrize("provider", PROVIDERS)
def test_native_adapter_and_model_override(provider, monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.setenv(f"{provider.upper()}_API_KEY", "offline-placeholder")
    name = "openai/custom-model" if provider == "openrouter" else "custom-model"
    monkeypatch.setenv(f"{provider.upper()}_MODEL", name)
    model = model_from_env(provider)
    assert model is not None
    assert model.model_name == name
    if provider == "openai":
        assert type(model).__name__ == "OpenAIResponsesModel"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_selected_provider_requires_its_own_key(provider):
    with pytest.raises(ValueError, match=f"{provider.upper()}_API_KEY"):
        model_from_env(provider)


def test_bad_provider_and_blank_model_fail(monkeypatch):
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        model_from_env("typo")
    monkeypatch.setenv("GROQ_API_KEY", "offline-placeholder")
    monkeypatch.setenv("GROQ_MODEL", " ")
    with pytest.raises(ValueError, match="GROQ_MODEL"):
        model_from_env("groq")


@pytest.mark.parametrize("provider", [None, "groq"])
def test_cli_configuration_activates_live_mode(settings, monkeypatch, provider):
    if provider is None:
        monkeypatch.delenv("LLM_PROVIDER")
    else:
        monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("GROQ_API_KEY", "offline-placeholder")
    configured = Settings.from_env(settings.project_root)
    assert configured.llm_provider == "groq"
    assert configured.report_file.name == "local_live_evaluation.md"
    assert type(OperationsAgent(configured).model).__name__ == "GroqModel"
    assert OperationsAgent(replace(configured, llm_provider="dry-run")).model is None


def test_default_groq_requires_its_key(settings, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER")
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        OperationsAgent(Settings.from_env(settings.project_root))


def test_openrouter_backup_is_optional(monkeypatch):
    assert model_settings("openrouter")["extra_body"] == {
        "provider": {"require_parameters": True},
        "models": ["openrouter/free"],
    }
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "")
    assert model_settings("openrouter")["extra_body"] == {
        "provider": {"require_parameters": True}
    }
