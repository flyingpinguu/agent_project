"""Select a native PydanticAI model without changing the agent's tools."""

import os

from pydantic_ai.models import Model, infer_model
from pydantic_ai.settings import ModelSettings

PROVIDERS = {
    "groq": ("groq", "openai/gpt-oss-20b"),
    "mistral": ("mistral", "ministral-14b-latest"),
    "openrouter": ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    "google": ("google", "gemma-4-26b-a4b-it"),
    "openai": ("openai", "gpt-6-luna"),
}


def model_from_env(provider: str | None = None) -> Model | None:
    """Default to Groq with its key; simulate only in explicit dry-run mode."""

    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()
    if provider == "dry-run":
        return None
    if provider not in PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER must be dry-run or one of {', '.join(PROVIDERS)}."
        )
    if not os.getenv(f"{provider.upper()}_API_KEY", "").strip():
        raise ValueError(f"Set {provider.upper()}_API_KEY for LLM_PROVIDER={provider}.")
    prefix, default = PROVIDERS[provider]
    name = os.getenv(f"{provider.upper()}_MODEL", default).strip()
    if not name:
        raise ValueError(f"{provider.upper()}_MODEL must not be blank.")
    return infer_model(f"{prefix}:{name}")


def model_settings(provider: str | None = None) -> ModelSettings:
    """Bound each request and keep OpenRouter's backup within that provider."""

    settings = ModelSettings(timeout=30, max_tokens=2048, max_retries=0)
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()
    if provider == "openrouter":
        backup = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/free").strip()
        settings["extra_body"] = {
            "provider": {"require_parameters": True},
            **({"models": [backup]} if backup else {}),
        }
    return settings
