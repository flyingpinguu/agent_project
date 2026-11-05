"""Configuration helpers for the reference solution."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Resolved project paths, execution mode, and agent limits."""

    project_root: Path
    knowledge_base_dir: Path
    operations_dir: Path
    evaluation_file: Path
    report_file: Path
    top_k: int = 3
    min_score: float = 0.18
    max_agent_steps: int = 8
    max_tool_calls: int = 6
    llm_provider: str = "groq"

    def __post_init__(self) -> None:
        if self.top_k < 1 or not 0 < self.min_score <= 1:
            raise ValueError("TOP_K must be positive and MIN_SCORE must be in (0, 1].")
        if self.max_agent_steps < 1 or self.max_tool_calls < 1:
            raise ValueError("Execution limits must be positive.")

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        """Create settings from environment variables with repo-local defaults."""

        root = project_root or Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        return cls(
            project_root=root,
            knowledge_base_dir=root / "data" / "knowledge_base",
            operations_dir=root / "data" / "operations",
            evaluation_file=root / "data" / "evaluation" / "agent_cases.json",
            report_file=root
            / "reports"
            / (
                "local_evaluation.md"
                if os.getenv("LLM_PROVIDER", "groq").strip().lower() == "dry-run"
                else "local_live_evaluation.md"
            ),
            top_k=int(os.getenv("TOP_K", "3")),
            min_score=float(os.getenv("MIN_SCORE", "0.18")),
            max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "8")),
            max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "6")),
            llm_provider=os.getenv("LLM_PROVIDER", "groq").strip().lower(),
        )
