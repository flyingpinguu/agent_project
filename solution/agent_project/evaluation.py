"""Evaluation and report rendering for the reference agent."""

from dataclasses import dataclass
from pathlib import Path

from solution.agent_project.agent import OperationsAgent
from solution.agent_project.config import Settings
from solution.agent_project.datasets import load_json
from solution.agent_project.memory import ConversationMemory
from solution.agent_project.models import AgentResult


@dataclass(frozen=True)
class CaseScore:
    """Per-case metrics used in the written report."""

    case_id: str
    status_ok: bool
    tools_ok: bool
    sources_ok: bool
    memory_ok: bool
    failure_ok: bool
    result: AgentResult


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate and row-level evaluation evidence."""

    rows: list[CaseScore]
    settings: Settings
    model_name: str = ""

    def metric(self, name: str) -> float:
        """Compute a ratio metric from boolean row attributes."""

        return (
            sum(getattr(row, name) for row in self.rows) / len(self.rows)
            if self.rows
            else 0.0
        )


def load_cases(path: Path) -> list[dict[str, object]]:
    """Load labeled evaluation cases."""

    return load_json(path)


def evaluate(
    agent: OperationsAgent, cases: list[dict[str, object]]
) -> EvaluationSummary:
    """Run the agent and score every labeled case."""

    rows: list[CaseScore] = []
    for case in cases:
        # Benchmark cases are independent, not turns in one conversation.
        agent.memory = ConversationMemory.load(agent.settings.operations_dir)
        agent.message_history = []
        result = agent.run_case(case)
        expected_tools = _string_set(case["expected_tools"])
        expected_sources = _string_set(case["expected_sources"])
        result_sources = set(result.citations)
        rows.append(
            CaseScore(
                case_id=str(case["case_id"]),
                status_ok=result.status == case["expected_status"],
                tools_ok=expected_tools <= set(result.tools_used),
                sources_ok=result_sources == expected_sources,
                memory_ok=result.memory_used is bool(case["must_use_memory"]),
                failure_ok=_failure_ok(case, result),
                result=result,
            )
        )
    return EvaluationSummary(
        rows, agent.settings, agent.model.model_name if agent.model else ""
    )


def _string_set(value: object) -> set[str]:
    """Convert a JSON list of strings into a typed set."""

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("Expected a list of strings.")
    return {str(item) for item in value}


def _failure_ok(case: dict[str, object], result: AgentResult) -> bool:
    if _string_set(case.get("forbidden_tools", [])) & set(result.tools_used):
        return False
    if not _string_set(case.get("expected_safety_events", [])) <= set(
        result.safety_events
    ):
        return False
    if case["expected_status"] == "abstained":
        return result.status == "abstained" and not result.citations
    if case["expected_status"] == "escalated":
        return result.status == "escalated" and "draft_escalation" in result.tools_used
    if case["expected_status"] == "review_required":
        return result.status == "review_required"
    return result.status == "answered"


def render_report(summary: EvaluationSummary) -> str:
    """Render the reference report required by the project brief."""

    lines = [
        "# Reference Evaluation Report",
        "",
        "## Configuration",
        "",
        (
            "The rule-based reference uses local JSON fixtures, "
            "paragraph-aware policy chunks, sparse lexical retrieval, "
            f"`TOP_K={summary.settings.top_k}`, `MIN_SCORE={summary.settings.min_score}`, "
            f"`MAX_AGENT_STEPS={summary.settings.max_agent_steps}`, and "
            f"`MAX_TOOL_CALLS={summary.settings.max_tool_calls}`. Each case starts with seed memory."
        ),
        "",
        "## Aggregate Results",
        "",
        (
            "These are workflow contract checks, not a measure of answer quality. "
            "Ticket updates are generic drafts; inspect policy-specific actions separately. "
            f"Each ratio uses all {len(summary.rows)} evaluated cases, including ordinary cases."
        ),
        "",
        "| Metric | Score |",
        "|---|---:|",
        f"| Expected status match | {summary.metric('status_ok'):.2f} |",
        f"| Required tool coverage | {summary.metric('tools_ok'):.2f} |",
        f"| Exact source-ID match | {summary.metric('sources_ok'):.2f} |",
        f"| Expected memory-flag match | {summary.metric('memory_ok'):.2f} |",
        f"| Outcome rule consistency | {summary.metric('failure_ok'):.2f} |",
        "",
        "## Per-Case Results",
        "",
        "| Case | Status | Tools | Citations | Memory | Safety events |",
        "|---|---|---|---|---|---|",
    ]
    if summary.model_name:
        lines[0] = "# Live Agent Evaluation Report"
        lines[4] = (
            f"Provider: `{summary.settings.llm_provider}`. Model: `{summary.model_name}`. "
            "PydanticAI selects tools over local fixtures and lexical retrieval. "
            f"`TOP_K={summary.settings.top_k}`, `MIN_SCORE={summary.settings.min_score}`, "
            f"`MAX_AGENT_STEPS={summary.settings.max_agent_steps}` model requests, "
            f"`MAX_TOOL_CALLS={summary.settings.max_tool_calls}` application tools. "
            "Each case starts with seed memory and empty message history. "
            "Live trajectories vary; source-ID checks do not prove semantic support."
        )
    for row in summary.rows:
        result = row.result
        lines.append(
            f"| {row.case_id} | {result.status} | {', '.join(result.tools_used)} | "
            f"{', '.join(result.citations) or '-'} | {result.memory_used} | "
            f"{', '.join(result.safety_events) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Failure And Fragility Analysis",
            "",
            (
                "- The supplied cafeteria case is intentionally unsupported. The agent should abstain "
                "even though it can retrieve weakly related operational language."
            ),
            (
                "- The retrieved-instruction case checks that the supplied attack example is "
                "removed from answer context and recorded as a safety event. A keyword filter "
                "does not establish general resistance to prompt injection."
            ),
            (
                "- The identity-review case checks that an unverified requester is routed to "
                "human review, and that an unavailable queue owner stops the escalation rather "
                "than drafting it."
            ),
            "",
            "## Threshold Decision",
            "",
            (
                "MIN_SCORE is an illustrative default, not a threshold calibrated on an "
                "independent dataset. These cases are regression fixtures, not an unbiased "
                "estimate of production performance. Tune retrieval on a separate development "
                "set, freeze the configuration, then evaluate on held-out queries."
            ),
            "",
            "## Prioritized Recommendations",
            "",
            (
                "1. Replace lexical retrieval with a dense retriever and preserve the same "
                "citation validation contract."
            ),
            (
                "2. Add human-reviewed trajectory labels for ambiguous escalation and "
                "identity-review cases."
            ),
            (
                "3. Connect tool stubs to sandboxed service APIs only after adding "
                "authorization checks and audit logging."
            ),
        ]
    )
    return "\n".join(lines) + "\n"
