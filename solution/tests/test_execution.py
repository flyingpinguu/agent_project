"""Regression tests for execution limits, conversation state, and configuration."""

from dataclasses import replace

import pytest

from solution.agent_project.agent import OperationsAgent
from solution.agent_project.config import Settings
from solution.agent_project.evaluation import evaluate, load_cases, render_report


@pytest.mark.parametrize("limit", ["max_agent_steps", "max_tool_calls"])
def test_budget_stops_before_next_tool(settings, limit):
    agent = OperationsAgent(replace(settings, **{limit: 2}))
    case = load_cases(settings.evaluation_file)[0]
    result = agent.run_case(case)
    assert result.status == "review_required"
    assert len(result.tool_calls) == 2
    assert "draft_ticket_update" not in result.tools_used


def test_conversation_remembers_explicit_ticket(agent, settings):
    case = load_cases(settings.evaluation_file)[0]
    agent.run_case(case)
    result = agent.run_case({**case, "case_id": "followup", "ticket_id": None})
    assert result.memory_used
    assert agent.memory.last_ticket_id == "T-1001"
    assert result.tool_calls[0].args == {"ticket_id": "T-1001"}


def test_memory_customer_mismatch_stops_lookup(agent):
    agent.memory.last_customer_id = "another-customer"
    result = agent.run_case(
        {"case_id": "mismatch", "ticket_id": None, "prompt": "Follow up"}
    )
    assert result.status == "review_required"
    assert result.tools_used == ["get_ticket"]


def test_missing_ticket_requires_review(agent):
    result = agent.run_case(
        {"case_id": "missing", "ticket_id": "absent", "prompt": "Help"}
    )
    assert result.status == "review_required"
    assert not result.citations


@pytest.mark.parametrize("ticket_id", ["T-1001", None])
@pytest.mark.parametrize(
    "prompt", ["Who won the 2022 football World Cup?", "What is the weather in Tokyo?"]
)
def test_unrelated_request_cannot_borrow_ticket_evidence(agent, ticket_id, prompt):
    result = agent.run_case(
        {"case_id": "off_topic", "ticket_id": ticket_id, "prompt": prompt}
    )
    assert result.status == "abstained"
    assert not result.citations
    assert "draft_ticket_update" not in result.tools_used
    assert "draft_escalation" not in result.tools_used


@pytest.mark.parametrize("queue_exists", [True, False])
def test_unavailable_identity_owner_requires_review(agent, queue_exists):
    agent.tools.customers["C-002"]["verified"] = False
    if not queue_exists:
        agent.tools.availability = []
    result = agent.run_case(
        {
            "case_id": "identity_handoff",
            "ticket_id": "T-1002",
            "prompt": "Help me recover MFA access to my account.",
        }
    )
    assert result.status == "review_required"
    assert "owner_unavailable" in result.safety_events
    assert "check_availability" in result.tools_used
    assert "draft_escalation" not in result.tools_used


def test_evaluation_is_independent_of_case_order(agent, settings):
    cases = load_cases(settings.evaluation_file)
    normal = evaluate(agent, cases)
    reverse = evaluate(agent, list(reversed(cases)))
    assert {r.case_id: r.result for r in normal.rows} == {
        r.case_id: r.result for r in reverse.rows
    }
    assert all(
        normal.metric(name) == 1
        for name in ("status_ok", "tools_ok", "sources_ok", "memory_ok", "failure_ok")
    )


def test_report_uses_actual_limits(settings):
    agent = OperationsAgent(replace(settings, max_tool_calls=2))
    summary = evaluate(agent, load_cases(settings.evaluation_file))
    assert "`MAX_TOOL_CALLS=2`" in render_report(summary)
    assert summary.metric("status_ok") < 1


@pytest.mark.parametrize(
    "changes",
    [
        {"top_k": 0},
        {"min_score": 0},
        {"min_score": float("nan")},
        {"max_tool_calls": 0},
    ],
)
def test_invalid_settings_rejected(settings, changes):
    with pytest.raises(ValueError):
        replace(settings, **changes)


def test_dotenv_is_loaded_without_overriding_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setenv("TOP_K", "2")
    monkeypatch.setenv("MIN_SCORE", "0.18")
    monkeypatch.delenv("MIN_SCORE", raising=False)
    # Keep the test's dotenv additions isolated from other tests.
    monkeypatch.setenv("MAX_AGENT_STEPS", "6")
    (tmp_path / ".env").write_text("TOP_K=9\nMIN_SCORE=0.42\n")
    try:
        config = Settings.from_env(tmp_path)
        assert config.top_k == 2
        assert config.min_score == 0.42
    finally:
        monkeypatch.delenv("MIN_SCORE", raising=False)
