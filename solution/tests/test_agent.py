"""Tests for agent outcomes, tools, memory, and failure behavior."""

from solution.agent_project.evaluation import load_cases


def _case(settings, case_id: str) -> dict[str, object]:
    return next(
        case
        for case in load_cases(settings.evaluation_file)
        if case["case_id"] == case_id
    )


def test_supported_refund_case_uses_tools_and_citation(agent, settings) -> None:
    result = agent.run_case(_case(settings, "refund_duplicate_payment"))

    assert result.status == "answered"
    assert result.citations == ["refunds"]
    assert {
        "get_ticket",
        "get_customer_context",
        "search_policy",
        "draft_ticket_update",
    } <= set(result.tools_used)


def test_followup_case_uses_seeded_memory(agent, settings) -> None:
    result = agent.run_case(_case(settings, "mfa_followup_memory"))

    assert result.status == "answered"
    assert result.memory_used
    assert result.citations == ["account-access"]


def test_unsupported_case_abstains(agent, settings) -> None:
    result = agent.run_case(_case(settings, "unsupported_cafeteria_policy"))

    assert result.status == "abstained"
    assert not result.citations


def test_incident_case_escalates_with_availability(agent, settings) -> None:
    result = agent.run_case(_case(settings, "checkout_incident_escalation"))

    assert result.status == "escalated"
    assert "check_availability" in result.tools_used
    assert "draft_escalation" in result.tools_used
