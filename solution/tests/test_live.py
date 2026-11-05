"""Exercise the live orchestration path with scripted model replies, never API keys."""

import json
from dataclasses import replace

import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from solution.agent_project.agent import OperationsAgent
from solution.agent_project.evaluation import evaluate, load_cases, render_report


def reply(messages, info):
    start, prompt = [
        (i, part.content)
        for i, message in enumerate(messages)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ][-1]
    payload = json.loads(prompt)
    assert "expected_status" not in payload and "expected_sources" not in payload
    ticket_id = payload["ticket_id"]
    returned = {
        part.tool_name: part.content
        for message in messages[start:]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    if "get_ticket" not in returned:
        parts = [ToolCallPart("get_ticket", {"ticket_id": ticket_id})]
    elif "get_customer_context" not in returned:
        parts = [
            ToolCallPart(
                "get_customer_context",
                {"customer_id": returned["get_ticket"]["customer_id"]},
            )
        ]
    elif "search_policy" not in returned:
        parts = [ToolCallPart("search_policy", {"query": payload["request"]})]
    elif "draft_ticket_update" not in returned:
        parts = [
            ToolCallPart(
                "draft_ticket_update",
                {
                    "ticket_id": ticket_id,
                    "customer_name": returned["get_customer_context"]["name"],
                    "citations": ["refunds"],
                },
            )
        ]
    else:
        parts = [
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "status": "answered",
                    "answer": "Review the duplicate payment under [refunds].",
                    "citations": ["refunds"],
                },
            )
        ]
    return ModelResponse(parts=parts)


def test_live_loop_uses_real_tools_and_updates_memory(settings):
    agent = OperationsAgent(settings, model=FunctionModel(reply))
    case = load_cases(settings.evaluation_file)[0]
    result = agent.run_case(case)
    assert result.status == "answered"
    assert result.citations == ["refunds"]
    assert result.tools_used == [
        "get_ticket",
        "get_customer_context",
        "search_policy",
        "draft_ticket_update",
    ]
    assert agent.memory.last_ticket_id == "T-1001"
    assert not agent.memory.notes
    assert agent.message_history


@pytest.mark.parametrize("limit", ["max_agent_steps", "max_tool_calls"])
def test_live_limits_stop_the_loop(settings, limit):
    agent = OperationsAgent(replace(settings, **{limit: 1}), model=FunctionModel(reply))
    result = agent.run_case(load_cases(settings.evaluation_file)[0])
    assert result.status == "review_required"
    assert "execution_budget_exhausted" in result.safety_events


@pytest.mark.parametrize("fault", ["citation", "missing_tools", "blank", "http"])
def test_live_failure_is_never_a_successful_simulation(settings, fault):
    def faulty(messages, info):
        if fault == "http":
            raise ModelHTTPError(429, "offline-test")
        if fault == "missing_tools":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {
                            "status": "answered",
                            "answer": "Proceed.",
                            "citations": [],
                        },
                    )
                ]
            )
        response = reply(messages, info)
        if response.parts[0].tool_name == info.output_tools[0].name:
            response.parts[0].args = {
                "status": "answered",
                "answer": " " if fault == "blank" else "Use [invented].",
                "citations": ["invented"],
            }
        return response

    agent = OperationsAgent(settings, model=FunctionModel(faulty))
    case = load_cases(settings.evaluation_file)[0]
    if fault in {"http", "blank"}:
        with pytest.raises(
            ModelHTTPError if fault == "http" else UnexpectedModelBehavior
        ):
            agent.run_case(case)
    else:
        result = agent.run_case(case)
        assert result.status == (
            "invalid_citations" if fault == "citation" else "review_required"
        )


@pytest.mark.parametrize(
    ("answer", "citations"),
    [
        ("Review the duplicate payment under the refund policy.", ["refunds"]),
        ("Review the duplicate payment under [refunds].", []),
    ],
)
def test_live_rejects_inconsistent_citations(settings, answer, citations):
    def inconsistent(messages, info):
        response = reply(messages, info)
        if response.parts[0].tool_name == info.output_tools[0].name:
            response.parts[0].args = {
                "status": "answered",
                "answer": answer,
                "citations": citations,
            }
        return response

    agent = OperationsAgent(settings, model=FunctionModel(inconsistent))
    case = load_cases(settings.evaluation_file)[0]
    row = evaluate(agent, [case]).rows[0]
    assert row.result.status == "invalid_citations"
    assert not row.result.citations
    assert "invalid_citations" in row.result.safety_events
    assert not row.status_ok and not row.failure_ok


def test_live_report_is_labeled_and_evaluation_resets_history(settings):
    lengths = []

    def recording(messages, info):
        lengths.append(
            sum(isinstance(p, UserPromptPart) for m in messages for p in m.parts)
        )
        return reply(messages, info)

    agent = OperationsAgent(settings, model=FunctionModel(recording))
    case = load_cases(settings.evaluation_file)[0]
    summary = evaluate(agent, [case, case])
    assert set(lengths) == {1}
    assert render_report(summary).startswith("# Live Agent Evaluation Report")


def test_followup_receives_real_message_history(settings):
    lengths = []

    def recording(messages, info):
        lengths.append(
            sum(isinstance(p, UserPromptPart) for m in messages for p in m.parts)
        )
        return reply(messages, info)

    agent = OperationsAgent(settings, model=FunctionModel(recording))
    case = load_cases(settings.evaluation_file)[0]
    agent.run_case(case)
    result = agent.run_case({**case, "ticket_id": None})
    assert result.status == "answered" and result.memory_used
    assert 2 in lengths


def test_customer_change_starts_with_empty_message_history(settings):
    lengths = []

    def recording(messages, info):
        prompts = [
            p for m in messages for p in m.parts if isinstance(p, UserPromptPart)
        ]
        if json.loads(prompts[-1].content)["ticket_id"] == "T-OTHER":
            lengths.append(len(prompts))
        return reply(messages, info)

    agent = OperationsAgent(settings, model=FunctionModel(recording))
    case = load_cases(settings.evaluation_file)[0]
    assert agent.run_case(case).status == "answered"
    agent.tools.tickets["T-OTHER"] = {
        **agent.tools.tickets["T-1001"],
        "ticket_id": "T-OTHER",
        "customer_id": "C-002",
    }
    result = agent.run_case({**case, "ticket_id": "T-OTHER"})
    assert result.status == "answered"
    assert lengths and set(lengths) == {1}
    assert agent.memory.last_customer_id == "C-002"


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("owner", ["Iris Novak", "wrong-owner"])
def test_live_handoff_requires_an_available_owner(settings, available, owner):
    def escalation(messages, info):
        returns = [
            p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
        ]
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart("get_ticket", {"ticket_id": "T-1003"}),
                    ToolCallPart("get_customer_context", {"customer_id": "C-003"}),
                    ToolCallPart(
                        "search_policy",
                        {"query": "checkout incident outage escalation"},
                    ),
                    ToolCallPart("check_availability", {"queue": "incident-response"}),
                ]
            )
        if not any(p.tool_name == "draft_escalation" for p in returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "draft_escalation",
                        {
                            "ticket_id": "T-1003",
                            "queue": "incident-response",
                            "owner": owner,
                            "citations": ["incidents"],
                        },
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "status": "escalated",
                        "answer": "Draft a handoff under [incidents].",
                        "citations": ["incidents"],
                    },
                )
            ]
        )

    agent = OperationsAgent(settings, model=FunctionModel(escalation))
    for row in agent.tools.availability:
        if row["queue"] == "incident-response":
            row["available"] = available
    case = load_cases(settings.evaluation_file)[2]
    result = agent.run_case(case)
    valid = available and owner == "Iris Novak"
    assert result.status == ("escalated" if valid else "review_required")
    draft = next(call for call in result.tool_calls if call.name == "draft_escalation")
    assert draft.args["owner"] == owner
    if not valid:
        assert "escalation_rejected" in result.safety_events


def test_live_identity_review_records_unavailable_owner_without_drafting(settings):
    def review(messages, info):
        returned = [
            p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
        ]
        if not returned:
            return ModelResponse(
                parts=[
                    ToolCallPart("get_ticket", {"ticket_id": "T-1007"}),
                    ToolCallPart("get_customer_context", {"customer_id": "C-007"}),
                    ToolCallPart(
                        "search_policy", {"query": "MFA account identity verification"}
                    ),
                    ToolCallPart("check_availability", {"queue": "identity-review"}),
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "status": "review_required",
                        "answer": "The owner is unavailable; request review [account-access].",
                        "citations": ["account-access"],
                    },
                )
            ]
        )

    agent = OperationsAgent(settings, model=FunctionModel(review))
    case = load_cases(settings.evaluation_file)[-1]
    row = evaluate(agent, [case]).rows[0]
    assert row.status_ok and row.tools_ok and row.sources_ok and row.memory_ok
    assert row.failure_ok
    assert "draft_escalation" not in row.result.tools_used
    assert "owner_unavailable" in row.result.safety_events
