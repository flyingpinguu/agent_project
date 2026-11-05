import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from src.agent import Settings, execute_tool, run_agent
from src.tools import Tools


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("src.agent.MEMORY_DIR", tmp_path / "conversations")


def reply(tool=None, arguments=None):
    message = {"role": "assistant", "content": None}
    if tool:
        message["tool_calls"] = [{"id": "call-1", "type": "function", "function": {
            "name": tool, "arguments": json.dumps(arguments),
        }}]
    else:
        message["content"] = "Finished."
    return ChatCompletion.model_validate({
        "id": "response-test", "created": 0, "model": "test", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message}],
    })


class FakeClient:
    """Supply model replies locally, without using credentials or an API."""
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def workflow():
    return [
        reply("get_ticket", {"ticket_id": "T-1001"}),
        reply("get_customer_context", {"customer_id": "C-001"}),
        reply("search_policy", {"query": "duplicate payment refund"}),
        reply("draft_ticket_update", {"ticket_id": "T-1001", "source_ids": ["refunds"],
                                      "message": "Please provide the last four digits of your payment method."}),
    ]


def test_model_calls_real_local_tools_and_saves_result(tmp_path):
    tools = Tools(trace_dir=tmp_path)
    client = FakeClient(workflow())
    result = run_agent("T-1001", "Handle the ticket", client, tools, Settings())
    assert result.status == "answered"
    assert result.message.endswith("[refunds]")
    saved = json.loads(tools.trace_path.read_text())
    assert len(saved["calls"]) == 4
    assert len(saved["agent"]["steps"]) == 4
    assert saved["agent"]["result"] == result.model_dump()
    assert any(message["role"] == "tool" for message in saved["agent"]["messages"])
    assert client.requests[0]["model"] == "openai/gpt-6-luna"
    assert client.requests[0]["reasoning_effort"] == "none"


@pytest.mark.parametrize("settings, expected_calls, reason", [
    (Settings(max_steps=2), 2, "Model-step limit"),
    (Settings(max_tool_calls=2), 2, "Tool-call limit"),
])
def test_limits_stop_before_drafting(tmp_path, settings, expected_calls, reason):
    tools = Tools(trace_dir=tmp_path)
    result = run_agent("T-1001", "Handle", FakeClient(workflow()), tools, settings)
    assert result.status == "review_required"
    assert reason in result.message
    assert len(tools.trace) == expected_calls


def test_api_failure_is_saved_without_error_body(tmp_path):
    tools = Tools(trace_dir=tmp_path)
    client = FakeClient([RuntimeError("sensitive provider error body")])
    result = run_agent("T-1001", "Handle", client, tools, Settings())
    assert result.status == "review_required"
    saved_text = tools.trace_path.read_text()
    assert "sensitive provider error body" not in saved_text
    assert "RuntimeError" in saved_text


def test_model_cannot_answer_without_tools(tmp_path):
    result = run_agent("T-1001", "Handle", FakeClient([reply()]),
                       Tools(trace_dir=tmp_path), Settings())
    assert result.status == "review_required"


def test_no_evidence_returns_abstention(tmp_path):
    client = FakeClient([
        reply("get_ticket", {"ticket_id": "T-1005"}),
        reply("get_customer_context", {"customer_id": "C-005"}),
        reply("search_policy", {"query": "cafeteria menu"}), reply(),
    ])
    result = run_agent("T-1005", "Handle", client, Tools(trace_dir=tmp_path), Settings())
    assert result.status == "abstained"


def test_cross_customer_access_and_premature_drafting_are_rejected(tmp_path):
    tools = Tools(trace_dir=tmp_path)
    settings = Settings()
    execute_tool(tools, "get_ticket", '{"ticket_id":"T-1001"}', "T-1001", settings)
    with pytest.raises(ValueError, match="Customer does not belong"):
        execute_tool(tools, "get_customer_context", '{"customer_id":"C-002"}', "T-1001", settings)
    with pytest.raises(ValueError, match="Read ticket, customer and policy"):
        execute_tool(tools, "draft_ticket_update", json.dumps({
            "ticket_id": "T-1001", "message": "Hello", "source_ids": ["refunds"],
        }), "T-1001", settings)
    assert all(call.error for call in tools.trace[1:])


def test_unavailable_owner_prevents_ticket_update_too(tmp_path):
    tools = Tools(trace_dir=tmp_path)
    settings = Settings()
    for name, arguments in [
        ("get_ticket", {"ticket_id": "T-1007"}),
        ("get_customer_context", {"customer_id": "C-007"}),
        ("search_policy", {"query": "MFA account owner"}),
        ("check_availability", {"queue": "identity-review"}),
    ]:
        execute_tool(tools, name, json.dumps(arguments), "T-1007", settings)
    with pytest.raises(ValueError, match="Owner unavailable"):
        execute_tool(tools, "draft_ticket_update", json.dumps({
            "ticket_id": "T-1007", "message": "Reset MFA", "source_ids": ["account-access"],
        }), "T-1007", settings)


def test_followup_reloads_and_resends_the_full_conversation(tmp_path):
    memory_dir = tmp_path / "conversations"
    first = run_agent("T-1001", "Handle the duplicate payment", FakeClient(workflow()),
                      Tools(trace_dir=tmp_path / "traces"), Settings(), memory_dir=memory_dir)
    path = memory_dir / f"{first.conversation_id}.json"
    previous = json.loads(path.read_text())["messages"]
    assert previous[-1] == {"role": "assistant", "content": first.message}
    assert any(message["role"] == "tool" for message in previous)

    # Fresh objects simulate restarting the process; only the JSON file survives.
    client = FakeClient(workflow())
    second = run_agent(None, "Make that reply shorter", client,
                       Tools(trace_dir=tmp_path / "traces"), Settings(),
                       conversation_id=first.conversation_id, memory_dir=memory_dir)
    sent = client.requests[0]["messages"]
    assert sent[:-1] == previous
    assert "Make that reply shorter" in sent[-1]["content"]
    assert second.conversation_id == first.conversation_id
    saved = json.loads(path.read_text())
    assert saved["messages"][:len(previous)] == previous
    assert len(saved["messages"]) > len(previous)
    assert second.trace_path != first.trace_path


def test_conversation_cannot_be_reused_for_another_ticket(tmp_path):
    first = run_agent("T-1001", "Handle", FakeClient(workflow()),
                      Tools(trace_dir=tmp_path), Settings())
    client = FakeClient([])
    with pytest.raises(ValueError, match="one ticket"):
        run_agent("T-1002", "Different case", client, Tools(trace_dir=tmp_path),
                  Settings(), conversation_id=first.conversation_id)
    assert client.requests == []


def test_old_evidence_does_not_authorize_a_new_draft(tmp_path):
    first = run_agent("T-1001", "Handle", FakeClient(workflow()),
                      Tools(trace_dir=tmp_path), Settings())
    tools = Tools(trace_dir=tmp_path)
    second = run_agent(None, "Use the same evidence", FakeClient([workflow()[-1]]),
                       tools, Settings(max_steps=1), conversation_id=first.conversation_id)
    assert second.status == "review_required"
    assert tools.accepted_sources == set()
    assert tools.trace[0].error == "Read the ticket first"


def test_budget_exit_saves_responses_for_unexecuted_tool_calls(tmp_path):
    response = reply("get_ticket", {"ticket_id": "T-1001"})
    second_call = response.choices[0].message.tool_calls[0].model_copy(deep=True)
    second_call.id = "call-2"
    response.choices[0].message.tool_calls.append(second_call)
    result = run_agent("T-1001", "Handle", FakeClient([response]),
                       Tools(trace_dir=tmp_path), Settings(max_tool_calls=1))
    assert result.status == "review_required"
    client = FakeClient([reply()])
    run_agent(None, "Continue", client, Tools(trace_dir=tmp_path), Settings(),
              conversation_id=result.conversation_id)
    sent = client.requests[0]["messages"]
    tool_results = {message["tool_call_id"] for message in sent if message["role"] == "tool"}
    assert tool_results == {"call-1", "call-2"}


@pytest.mark.parametrize("conversation_id", ["missing", "../../outside"])
def test_missing_or_invalid_conversation_is_rejected(tmp_path, conversation_id):
    with pytest.raises(ValueError):
        run_agent(None, "Continue", FakeClient([]), Tools(trace_dir=tmp_path),
                  Settings(), conversation_id=conversation_id)
