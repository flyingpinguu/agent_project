import json
import shutil

import pytest
from pydantic import ValidationError

from src.tools import (
    AvailabilityInput, CustomerInput, EscalationInput, PolicySearchInput,
    TicketInput, TicketUpdateInput, Tools,
)
from src.validate_data import DATA_DIR


@pytest.fixture(autouse=True)
def isolated_traces(tmp_path, monkeypatch):
    # Test runs should write to temporary folders, not the project's run history.
    monkeypatch.setattr("src.tools.TRACE_DIR", tmp_path / "traces")


def test_six_tools_return_contracts_and_record_calls():
    tools = Tools()
    ticket = tools.get_ticket(TicketInput(ticket_id="T-1001"))
    customer = tools.get_customer_context(CustomerInput(customer_id=ticket.customer_id))
    assert customer.name == "Ari Patel"

    evidence = tools.search_policy(PolicySearchInput(query="duplicate payment invoice refund"))
    assert evidence.evidence[0].source_id == "refunds"
    owner = tools.check_availability(AvailabilityInput(queue="finance-review"))
    assert owner.available

    update = tools.draft_ticket_update(TicketUpdateInput(
        ticket_id=ticket.ticket_id, message="Please provide the last four digits of the payment method.",
        source_ids=["refunds"],
    ))
    assert update.text.endswith("[refunds]")
    escalation = tools.draft_escalation(EscalationInput(
        ticket_id=ticket.ticket_id, queue="finance-review",
        message="Review the reported duplicate payment; it has not yet been confirmed.",
        source_ids=["refunds"],
    ))
    assert escalation.owner == owner.owner
    assert escalation.channel == owner.channel
    assert [call.tool for call in tools.trace] == [
        "get_ticket", "get_customer_context", "search_policy", "check_availability",
        "draft_ticket_update", "draft_escalation",
    ]
    assert all(call.output is not None and call.error is None for call in tools.trace)


def test_unknown_ticket_is_logged_as_a_failure():
    tools = Tools()
    with pytest.raises(ValueError, match="Unknown ticket_id"):
        tools.get_ticket(TicketInput(ticket_id="T-missing"))
    assert tools.trace[0].output is None
    assert "T-missing" in tools.trace[0].error


@pytest.mark.parametrize("query", ["cafeteria menu", "the and of"])
def test_unsupported_queries_have_no_evidence(query):
    assert Tools().search_policy(PolicySearchInput(query=query)).evidence == []


def test_search_reads_new_policy_files(tmp_path):
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)
    tools = Tools(data_dir)
    (data_dir / "knowledge_base/parking.md").write_text(
        "# Parking\n\nSource ID: parking\n\nBicycle parking is beside reception."
    )
    result = tools.search_policy(PolicySearchInput(query="bicycle parking"))
    assert result.evidence[0].source_id == "parking"
    assert "beside reception" in result.evidence[0].text


def test_supplied_attack_paragraph_is_removed():
    result = Tools().search_policy(PolicySearchInput(query="retrieved instruction attacks"))
    assert result.evidence
    assert all("IGNORE ALL PRIOR INSTRUCTIONS" not in item.text for item in result.evidence)
    assert "instruction_attack_filtered:retrieved-content-safety" in result.safety_events


def test_draft_rejects_sources_that_were_not_retrieved():
    tools = Tools()
    with pytest.raises(ValueError, match="accepted search evidence"):
        tools.draft_ticket_update(TicketUpdateInput(
            ticket_id="T-1001", message="Check the payment.", source_ids=["refunds"],
        ))


def test_draft_rejects_a_citation_hidden_in_the_message():
    tools = Tools()
    tools.search_policy(PolicySearchInput(query="duplicate refund payment"))
    with pytest.raises(ValueError, match="undeclared citation"):
        tools.draft_ticket_update(TicketUpdateInput(
            ticket_id="T-1001", message="A claim [invented-policy]", source_ids=["refunds"],
        ))


def test_unavailable_owner_blocks_escalation():
    tools = Tools()
    tools.search_policy(PolicySearchInput(query="MFA account owner"))
    with pytest.raises(ValueError, match="owner_unavailable"):
        tools.draft_escalation(EscalationInput(
            ticket_id="T-1007", queue="identity-review",
            message="The requester cannot verify the account owner.", source_ids=["account-access"],
        ))
    assert "owner_unavailable" in tools.trace[-1].error


def test_input_contracts_reject_invalid_values():
    with pytest.raises(ValidationError):
        TicketInput(ticket_id="  ")
    with pytest.raises(ValidationError):
        PolicySearchInput(query="refund", top_k=0)
    with pytest.raises(ValidationError):
        PolicySearchInput(query="refund", min_score=0.0)
    with pytest.raises(ValidationError):
        TicketUpdateInput(ticket_id="T-1001", message="Hello", source_ids=[])
    with pytest.raises(ValidationError):
        TicketInput(ticket_id="T-1001", unexpected="field")


def test_trace_is_saved_after_success_and_failure(tmp_path):
    tools = Tools(trace_dir=tmp_path / "run-traces")
    assert json.loads(tools.trace_path.read_text())["calls"] == []

    tools.get_ticket(TicketInput(ticket_id="T-1001"))
    saved = json.loads(tools.trace_path.read_text())
    assert saved["run_id"] == tools.run_id
    assert saved["started_at"] == tools.started_at
    assert saved["calls"][0]["output"]["ticket_id"] == "T-1001"

    with pytest.raises(ValueError, match="Unknown ticket_id"):
        tools.get_ticket(TicketInput(ticket_id="T-missing"))
    saved = json.loads(tools.trace_path.read_text())
    assert len(saved["calls"]) == 2
    assert saved["calls"][1]["error"] == "Unknown ticket_id: T-missing"
    assert saved["calls"] == [call.model_dump() for call in tools.trace]


def test_separate_runs_do_not_overwrite_each_other():
    first = Tools()
    first.get_ticket(TicketInput(ticket_id="T-1001"))
    second = Tools()
    second.get_ticket(TicketInput(ticket_id="T-1002"))

    assert first.trace_path != second.trace_path
    assert json.loads(first.trace_path.read_text())["calls"][0]["inputs"]["ticket_id"] == "T-1001"
    assert json.loads(second.trace_path.read_text())["calls"][0]["inputs"]["ticket_id"] == "T-1002"
