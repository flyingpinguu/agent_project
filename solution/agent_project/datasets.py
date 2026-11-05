"""Data loading and validation for local project fixtures."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solution.agent_project.documents import source_ids


@dataclass(frozen=True)
class DataSummary:
    """Counts emitted by the data inspection command."""

    ticket_count: int
    customer_count: int
    availability_count: int
    case_count: int
    source_count: int


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON from a fixture file."""

    return json.loads(path.read_text(encoding="utf-8"))


def validate_data(project_root: Path) -> DataSummary:
    """Validate fixture relationships and evaluation labels."""

    operations_dir = project_root / "data" / "operations"
    tickets = load_json(operations_dir / "tickets.json")
    customers = load_json(operations_dir / "customers.json")
    availability = load_json(operations_dir / "availability.json")
    memory = load_json(operations_dir / "seed_memory.json")
    cases = load_json(project_root / "data" / "evaluation" / "agent_cases.json")
    sources = source_ids(project_root / "data" / "knowledge_base")

    ticket_ids = {ticket.get("ticket_id") for ticket in tickets}
    customer_ids = {customer.get("customer_id") for customer in customers}
    case_ids = [case["case_id"] for case in cases]

    if len(ticket_ids) != len(tickets) or len(customer_ids) != len(customers):
        raise ValueError("Ticket and customer IDs must be unique.")
    for customer in customers:
        if not {"customer_id", "name", "verified"} <= set(customer):
            raise ValueError("Customer is missing required keys.")
        if type(customer["verified"]) is not bool:
            raise ValueError("Customer verified must be a boolean.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique.")
    for ticket in tickets:
        required = {"ticket_id", "customer_id", "category", "subject", "body"}
        if not required <= set(ticket):
            raise ValueError(f"Ticket is missing required keys: {ticket}")
        if ticket["customer_id"] not in customer_ids:
            raise ValueError(
                f"Ticket references unknown customer: {ticket['ticket_id']}"
            )
    for case in cases:
        if case["ticket_id"] is not None and case["ticket_id"] not in ticket_ids:
            raise ValueError(f"Case references unknown ticket: {case['case_id']}")
        missing_sources = set(case["expected_sources"]) - sources
        if missing_sources:
            raise ValueError(
                f"Case references unknown sources: {sorted(missing_sources)}"
            )
    if (
        memory["last_ticket_id"] not in ticket_ids
        or memory["last_customer_id"] not in customer_ids
    ):
        raise ValueError("Seed memory references unknown ticket or customer.")
    remembered_ticket = next(
        ticket for ticket in tickets if ticket["ticket_id"] == memory["last_ticket_id"]
    )
    if remembered_ticket["customer_id"] != memory["last_customer_id"]:
        raise ValueError("Seed memory customer does not match its ticket.")

    return DataSummary(
        ticket_count=len(tickets),
        customer_count=len(customers),
        availability_count=len(availability),
        case_count=len(cases),
        source_count=len(sources),
    )
