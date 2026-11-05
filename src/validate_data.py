"""Check the local data before giving it to the agent.

Run from the project root: uv run python -m src.validate_data
"""

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def index_records(records: list[dict], required_fields: list[str]) -> dict:
    """Check required fields and build a lookup using the first field as ID."""
    index = {}
    id_field = required_fields[0]
    for record in records:
        for field in required_fields:
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Missing or invalid {field}: {record}")
        record_id = record[id_field]
        if record_id in index:
            raise ValueError(f"Duplicate {id_field}: {record_id}")
        index[record_id] = record
    return index


def validate_data(data_dir: Path = DATA_DIR) -> dict[str, int]:
    customers = index_records(
        load_json(data_dir / "operations/customers.json"),
        ["customer_id", "name", "email", "plan"],
    )
    tickets = index_records(
        load_json(data_dir / "operations/tickets.json"),
        ["ticket_id", "customer_id", "category", "subject", "body"],
    )
    for ticket in tickets.values():
        if ticket["customer_id"] not in customers:
            raise ValueError(f"Unknown customer in ticket {ticket['ticket_id']}")

    # Read the explicit source ID rather than assuming it matches the filename.
    sources = {}
    for path in sorted((data_dir / "knowledge_base").glob("*.md")):
        content = path.read_text(encoding="utf-8")
        ids = [line.removeprefix("Source ID:").strip()
               for line in content.splitlines() if line.startswith("Source ID:")]
        if len(ids) != 1 or not ids[0]:
            raise ValueError(f"Expected one source ID in {path.name}")
        if ids[0] in sources:
            raise ValueError(f"Duplicate source ID: {ids[0]}")
        sources[ids[0]] = content
    if not sources:
        raise ValueError("No approved policies found")

    memory = load_json(data_dir / "operations/seed_memory.json")
    memory_ticket = tickets.get(memory.get("last_ticket_id"))
    if memory_ticket is None:
        raise ValueError("Seed memory references an unknown ticket")
    if memory.get("last_customer_id") != memory_ticket["customer_id"]:
        raise ValueError("Seed memory customer does not match its ticket")

    cases = index_records(
        load_json(data_dir / "evaluation/agent_cases.json"),
        ["case_id", "prompt", "expected_status"],
    )
    for case in cases.values():
        ticket_id = case.get("ticket_id")
        if ticket_id is not None and ticket_id not in tickets:
            raise ValueError(f"Unknown ticket in case {case['case_id']}")
        if ticket_id is None and case.get("must_use_memory") is not True:
            raise ValueError(f"Case {case['case_id']} has no ticket or memory")
        if case.get("must_use_memory") is True and ticket_id is not None:
            if tickets[ticket_id]["customer_id"] != memory["last_customer_id"]:
                raise ValueError(f"Unrelated seed memory in case {case['case_id']}")
        for source_id in case["expected_sources"]:
            if source_id not in sources:
                raise ValueError(f"Unknown source {source_id} in case {case['case_id']}")

    index_records(
        load_json(data_dir / "operations/availability.json"),
        ["queue", "owner", "channel"],
    )
    return {"tickets": len(tickets), "customers": len(customers),
            "policies": len(sources), "cases": len(cases)}


if __name__ == "__main__":
    counts = validate_data()
    print("Data checks passed:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
