import json
import shutil

import pytest

from src.validate_data import DATA_DIR, validate_data


def test_supplied_data_passes():
    assert validate_data() == {
        "tickets": 7, "customers": 7, "policies": 6, "cases": 7,
    }


@pytest.mark.parametrize(
    "filename, field, value, message",
    [
        ("operations/tickets.json", "body", "", "Missing or invalid body"),
        ("operations/tickets.json", "customer_id", "C-999", "Unknown customer"),
        ("evaluation/agent_cases.json", "ticket_id", "T-999", "Unknown ticket"),
        ("evaluation/agent_cases.json", "expected_sources", ["unknown"], "Unknown source"),
        ("evaluation/agent_cases.json", "case_id", "mfa_followup_memory", "Duplicate case_id"),
    ],
)
def test_broken_references_are_rejected(tmp_path, filename, field, value, message):
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)
    path = data_dir / filename
    records = json.loads(path.read_text())
    records[0][field] = value
    path.write_text(json.dumps(records))

    with pytest.raises(ValueError, match=message):
        validate_data(data_dir)


def test_memory_cannot_reference_a_different_customer(tmp_path):
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)
    path = data_dir / "operations/seed_memory.json"
    memory = json.loads(path.read_text())
    memory["last_customer_id"] = "C-001"
    path.write_text(json.dumps(memory))

    with pytest.raises(ValueError, match="Seed memory customer does not match"):
        validate_data(data_dir)
