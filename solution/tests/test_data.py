"""Tests for project fixture contracts."""

import json
import shutil
from pathlib import Path

import pytest

from solution.agent_project.agent import OperationsAgent
from solution.agent_project.config import Settings
from solution.agent_project.datasets import validate_data


def test_data_contracts_are_valid(project_root: Path) -> None:
    summary = validate_data(project_root)

    assert summary.ticket_count == 7
    assert summary.customer_count == 7
    assert summary.source_count == 6
    assert summary.case_count == 7


@pytest.mark.parametrize("filename", ["tickets.json", "customers.json"])
def test_duplicate_ids_rejected_before_agent_starts(project_root, tmp_path, filename):
    shutil.copytree(project_root / "data", tmp_path / "data")
    path = tmp_path / "data" / "operations" / filename
    rows = json.loads(path.read_text())
    rows.append(rows[0])
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="must be unique"):
        OperationsAgent(Settings.from_env(tmp_path))


def test_seed_memory_must_match_ticket_customer(project_root, tmp_path):
    shutil.copytree(project_root / "data", tmp_path / "data")
    path = tmp_path / "data" / "operations" / "seed_memory.json"
    memory = json.loads(path.read_text())
    memory["last_customer_id"] = "C-001"
    path.write_text(json.dumps(memory))
    with pytest.raises(ValueError, match="does not match"):
        validate_data(tmp_path)


def test_verification_flag_is_not_a_truthy_string(project_root, tmp_path):
    shutil.copytree(project_root / "data", tmp_path / "data")
    path = tmp_path / "data" / "operations" / "customers.json"
    customers = json.loads(path.read_text())
    customers[0]["verified"] = "false"
    path.write_text(json.dumps(customers))
    with pytest.raises(ValueError, match="boolean"):
        validate_data(tmp_path)
