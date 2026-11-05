"""Tests for benchmark coverage and report sections."""

from dataclasses import replace

import pytest

from solution.agent_project.cli import main
from solution.agent_project.evaluation import evaluate, load_cases, render_report


def test_evaluation_scores_every_case(agent, settings) -> None:
    cases = load_cases(settings.evaluation_file)
    summary = evaluate(agent, cases)

    assert len(summary.rows) == len(cases)
    assert summary.metric("tools_ok") == 1.0


def test_report_contains_required_sections(agent, settings) -> None:
    summary = evaluate(agent, load_cases(settings.evaluation_file))
    report = render_report(summary)

    assert "## Aggregate Results" in report
    assert "## Per-Case Results" in report
    assert "## Failure And Fragility Analysis" in report
    assert "## Threshold Decision" in report
    assert "## Prioritized Recommendations" in report
    assert "Required tool coverage" in report
    assert "Exact source-ID match" in report
    assert f"all {len(summary.rows)} evaluated cases" in report
    assert "not a threshold calibrated" in report


def test_cli_keeps_reference_report_unchanged(settings, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "solution.agent_project.cli.Settings.from_env", lambda: settings
    )
    reference = settings.project_root / "solution/reports/reference_evaluation.md"
    original = reference.read_bytes()
    output = tmp_path / "evaluation.md"
    main(["evaluate", "--output", str(output)])
    assert output.read_text().startswith("# Reference Evaluation Report")
    assert reference.read_bytes() == original
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--output", str(reference)])
    assert error.value.code == 2
    assert reference.read_bytes() == original


def test_default_report_is_a_local_output(settings) -> None:
    assert settings.report_file == settings.project_root / "reports/local_evaluation.md"


@pytest.mark.parametrize(
    "fault", ["draft_ticket_update", "draft_escalation", "missing_review_reason"]
)
def test_review_case_rejects_unsafe_trace(agent, settings, monkeypatch, fault):
    case = load_cases(settings.evaluation_file)[-1]
    result = agent.run_case(case)
    if fault == "missing_review_reason":
        result = replace(result, safety_events=[])
    else:
        result = replace(result, tools_used=[*result.tools_used, fault])
    monkeypatch.setattr(agent, "run_case", lambda _: result)
    row = evaluate(agent, [case]).rows[0]
    assert row.status_ok and row.tools_ok and row.sources_ok and row.memory_ok
    assert not row.failure_ok
