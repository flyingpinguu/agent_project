"""Command-line interface for the reference solution."""

import argparse
from pathlib import Path

from solution.agent_project.agent import OperationsAgent
from solution.agent_project.config import Settings
from solution.agent_project.datasets import validate_data
from solution.agent_project.evaluation import evaluate, load_cases, render_report


def main(argv: list[str] | None = None) -> None:
    """Run the requested reference-solution command."""

    parser = argparse.ArgumentParser(
        description="Northstar Agent Project reference CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect-data")
    subparsers.add_parser("index")
    run_case = subparsers.add_parser("run-case")
    run_case.add_argument("case_id")
    chat = subparsers.add_parser("chat")
    chat.add_argument("--case", required=True, dest="case_id")
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.command == "inspect-data":
        summary = validate_data(settings.project_root)
        print(
            f"tickets={summary.ticket_count} customers={summary.customer_count} "
            f"sources={summary.source_count} cases={summary.case_count}"
        )
        return

    agent = OperationsAgent(settings)
    if args.command == "index":
        print(f"indexed_chunks={len(agent.retriever.chunks)}")
        return

    cases = load_cases(settings.evaluation_file)
    print(
        f"provider={settings.llm_provider} model={agent.model.model_name if agent.model else 'scripted'}"
    )
    if args.command in {"run-case", "chat"}:
        case = next((item for item in cases if item["case_id"] == args.case_id), None)
        if case is None:
            parser.error(f"Unknown case: {args.case_id}")
        result = agent.run_case(case)
        print(f"status={result.status}")
        print(f"citations={','.join(result.citations) or '-'}")
        print(result.answer)
        return

    if args.command == "evaluate":
        output = (args.output or settings.report_file).resolve()
        reference = settings.project_root / "solution/reports/reference_evaluation.md"
        if output == reference.resolve():
            parser.error(
                "Choose a local output path; the reference report is versioned."
            )
        summary = evaluate(agent, cases)
        report = render_report(summary)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"wrote={output}")


if __name__ == "__main__":
    main()
