# Checklist for the Agent Project

<!-- If you are done with a topic mark the checkboxes with an `x` (like `[x]`) -->

- [ ] I read and understood the tasks.
- [ ] I validated the policy, operations, memory, and evaluation data and kept them separate.
- [ ] I implemented typed tools with explicit contracts and logged every tool call in the trace.
- [ ] I indexed the policy sources and preserved source IDs through retrieval and citations.
- [ ] I defined a retrieval threshold so unsupported requests abstain or route to review.
- [ ] Generated citations are validated against the accepted evidence.
- [ ] I implemented conversation and case memory without leaking unrelated customer context.
- [ ] The agent enforces step and tool-call budgets and returns `review_required` or `escalated` when it cannot finish safely.
- [ ] Escalation cases use availability context.
- [ ] I wrote `reports/evaluation_report.md` with separate tool, retrieval, memory, and outcome metrics, failure analysis, and prioritized recommendations.
- [ ] I added tests and they pass from the documented environment.
- [ ] I documented how to run the project and answered the submission questions.

Bonus:

- [ ] I ran the agent with a provider-backed LLM in addition to the offline dry-run mode.
- [ ] I documented any disagreements with the evaluation labels in the report.
