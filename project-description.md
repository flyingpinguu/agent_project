# Project Description

Northstar runs a professional learning platform with support colleagues who
handle account access, refunds, privacy requests, programme incidents, and
technical escalations. The team wants an operations assistant that can inspect
ticket context, retrieve approved policies, remember follow-up context, and
draft the next operational action without inventing policy.

You receive approved policy sources under
[data/knowledge_base/](data/knowledge_base/), operational fixtures under
[data/operations/](data/operations/), and evaluation cases under
[data/evaluation/agent_cases.json](data/evaluation/agent_cases.json). Build a
local-first agent and an evaluation report that gives reviewers enough evidence
to decide whether the assistant should move beyond a prototype.

## Main Task

Build and evaluate an agent that:

- reads ticket and customer context through typed tools,
- retrieves approved policy evidence before giving operational guidance,
- uses conversation memory for follow-up turns,
- respects bounded step and tool-call budgets,
- drafts ticket updates or escalations only when evidence supports them,
- abstains when the available policy does not answer the request,
- reports agent failures transparently.

The separate [reference solution](solution/README.md) demonstrates a deterministic
baseline for the workflow and evaluation contracts. Its generic drafts and
in-process memory are starting points, not a complete assessed submission.

## Supplied Data

### Approved Policies

Every Markdown document in [data/knowledge_base/](data/knowledge_base/) is an
approved source. Preserve stable source IDs through chunking, retrieval, answer
generation, and citation validation.

### Operations Fixtures

[data/operations/](data/operations/) contains tickets, customer context,
on-call availability, and seeded memory. Treat these as local substitutes for
systems such as helpdesk, CRM, status page, and memory stores.

### Evaluation Cases

[data/evaluation/agent_cases.json](data/evaluation/agent_cases.json) labels the
expected status, required tools, expected source IDs, and whether memory should
be used. Some cases also specify forbidden tools and required safety events.
Do not edit labels merely to improve metrics. When you disagree with a
label, document the disagreement in your report.

## Required Implementation

### 1. Data Validation

Validate all data before running the agent:

- every ticket has an ID, customer ID, category, subject, and body,
- every customer referenced by a ticket exists,
- every evaluation case has a unique ID and valid ticket references,
- every expected source ID exists in the policy corpus,
- seeded memory does not leak unrelated customer data into a case.

### 2. Typed Operation Tools

Implement tools with explicit input and output contracts. At minimum include:

- ticket lookup,
- customer context lookup,
- policy search,
- on-call or availability lookup,
- ticket update drafting,
- escalation drafting.

Log each tool call in the trace. The evaluation must be able to inspect which
tools were used and whether they matched the task.

### 3. Retrieval And Grounding

Index the approved policy sources and retrieve evidence for each request.
Answers must cite source IDs in square brackets, such as `[refunds]`, and the
application must reject citations that were not present in accepted evidence.

Define a retrieval threshold or equivalent no-answer rule. Unsupported requests
must abstain or hand off rather than use the nearest weak match.

### 4. Memory

Implement short-term memory for a conversation and durable case memory for
follow-up turns. Memory may help resolve references such as "the same customer"
or "that incident," but it must not override current policy evidence.

### 5. Bounded Agent Control

Set maximum step and tool-call limits. The agent should return a visible
`review_required` or `escalated` outcome when it cannot complete safely inside
those bounds.

### 6. Evaluation Report

Submit `reports/evaluation_report.md` containing:

- configuration and agent-control limits,
- aggregate scores for task completion, tool correctness, retrieval grounding,
  memory use, and failure recovery,
- a per-case results table,
- at least two failure or fragile-success analyses,
- the evidence behind the retrieval threshold,
- three prioritized improvement recommendations.

## Suggested Repository Layout

```text
your-agent-project/
├── data/
│   ├── knowledge_base/
│   ├── operations/
│   └── evaluation/
├── reports/
│   └── evaluation_report.md
├── src/
│   ├── agent.py
│   ├── tools.py
│   ├── retrieval.py
│   ├── memory.py
│   └── evaluation.py
├── tests/
├── .env.example
├── README.md
├── pyproject.toml
└── uv.lock
```

## Suggested Milestones

### Milestone 1 - Data, Tools, And Retrieval

Evidence to show:

- validated local fixtures,
- typed tool contracts and trace logging,
- policy chunks with source metadata,
- retrieval results for supported and unsupported requests.

### Milestone 2 - Agent With Memory

Evidence to show:

- one ticket resolved with valid policy citations,
- one follow-up turn that uses memory correctly,
- one unsupported request that abstains,
- one risky case that escalates or requests human review.

### Milestone 3 - Evaluation And Recommendation

Evidence to show:

- generated evaluation report,
- per-case tool and retrieval traces,
- failure analysis connected to observed traces,
- prioritized recommendations tied to measured limitations.

## Minimum Acceptance Checklist

- [ ] Policy, operations, memory, and evaluation fixtures remain separate.
- [ ] Source metadata survives retrieval and citation output.
- [ ] Tools have explicit contracts and traceable calls.
- [ ] The agent enforces step and tool-call budgets.
- [ ] Supported tickets produce cited, evidence-grounded actions.
- [ ] Unsupported questions abstain or route to review.
- [ ] Escalation cases use availability context.
- [ ] Follow-up cases use memory without leaking unrelated context.
- [ ] The report separates tool, retrieval, memory, and outcome metrics.
- [ ] Tests pass from the documented environment.

## Answer The Following Questions

1. Which tool failures would be most harmful in the operations scenario?
2. How did you select your retrieval threshold?
3. What evidence shows that memory helped rather than polluted the answer?
4. When should this agent hand off to a human reviewer?
5. What would you change before connecting it to real systems?
