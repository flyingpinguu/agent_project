# Agent Project

Use this repository as a **template** for your agent project. Build a
Northstar operations assistant that resolves support tickets by
combining typed tool calls, retrieval over approved policies, and conversation
memory for follow-up requests. Create pull requests in your own copy even if you are working alone, and use them to track your progress.

This assignment follows the earlier lessons on
tool-calling agents, retrieval-augmented generation, and
conversation state. Unlike the lesson repositories, this project asks you to
make and justify system decisions in a complete assessed workflow.

## Learning Objectives

By the end of this repository, you should be able to:

- Design and build a task-oriented agent that integrates tool calling,
  retrieval, and conversation memory for a multi-step operations workflow.
- Evaluate agent behavior using task completion, tool-use correctness,
  retrieval grounding, memory use, and failure-recovery evidence.
- Decide when an agent should answer, abstain, or hand off to a human reviewer.

## Learning Path

```mermaid
flowchart LR
    A["Support ticket"] --> B["Agent planner"]
    C["Conversation memory"] --> B
    D["Typed operation tools"] --> B
    E["Approved policies"] --> F["Retrieve evidence"]
    F --> B
    B --> G["Resolve, abstain, or escalate"]
    G --> H["Evaluation report"]
```

| File / Folder | Description |
|---|---|
| [**Project Description**](project-description.md) | Scenario, milestones, deliverables, and acceptance checklist. |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**data/knowledge_base**](data/knowledge_base/) | Approved support policies the agent may cite. |
| [**data/operations**](data/operations/) | Tickets, customers, availability, and seed memory for the agent's tools. |
| [**data/evaluation**](data/evaluation/agent_cases.json) | Labeled cases for scoring the agent. |
| [**solution**](solution/README.md) | Runnable reference and tests. Inspect after attempting the project. |
| [**.env.example**](.env.example) | Execution mode, provider credentials, and local limits. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.5 or newer
and Python 3.13.17 or a newer 3.13 patch release, as defined in
[`.python-version`](.python-version).

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd mle-agent-project`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

The command below installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

The reference uses local fixtures and lexical retrieval, with Groq as the default PydanticAI provider. Set `LLM_PROVIDER=dry-run` explicitly for offline runs. Sentence Transformers and Qdrant remain available for your own retrieval implementation.

---

### 5. Set Up Environment Variables

Copy the execution mode, provider settings, and limits:

```bash
cp .env.example .env
```

Keep `LLM_PROVIDER=groq` and fill in `GROQ_API_KEY` for your first run. The reference loads `.env` from the repository root; exported environment variables take precedence.

For live model-controlled tool calling, set `LLM_PROVIDER` and its matching key:

- **Groq (primary):** `LLM_PROVIDER=groq`, `GROQ_API_KEY`, optional `GROQ_MODEL`.
- **Mistral (alternative):** `LLM_PROVIDER=mistral`, `MISTRAL_API_KEY`, optional `MISTRAL_MODEL`.
- **OpenRouter (optional):** `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, optional `OPENROUTER_MODEL`. `OPENROUTER_FALLBACK_MODEL=openrouter/free` enables a server-side model backup; leave it blank to disable. Free accounts currently share [50 requests/day across free models](https://openrouter.ai/pricing).
- **Google AI Studio (optional):** `LLM_PROVIDER=google`, `GOOGLE_API_KEY`, `GOOGLE_MODEL=gemma-4-26b-a4b-it` for direct [Gemma access](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api), independent of OpenRouter. Free [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) vary by project and model; use only non-sensitive exercise data on the free tier.
- **OpenAI (optional, paid):** `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, optional `OPENAI_MODEL`.

Only the selected provider is called. Missing keys and API errors never fall back to simulation or another API provider. Set `LLM_PROVIDER=dry-run` explicitly for offline runs; tests isolate this mode from local credentials. All tools still use local fixtures and create drafts, never real support updates.

> [!CAUTION]
> Never commit `.env`. It is git-ignored because environment files often hold secrets. Only `.env.example`, with placeholder or default values, belongs in the repository.

---

### 6. Open the Repository in VS Code

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Select the Python interpreter in `.venv/` created by `uv sync`. Run the [reference solution commands](solution/README.md#run-the-reference-workflow) from the project root.

## Required Workflow

### Our implementation: first step

Validate the local data before building the agent:

```bash
uv run python -m src.validate_data
uv run pytest -q
```

The validator in `src/validate_data.py` checks required fields, unique IDs,
ticket/customer links, policy source IDs, evaluation references, and the
customer/ticket link in seed memory. These checks do not inspect free-text
memory notes for unrelated personal data; memory isolation also needs to be
enforced when we implement memory loading. No API key is needed for this step.

### Our implementation: tools

The six tools and their Pydantic input/output contracts are in `src/tools.py`.
Create a fresh `Tools()` instance per request: it holds the tool trace and the
source IDs accepted during that request. For example:

```python
from src.tools import Tools, TicketInput, PolicySearchInput

tools = Tools()
ticket = tools.get_ticket(TicketInput(ticket_id="T-1001"))
result = tools.search_policy(PolicySearchInput(query="duplicate payment refund"))
print(ticket.subject)
print(result.model_dump())
```

Ticket, customer, and availability lookups match exact IDs or queue names.
Policy search scores paragraphs by the fraction of query keywords they contain,
ignoring common words. Each result carries its source ID, text, and score.
The default threshold of 0.2 is provisional; evaluation must establish a suitable
threshold. New wording can still cause keyword search to miss relevant evidence.
The supplied attack-example paragraph is filtered and recorded as a safety event;
this simple filter is not a general instruction-attack detector.

The two drafting tools accept proposed text and return a draft with citations.
They reject unretrieved source IDs and undeclared citations. Escalation also
checks that the selected owner is available. These tools do not establish whether
every claim follows from the policy or whether the selected queue is appropriate;
agent control and evaluation will address those checks. They never modify fixtures
or send a message. Successful calls and tool errors are recorded in `tools.trace`.
Each `Tools()` instance also creates its own JSON file under `storage/traces/`,
containing a unique run ID, UTC start time, and the calls. The file is updated
after every successful or failed call. `tools.trace_path` gives its location;
use `Tools(trace_dir=some_path)` to choose another directory. Trace files are
local generated artifacts and are ignored by Git.
Invalid input models raise Pydantic validation errors before a tool is called.

Run only our tests with `uv run pytest tests -q`; `uv run pytest -q` also includes
the supplied reference-solution tests.

### Our implementation: one-ticket agent

`src/agent.py` connects our tools to GPT-6 Luna through OpenRouter. The loop is
explicit: ask the model for a tool call, validate its arguments, execute the
tool, and send its result back. A successful draft ends the run immediately.

Set these values in the ignored `.env` file (exported variables take precedence):

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=openai/gpt-6-luna
MAX_AGENT_STEPS=8
MAX_TOOL_CALLS=6
```

Run one ticket:

```bash
uv run python -m src.agent T-1001
```

Optional: add `--prompt "Investigate the duplicate payment and draft a response."`.
The result includes the status, draft text, and JSON trace path. The trace also
contains model messages, usage when returned by the provider, control limits,
and the final result. It never stores the configured API key.

Each model request counts as one step. Every attempted tool call counts toward
the tool budget, including invalid requests. Exhausted budgets and API failures
return `review_required`; no API or model fallback is used. Requests have a
60-second timeout and no automatic retries. The agent restricts access to the
requested ticket and its customer, requires context and evidence before drafting,
and requires an availability check before escalation. Chat Completions uses
`reasoning_effort="none"` for this model's function calls.

This is a prototype: semantic policy compliance still depends on the model and
needs evaluation.

### Continuing the same case

Every completed turn saves the full conversation under
`storage/conversations/<conversation_id>.json`. This includes the system prompt,
user messages, assistant messages, tool requests, tool results, and the final
reply shown to the user. The result printed by the command includes the
`conversation_id` to use for a follow-up:

```bash
uv run python -m src.agent --conversation YOUR_CONVERSATION_ID --prompt "Make that reply shorter."
```

The agent loads all previous messages from JSON, appends the new request, and
sends the full history. This also works after restarting the program. We do not
configure caching or summarize/truncate history. The history will therefore grow
with each turn; a provider context-limit error returns `review_required`.

Each conversation belongs to one ticket. Start a new conversation by providing
a ticket ID without `--conversation`. A follow-up can omit the ticket ID because
it is stored with the history. Previous tool results stay in the history for
context, but each turn starts with fresh tool state and must read the current
ticket/customer and search policies again before drafting. Execution budgets
reset each turn. Each turn has its own trace file with the shared conversation ID.
`src/memory.py` contains the small JSON loading and saving functions. Conversation
files are Git-ignored, like traces. Continue a conversation one turn at a time;
simultaneous writers to the same conversation are not supported.

Work through the project in this order:

1. Validate the supplied ticket, policy, memory, and evaluation data.
2. Build typed tools for ticket lookup, customer context, policy retrieval,
   availability checks, ticket updates, and escalation drafts.
3. Index approved policy sources and preserve source IDs through retrieval and
   answer validation.
4. Implement short-term conversation memory plus durable case memory for
   follow-up turns.
5. Add bounded agent execution with tool-call limits, evidence thresholds,
   unsupported-question abstention, and human-review handoff.
6. Evaluate task completion, tool correctness, retrieval grounding, memory use,
   and failure recovery in a written report.

## References & Further Reading

- [PydanticAI Agents](https://pydantic.dev/docs/ai/agents/)
- [PydanticAI Function Tools](https://pydantic.dev/docs/ai/tools/)
- [PydanticAI Message History](https://pydantic.dev/docs/ai/core-concepts/message-history/)
