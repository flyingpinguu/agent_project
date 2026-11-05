"""Operations agent with deterministic and provider-backed execution paths."""

from collections.abc import Iterable

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model

from solution.agent_project.config import Settings
from solution.agent_project.datasets import validate_data
from solution.agent_project.documents import chunk_documents, load_documents
from solution.agent_project.embeddings import cosine, vectorize
from solution.agent_project.live import run_live_case
from solution.agent_project.memory import ConversationMemory
from solution.agent_project.models import AgentResult, SearchHit
from solution.agent_project.provider import model_from_env
from solution.agent_project.retrieval import PolicyRetriever
from solution.agent_project.tools import (
    EscalationRejected,
    ToolBudgetExceeded,
    ToolRegistry,
)

ESCALATION_QUEUES = {
    "billing": "finance-review",
    "incident": "incident-response",
    "privacy": "privacy",
    "account": "identity-review",
}


class OperationsAgent:
    """Small agent loop with explicit tools, retrieval, and memory."""

    def __init__(self, settings: Settings, *, model: Model | None = None) -> None:
        self.settings = settings
        self.model = (
            model if model is not None else model_from_env(settings.llm_provider)
        )
        self.message_history: list[ModelMessage] = []
        validate_data(settings.project_root)
        chunks = chunk_documents(load_documents(settings.knowledge_base_dir))
        self.retriever = PolicyRetriever(chunks, settings.top_k, settings.min_score)
        # The scripted path counts each tool as a step; live model requests are separate.
        self.tools = ToolRegistry(
            settings.operations_dir,
            self.retriever,
            settings.max_tool_calls
            if self.model is not None
            else min(settings.max_agent_steps, settings.max_tool_calls),
        )
        self.memory = ConversationMemory.load(settings.operations_dir)

    def run_case(self, case: dict[str, object]) -> AgentResult:
        """Run one labeled evaluation case through the reference agent."""

        self.tools.reset_trace()
        try:
            if self.model is not None:
                return run_live_case(self, case, self.model)
            return self._run_case(case)
        except (ToolBudgetExceeded, UsageLimitExceeded):
            return self._result(
                case,
                "review_required",
                "Execution budget exhausted; request human review.",
                [],
                case.get("ticket_id") is None,
                ["execution_budget_exhausted"],
            )
        except EscalationRejected as error:
            return self._result(
                case,
                "review_required",
                str(error),
                [],
                case.get("ticket_id") is None,
                ["escalation_rejected"],
            )
        except KeyError:
            return self._result(
                case,
                "review_required",
                "Ticket or customer context is unavailable; request human review.",
                [],
                case.get("ticket_id") is None,
                ["missing_context"],
            )

    def _run_case(self, case: dict[str, object]) -> AgentResult:
        prompt = str(case["prompt"])
        memory_used = False
        ticket_id = case.get("ticket_id")
        if ticket_id is None:
            ticket_id = self.memory.last_ticket_id
            memory_used = True
        if ticket_id is None:
            return self._result(
                case, "review_required", "No ticket is available.", [], memory_used
            )

        ticket = self.tools.get_ticket(str(ticket_id))
        if memory_used and ticket["customer_id"] != self.memory.last_customer_id:
            return self._result(
                case,
                "review_required",
                "Conversation memory does not match the ticket customer.",
                [],
                True,
                ["memory_customer_mismatch"],
            )
        customer = self.tools.get_customer_context(ticket["customer_id"])
        self.memory.remember_ticket(ticket["ticket_id"], ticket["customer_id"])
        ticket_context = f"{ticket['category']} {ticket['subject']} {ticket['body']}"
        query = f"{prompt} {ticket_context}"
        hits, safety_events = self.tools.search_policy(query)
        # Ticket context must not make an unrelated request appear supported.
        request_vector = vectorize(prompt)
        if not any(
            cosine(request_vector, vectorize(text)) >= self.settings.min_score
            for text in [ticket_context, *(hit.chunk.text for hit in hits)]
        ):
            hits = []
            safety_events.append("request_not_supported")
        citations = self._ordered_citations(hits)

        if not citations:
            answer = (
                f"No approved Northstar policy source supports this request for "
                f"{ticket['ticket_id']}; route it to human review."
            )
            return self._result(
                case, "abstained", answer, [], memory_used, safety_events
            )

        queue = None
        if ticket["category"] in {"incident", "privacy"}:
            escalation_hits, escalation_events = self.tools.search_policy(
                f"escalation rules {ticket['category']} {ticket['subject']}"
            )
            safety_events.extend(escalation_events)
            citations = self._ordered_citations([*hits, *escalation_hits])
            queue = ESCALATION_QUEUES[ticket["category"]]
        elif ticket["category"] == "account" and not customer["verified"]:
            queue = "identity-review"

        if queue is not None:
            availability = self.tools.check_availability(queue)
            if not availability["available"]:
                return self._result(
                    case,
                    "review_required",
                    f"No available owner for {queue}; request human review.",
                    citations,
                    memory_used,
                    [*safety_events, "owner_unavailable"],
                )
            answer = self.tools.draft_escalation(
                ticket["ticket_id"], queue, str(availability["owner"]), citations
            )
            status = "escalated"
        else:
            answer = self.tools.draft_ticket_update(
                ticket["ticket_id"], customer["name"], citations
            )
            status = "answered"

        invalid = self._invalid_citations(answer, citations)
        if invalid:
            status = "invalid_citations"
            safety_events = [*safety_events, f"invalid_citations:{','.join(invalid)}"]

        return self._result(case, status, answer, citations, memory_used, safety_events)

    def _result(
        self,
        case: dict[str, object],
        status: str,
        answer: str,
        citations: list[str],
        memory_used: bool,
        safety_events: list[str] | None = None,
    ) -> AgentResult:
        """Build a result with the current tool trace."""

        return AgentResult(
            case_id=str(case["case_id"]),
            status=status,
            answer=answer,
            citations=citations,
            tools_used=[call.name for call in self.tools.calls],
            memory_used=memory_used,
            safety_events=safety_events or [],
            tool_calls=list(self.tools.calls),
        )

    @staticmethod
    def _ordered_citations(hits: Iterable[SearchHit]) -> list[str]:
        citations: list[str] = []
        for hit in hits:
            source_id = hit.chunk.source_id
            if source_id not in citations:
                citations.append(source_id)
        return citations

    @staticmethod
    def _invalid_citations(answer: str, accepted: list[str]) -> list[str]:
        """Return citation IDs that were generated without retrieved evidence."""

        found = [
            token.split("]", 1)[0] for token in answer.split("[")[1:] if "]" in token
        ]
        return [citation for citation in found if citation not in accepted]
