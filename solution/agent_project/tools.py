"""Typed local operation tools used by the deterministic agent."""

from pathlib import Path

from solution.agent_project.datasets import load_json
from solution.agent_project.models import SearchHit, ToolCall
from solution.agent_project.retrieval import PolicyRetriever


class ToolBudgetExceeded(RuntimeError):
    """The next tool call would exceed the configured execution budget."""


class EscalationRejected(RuntimeError):
    """A proposed escalation has no matching available recipient."""


class ToolRegistry:
    """Local substitutes for ticket, CRM, policy, and escalation systems."""

    def __init__(
        self, operations_dir: Path, retriever: PolicyRetriever, call_limit: int = 6
    ) -> None:
        self.operations_dir = operations_dir
        self.retriever = retriever
        self.tickets = {
            item["ticket_id"]: item
            for item in load_json(operations_dir / "tickets.json")
        }
        self.customers = {
            item["customer_id"]: item
            for item in load_json(operations_dir / "customers.json")
        }
        self.availability = load_json(operations_dir / "availability.json")
        self.calls: list[ToolCall] = []
        self.call_limit = call_limit

    def reset_trace(self) -> None:
        """Clear trace entries for a new agent run."""

        self.calls.clear()

    def _record(self, name: str, **kwargs: str) -> None:
        if len(self.calls) >= self.call_limit:
            raise ToolBudgetExceeded("Execution budget exhausted.")
        self.calls.append(ToolCall(name=name, args=kwargs))

    def get_ticket(self, ticket_id: str) -> dict[str, str]:
        """Fetch a support ticket by stable ID."""

        self._record("get_ticket", ticket_id=ticket_id)
        return self.tickets[ticket_id]

    def get_customer_context(self, customer_id: str) -> dict[str, str]:
        """Fetch local customer context for a ticket."""

        self._record("get_customer_context", customer_id=customer_id)
        return self.customers[customer_id]

    def search_policy(self, query: str) -> tuple[list[SearchHit], list[str]]:
        """Retrieve approved policy evidence for a request."""

        self._record("search_policy", query=query)
        return self.retriever.search(query)

    def check_availability(self, queue: str) -> dict[str, str | bool]:
        """Find the responsible queue owner for a handoff."""

        self._record("check_availability", queue=queue)
        for item in self.availability:
            if item["queue"] == queue:
                return item
        return {
            "queue": queue,
            "owner": "unassigned",
            "channel": "#support-review",
            "available": False,
        }

    def draft_ticket_update(
        self,
        ticket_id: str,
        customer_name: str,
        citations: list[str],
    ) -> str:
        """Draft a customer-visible update without writing to a real helpdesk."""

        self._record("draft_ticket_update", ticket_id=ticket_id)
        citation_text = ", ".join(f"[{citation}]" for citation in citations)
        return (
            f"Draft update for {customer_name} on {ticket_id}: we can proceed "
            f"with the documented next step based on {citation_text}."
        )

    def draft_escalation(
        self,
        ticket_id: str,
        queue: str,
        owner: str,
        citations: list[str],
    ) -> str:
        """Draft an internal escalation for a human queue."""

        self._record("draft_escalation", ticket_id=ticket_id, queue=queue, owner=owner)
        availability = next(
            (item for item in self.availability if item["queue"] == queue), None
        )
        if not availability or not availability["available"]:
            raise EscalationRejected(f"No available owner for {queue}; request review.")
        if owner != availability["owner"]:
            raise EscalationRejected(
                f"The proposed owner does not match {queue}; request review."
            )
        citation_text = ", ".join(f"[{citation}]" for citation in citations)
        return (
            f"Escalate {ticket_id} to {queue} ({owner}) with evidence {citation_text}."
        )
