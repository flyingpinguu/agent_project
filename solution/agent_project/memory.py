"""Conversation memory helpers for follow-up cases."""

from dataclasses import dataclass
from pathlib import Path

from solution.agent_project.datasets import load_json


@dataclass
class ConversationMemory:
    """Seeded memory with updates for the lifetime of one agent instance."""

    conversation_id: str
    last_ticket_id: str | None
    last_customer_id: str | None
    notes: list[str]

    @classmethod
    def load(cls, operations_dir: Path) -> "ConversationMemory":
        """Load seeded memory from the local fixture."""

        payload = load_json(operations_dir / "seed_memory.json")
        return cls(
            conversation_id=payload["conversation_id"],
            last_ticket_id=payload.get("last_ticket_id"),
            last_customer_id=payload.get("last_customer_id"),
            notes=list(payload.get("notes", [])),
        )

    def remember_ticket(self, ticket_id: str, customer_id: str) -> None:
        """Record the latest inspected ticket for follow-up turns."""

        if self.last_customer_id != customer_id:
            self.notes.clear()
        self.last_ticket_id = ticket_id
        self.last_customer_id = customer_id
