"""Six local tools. Create a fresh Tools instance for each request.

Pydantic models describe and validate what each tool accepts and returns.
Drafting stores text in the result only; it does not change the fixtures.
"""

import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.validate_data import DATA_DIR, load_json, validate_data


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TRACE_DIR = DATA_DIR.parent / "storage" / "traces"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class TicketInput(Contract):
    ticket_id: Text


class Ticket(Contract):
    ticket_id: Text
    customer_id: Text
    category: Text
    subject: Text
    body: Text


class CustomerInput(Contract):
    customer_id: Text


class Customer(Contract):
    customer_id: Text
    name: Text
    email: Text
    plan: Text
    verified: bool


class PolicySearchInput(Contract):
    query: Text
    top_k: int = Field(default=3, ge=1, le=10)
    min_score: float = Field(default=0.2, gt=0, le=1)


class PolicyEvidence(Contract):
    source_id: Text
    text: Text
    score: float = Field(ge=0, le=1)


class PolicySearchOutput(Contract):
    evidence: list[PolicyEvidence]
    safety_events: list[str]


class AvailabilityInput(Contract):
    queue: Text


class Availability(Contract):
    queue: Text
    owner: Text
    channel: Text
    available: bool


class TicketUpdateInput(Contract):
    ticket_id: Text
    message: Text
    source_ids: list[Text] = Field(min_length=1)


class EscalationInput(TicketUpdateInput):
    queue: Text


class Draft(Contract):
    kind: Literal["ticket_update", "escalation"]
    ticket_id: Text
    text: Text
    source_ids: list[Text]
    queue: str | None = None
    owner: str | None = None
    channel: str | None = None


class ToolCall(Contract):
    tool: str
    inputs: dict
    output: dict | None = None
    error: str | None = None


# Ignore common words so that "the" alone cannot count as policy evidence.
STOP_WORDS = set("a an and are as at be by can for from how i in is it of on or "
                 "please should that the this to we what when with you your".split())


def keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


class Tools:
    def __init__(self, data_dir: Path = DATA_DIR, trace_dir: Path | None = None):
        validate_data(data_dir)
        self.data_dir = data_dir
        self.trace: list[ToolCall] = []
        self.accepted_sources: set[str] = set()
        self.agent_run: dict | None = None
        self.run_id = uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.trace_path = (trace_dir if trace_dir is not None else TRACE_DIR) / f"{self.run_id}.json"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_trace()

    def _save_trace(self):
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "calls": [call.model_dump() for call in self.trace],
        }
        if self.agent_run is not None:
            payload["agent"] = self.agent_run
        # Replace only after writing finishes, keeping the previous JSON intact.
        temporary_path = self.trace_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary_path.replace(self.trace_path)

    @contextmanager
    def _call(self, name: str, request: Contract):
        """Record both successful calls and failures in one place."""
        call = ToolCall(tool=name, inputs=request.model_dump())
        self.trace.append(call)
        try:
            yield call
        except Exception as error:
            call.error = str(error)
            raise
        finally:
            self._save_trace()

    def _lookup(self, filename: str, id_field: str, value: str) -> dict:
        records = load_json(self.data_dir / "operations" / filename)
        for record in records:
            if record[id_field] == value:
                return record
        raise ValueError(f"Unknown {id_field}: {value}")

    def get_ticket(self, request: TicketInput) -> Ticket:
        with self._call("get_ticket", request) as call:
            result = Ticket(**self._lookup("tickets.json", "ticket_id", request.ticket_id))
            call.output = result.model_dump()
            return result

    def get_customer_context(self, request: CustomerInput) -> Customer:
        with self._call("get_customer_context", request) as call:
            result = Customer(**self._lookup("customers.json", "customer_id", request.customer_id))
            call.output = result.model_dump()
            return result

    def search_policy(self, request: PolicySearchInput) -> PolicySearchOutput:
        with self._call("search_policy", request) as call:
            query_words = keywords(request.query)
            matches = []
            safety_events = []
            # Read files on each search so edits are picked up automatically.
            for path in sorted((self.data_dir / "knowledge_base").glob("*.md")):
                content = path.read_text(encoding="utf-8")
                source_id = next(line.split(":", 1)[1].strip()
                                 for line in content.splitlines() if line.startswith("Source ID:"))
                # Each paragraph is a small chunk that keeps its policy source ID.
                for paragraph in content.split("\n\n"):
                    if paragraph.startswith(("#", "Source ID:")):
                        continue
                    # A basic filter for the supplied attack example, not a general detector.
                    if "ignore all prior instructions" in paragraph.lower():
                        safety_events.append(f"instruction_attack_filtered:{source_id}")
                        continue
                    score = len(query_words & keywords(paragraph)) / max(len(query_words), 1)
                    if score >= request.min_score:
                        matches.append(PolicyEvidence(source_id=source_id, text=paragraph, score=score))
            matches.sort(key=lambda match: match.score, reverse=True)
            result = PolicySearchOutput(evidence=matches[:request.top_k], safety_events=safety_events)
            self.accepted_sources.update(item.source_id for item in result.evidence)
            call.output = result.model_dump()
            return result

    def check_availability(self, request: AvailabilityInput) -> Availability:
        with self._call("check_availability", request) as call:
            result = Availability(**self._lookup("availability.json", "queue", request.queue))
            call.output = result.model_dump()
            return result

    def _draft_text(self, request: TicketUpdateInput) -> str:
        self._lookup("tickets.json", "ticket_id", request.ticket_id)
        if not set(request.source_ids) <= self.accepted_sources:
            raise ValueError("Draft sources must come from accepted search evidence")
        citations = set(re.findall(r"\[([^\[\]\n]+)\]", request.message))
        if not citations <= set(request.source_ids):
            raise ValueError("Message contains an undeclared citation")
        missing = [source for source in request.source_ids if source not in citations]
        return request.message + "".join(f" [{source}]" for source in missing)

    def draft_ticket_update(self, request: TicketUpdateInput) -> Draft:
        with self._call("draft_ticket_update", request) as call:
            result = Draft(kind="ticket_update", ticket_id=request.ticket_id,
                           text=self._draft_text(request), source_ids=request.source_ids)
            call.output = result.model_dump()
            return result

    def draft_escalation(self, request: EscalationInput) -> Draft:
        with self._call("draft_escalation", request) as call:
            owner = Availability(**self._lookup("availability.json", "queue", request.queue))
            if not owner.available:
                raise ValueError(f"owner_unavailable: {request.queue}")
            message = self._draft_text(request)
            result = Draft(kind="escalation", ticket_id=request.ticket_id,
                           text=f"Ticket {request.ticket_id} → {owner.queue} ({owner.owner})\n{message}",
                           source_ids=request.source_ids, queue=owner.queue,
                           owner=owner.owner, channel=owner.channel)
            call.output = result.model_dump()
            return result
