"""Shared data models for the reference agent."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """Approved policy source loaded from Markdown."""

    source_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """Small retrieval unit with stable citation metadata."""

    chunk_id: str
    source_id: str
    title: str
    position: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    """Retrieved evidence accepted by the policy search step."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ToolCall:
    """Tool trace entry used by evaluation."""

    name: str
    args: dict[str, str]


@dataclass
class AgentResult:
    """Final agent response and trace evidence."""

    case_id: str
    status: str
    answer: str
    citations: list[str]
    tools_used: list[str]
    memory_used: bool
    safety_events: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
