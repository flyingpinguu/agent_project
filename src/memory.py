"""Save the full message history for one conversation about one ticket."""

import re
from pathlib import Path
from uuid import uuid4

from pydantic import Field

from src.tools import Contract
from src.validate_data import DATA_DIR


MEMORY_DIR = DATA_DIR.parent / "storage" / "conversations"


class Conversation(Contract):
    conversation_id: str = Field(default_factory=lambda: uuid4().hex)
    ticket_id: str
    messages: list[dict] = Field(default_factory=list)


def conversation_path(conversation_id: str, directory: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", conversation_id):
        raise ValueError("Conversation ID must contain only letters, numbers, underscores or hyphens")
    return directory / f"{conversation_id}.json"


def load_conversation(conversation_id: str, directory: Path) -> Conversation:
    path = conversation_path(conversation_id, directory)
    if not path.exists():
        raise ValueError(f"Conversation not found: {conversation_id}")
    conversation = Conversation.model_validate_json(path.read_text(encoding="utf-8"))
    if conversation.conversation_id != conversation_id:
        raise ValueError("Saved conversation ID does not match its filename")
    return conversation


def save_conversation(conversation: Conversation, directory: Path):
    path = conversation_path(conversation.conversation_id, directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(conversation.model_dump_json(indent=2), encoding="utf-8")
    temporary_path.replace(path)
