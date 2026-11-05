"""Markdown loading and paragraph-aware chunking."""

import re
from pathlib import Path

from solution.agent_project.models import Chunk, Document

SOURCE_ID_RE = re.compile(r"^Source ID:\s*(?P<source_id>[-a-z0-9_]+)\s*$", re.MULTILINE)


def load_documents(data_dir: Path) -> list[Document]:
    """Load approved policy documents from Markdown files."""

    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next(
            (
                line.removeprefix("#").strip()
                for line in text.splitlines()
                if line.startswith("# ")
            ),
            path.stem,
        )
        match = SOURCE_ID_RE.search(text)
        source_id = match.group("source_id") if match else path.stem
        documents.append(Document(source_id=source_id, title=title, text=text))
    return documents


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Split policy files into citable paragraphs while preserving source IDs."""

    chunks: list[Chunk] = []
    for document in documents:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", document.text)
            if paragraph.strip() and not paragraph.startswith("# ")
        ]
        for position, paragraph in enumerate(paragraphs):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.source_id}-{position}",
                    source_id=document.source_id,
                    title=document.title,
                    position=position,
                    text=paragraph,
                )
            )
    return chunks


def source_ids(data_dir: Path) -> set[str]:
    """Return the set of approved source IDs."""

    return {document.source_id for document in load_documents(data_dir)}
