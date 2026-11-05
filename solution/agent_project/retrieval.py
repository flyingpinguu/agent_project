"""Policy retrieval with citation metadata and safety filtering."""

from solution.agent_project.embeddings import cosine, vectorize
from solution.agent_project.models import Chunk, SearchHit

UNSAFE_MARKERS = (
    "ignore all prior instructions",
    "reveal the system prompt",
    "api keys",
    "hidden tool outputs",
)


def unsafe_instruction(text: str) -> bool:
    """Detect retrieved instruction-attack examples."""

    lowered = text.lower()
    return any(marker in lowered for marker in UNSAFE_MARKERS)


class PolicyRetriever:
    """Deterministic retriever used by tools and evaluation."""

    def __init__(self, chunks: list[Chunk], top_k: int, min_score: float) -> None:
        self.chunks = chunks
        self.top_k = top_k
        self.min_score = min_score
        self._vectors = {
            chunk.chunk_id: vectorize(f"{chunk.title} {chunk.text}") for chunk in chunks
        }

    def search(self, query: str) -> tuple[list[SearchHit], list[str]]:
        """Return accepted evidence and safety events for a query."""

        query_vector = vectorize(query)
        scored = [
            SearchHit(
                chunk=chunk, score=cosine(query_vector, self._vectors[chunk.chunk_id])
            )
            for chunk in self.chunks
        ]
        ranked = sorted(scored, key=lambda hit: hit.score, reverse=True)
        hits: list[SearchHit] = []
        events: list[str] = []
        for hit in ranked:
            if len(hits) >= self.top_k:
                break
            if hit.score < self.min_score:
                continue
            if unsafe_instruction(hit.chunk.text):
                events.append(f"removed_unsafe_chunk:{hit.chunk.chunk_id}")
                continue
            hits.append(hit)
        return hits, events
