from typing import List

from core.memory import cosine_similarity, get_embedding

from .types import GraphData, RetrievalHit


def rank_results(results: List[RetrievalHit], graph: GraphData, limit: int = 3) -> List[RetrievalHit]:
    if not results:
        return []

    core_intent = str(graph.get("core_intent", "learning the basics"))
    print(f"[Articles Tool] Semantically re-ranking {len(results)} results against intent: '{core_intent}'")

    intent_emb = get_embedding(core_intent)

    for result in results:
        snippet_emb = get_embedding(result.snippet)
        result.score = cosine_similarity(intent_emb, snippet_emb)

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:limit]
