from typing import List

from core.llm import generate_completion

from .types import GraphData, RetrievalHit


def _summarize_hit(hit: RetrievalHit, core_intent: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Write exactly two concise sentences about why this article is useful for the user's intent. "
                "Use only provided title/snippet. No markdown, no emojis, no links in summary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Core intent: {core_intent}\n"
                f"Title: {hit.title}\n"
                f"Snippet: {hit.snippet}"
            ),
        },
    ]
    summary = generate_completion(messages, temperature=0.2).strip()
    if not summary:
        return "This article appears relevant to your stated goal based on the extracted snippet. It is a verified source from the retrieval step."
    return summary


def format_articles_response(
    graph: GraphData,
    ranked_hits: List[RetrievalHit],
    queries: List[str],
) -> str:
    """Render only retrieval-grounded output; never fabricate article links."""
    core_intent = str(graph.get("core_intent", "learning the basics"))

    if not ranked_hits:
        attempted_queries = "\n".join([f"- {query}" for query in queries]) if queries else "- (none)"
        return (
            "I couldn't retrieve verified article pages right now (likely rate limits or temporary blocks).\n\n"
            "You can retry in a minute, or refine your topic with a narrower angle.\n\n"
            "Queries attempted:\n"
            f"{attempted_queries}"
        )

    header = f"Found {len(ranked_hits)} verified article(s) (up to 3)."
    blocks: List[str] = []
    for hit in ranked_hits:
        summary = _summarize_hit(hit, core_intent)
        blocks.append(f"Link: {hit.url}\nSummary: {summary}")

    return header + "\n\n" + "\n\n".join(blocks)
