import json
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from googlesearch import search

from core.llm import generate_completion
from core.memory import cosine_similarity, get_embedding
from database.models import DomainSource, SessionLocal

from .types import GraphData, RetrievalHit


CURATED_SITES: List[str] = [
    "wikipedia.org",
    "medium.com",
    "reddit.com",
    "plato.stanford.edu",
    "developer.mozilla.org",
    "arxiv.org",
    "fastapi.tiangolo.com",
    "freecodecamp.org",
    "smashingmagazine.com",
]


def _clean_json_payload(payload: str) -> str:
    return payload.replace("```json", "").replace("```", "").strip()


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def get_verified_sites_for_domain(
    domain: str,
    core_intent: str = "",
    article_archetype: str = "",
) -> List[str]:
    """Resolve trusted domains for a concept domain."""
    db = SessionLocal()
    try:
        domain_emb = get_embedding(domain)
        best_match = None
        best_score = -1.0

        all_sources = db.query(DomainSource).all()
        for src in all_sources:
            src_emb = src.get_embedding()
            if src_emb:
                score = cosine_similarity(domain_emb, src_emb)
                if score > best_score:
                    best_score = score
                    best_match = src

        if best_match and best_score > 0.85:
            print(
                f"[Articles Tool] Found semantic domain match: {best_match.domain_name} "
                f"(Score: {best_score:.2f})"
            )
            existing_sites = best_match.get_sites()
            combined_sites: List[str] = []
            for site in CURATED_SITES + existing_sites:
                if site not in combined_sites:
                    combined_sites.append(site)
            best_match.set_sites(combined_sites)
            db.commit()
            return combined_sites

        print(f"[Articles Tool] New Domain detected ({domain}). Using curated organic sources.")

        combined_sites = CURATED_SITES.copy()

        new_source = DomainSource(domain_name=domain)
        new_source.set_embedding(domain_emb)
        new_source.set_sites(combined_sites)
        db.add(new_source)
        db.commit()

        return combined_sites
    finally:
        db.close()


def generate_queries_from_graph(
    graph: GraphData,
    verified_sites: List[str],
    core_intent: str = "",
    article_archetype: str = "",
) -> List[str]:
    """Generate up to 3 focused site-filtered queries from graph intent."""
    if not verified_sites:
        verified_sites = ["wikipedia.org"]

    combined_text = f"{graph.get('domain', '')} {core_intent} {article_archetype}".lower()

    heuristic_domain = None
    if "fastapi" in combined_text and "fastapi.tiangolo.com" in verified_sites:
        heuristic_domain = "fastapi.tiangolo.com"
    elif any(
        token in combined_text
        for token in ["philosophy", "determinism", "free will", "stoic", "ethics"]
    ) and "plato.stanford.edu" in verified_sites:
        heuristic_domain = "plato.stanford.edu"
    elif any(
        token in combined_text for token in ["html", "css", "web", "website", "frontend"]
    ) and "developer.mozilla.org" in verified_sites:
        heuristic_domain = "developer.mozilla.org"
    elif any(token in combined_text for token in ["paper", "academic", "research"]) and "arxiv.org" in verified_sites:
        heuristic_domain = "arxiv.org"

    sniper_domain = heuristic_domain
    sites_filter = f" site:{sniper_domain}" if sniper_domain else ""
    if sniper_domain:
        print(f"[Articles Tool] Sniper Strategy selected primary domain: {sniper_domain}")
    else:
        print(f"[Articles Tool] Broad Search Strategy (no specific domain).")

    exact_phrase = str(graph.get("exact_phrase_weight", "")).strip()

    unknowns: List[str] = []
    for node in graph.get("nodes", []):
        if isinstance(node, dict) and node.get("status") in {"target_concept", "unknown_concept"}:
            node_id = str(node.get("id", "")).strip()
            if node_id:
                unknowns.append(node_id)

    unknowns_str = ", ".join(unknowns) if unknowns else str(graph.get("domain", "general tutorial"))

    messages = [
        {
            "role": "system",
            "content": (
                "You generate specific search queries. "
                f"Return exactly 3 targeted strings for topics: {unknowns_str}. "
                f"Use exact phrase \"{exact_phrase}\" when useful. "
                f"Append '{sites_filter}' to every query. Output JSON array only."
            ),
        },
        {"role": "user", "content": json.dumps(graph)},
    ]

    response = generate_completion(messages, temperature=0.1)
    try:
        queries = json.loads(_clean_json_payload(response))
        if isinstance(queries, list):
            clean_queries = [str(query).strip() for query in queries if str(query).strip()]
            if clean_queries:
                return clean_queries[:3]
    except Exception as exc:
        print(f"Error parsing queries JSON: {exc}")

    fallback_root = f'"{exact_phrase}" {unknowns_str}'.strip() if exact_phrase else unknowns_str
    return [
        f"{fallback_root} {sites_filter}".strip(),
        f"{str(graph.get('domain', 'topic'))} overview {sites_filter}".strip(),
        f"beginner guide {unknowns_str} {sites_filter}".strip(),
    ]


def execute_searches(queries: List[str]) -> List[RetrievalHit]:
    """Fetch and verify article-like pages from generated queries."""
    from duckduckgo_search import DDGS
    
    raw_results: List[RetrievalHit] = []
    seen_urls = set()

    for query in queries:
        try:
            print(f"[Scraper] Executing DuckDuckGo Search for: {query}")
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, backend="lite"))
                for res in results:
                    url = res.get("href", "")
                    title = res.get("title", "Untitled Article")
                    snippet = res.get("body", "")
                    
                    if not url or url in seen_urls:
                        continue
                        
                    seen_urls.add(url)
                    raw_results.append(
                        RetrievalHit(
                            url=url,
                            title=title,
                            snippet=snippet[:1000].strip() if snippet else "",
                            source_query=query,
                            retrieval_status="verified",
                        )
                    )
        except Exception as exc:
            print(f"[Scraper] DDGS failed for '{query}': {exc}")

    return raw_results
