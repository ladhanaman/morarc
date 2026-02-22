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

        print(f"[Articles Tool] New Domain detected ({domain}). Generating organic sources...")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert librarian and domain sniper. "
                    f"The user wants to explore '{domain}'. "
                    f"Core intent: '{core_intent}'. Desired format: '{article_archetype}'. "
                    "Return exactly 6 best website domain names as pure JSON array."
                ),
            },
            {"role": "user", "content": f"Subject: {domain}"},
        ]
        response = generate_completion(messages, temperature=0.2)

        try:
            proposed_sites = json.loads(_clean_json_payload(response))
            if not isinstance(proposed_sites, list):
                proposed_sites = []
        except Exception:
            proposed_sites = []

        if not proposed_sites:
            proposed_sites = ["wikipedia.org", "medium.com", "reddit.com"]

        verified_sites: List[str] = []
        for site in proposed_sites:
            if not isinstance(site, str):
                continue
            normalized = site.replace("https://", "").replace("http://", "").split("/")[0]
            if not normalized:
                continue
            url = f"https://{normalized}"
            try:
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
                if response.status_code == 200:
                    verified_sites.append(normalized)
            except Exception as exc:
                print(f"[Ping Failed] {normalized}: {exc}")

        combined_sites: List[str] = []
        for site in CURATED_SITES + verified_sites:
            if site not in combined_sites:
                combined_sites.append(site)

        if not combined_sites:
            combined_sites = ["wikipedia.org", "medium.com"]

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
    if not sniper_domain:
        sniper_messages = [
            {
                "role": "system",
                "content": (
                    f"Given core intent '{core_intent}' and format '{article_archetype}', "
                    "pick the SINGLE BEST domain from list. Return domain only."
                ),
            },
            {"role": "user", "content": f"Options: {', '.join(verified_sites)}"},
        ]
        sniper_domain = generate_completion(sniper_messages, temperature=0.1).strip()

    if sniper_domain not in verified_sites:
        sniper_domain = verified_sites[0]
    if sniper_domain not in CURATED_SITES:
        curated_available = [site for site in verified_sites if site in CURATED_SITES]
        if curated_available:
            sniper_domain = curated_available[0]

    sites_filter = f"site:{sniper_domain}"
    print(f"[Articles Tool] Sniper Strategy selected primary domain: {sniper_domain}")

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
    raw_results: List[RetrievalHit] = []
    seen_urls = set()

    for query in queries:
        try:
            print(f"[Scraper] Executing Google Search for: {query}")
            for url in search(query, num_results=3, sleep_interval=2):
                if not isinstance(url, str) or not _is_http_url(url) or url in seen_urls:
                    continue

                try:
                    headers = {
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        ),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    }
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code != 200:
                        continue

                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        continue

                    soup = BeautifulSoup(response.text, "html.parser")
                    title = soup.title.string.strip() if soup.title and soup.title.string else "Untitled Article"

                    for tag in soup(["script", "style", "nav", "header", "footer"]):
                        tag.decompose()

                    snippet = soup.get_text(separator=" ", strip=True)[:1000].strip()
                    if not snippet:
                        continue

                    seen_urls.add(url)
                    raw_results.append(
                        RetrievalHit(
                            url=url,
                            title=title,
                            snippet=snippet,
                            source_query=query,
                            retrieval_status="verified",
                        )
                    )
                except Exception as exc:
                    print(f"[Scraper] Failed to extract snippet from {url}: {str(exc)[:80]}")
                    continue
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str:
                print("[Scraper] Error 429 hit. Stopping search to avoid IP ban.")
                break
            print(f"[Scraper] Search failed for '{query}': {err_str[:80]}")

    return raw_results
