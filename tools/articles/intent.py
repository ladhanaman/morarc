import json
import re
from typing import Dict, List, Optional

from core.llm import generate_completion

from .types import GraphData


def _clean_json_payload(payload: str) -> str:
    return payload.replace("```json", "").replace("```", "").strip()


def detect_past_domain(active_rag_context: Optional[str]) -> Optional[str]:
    if not active_rag_context:
        return None
    match = re.search(r"Domain: (.*?)\\n", active_rag_context)
    if match:
        return match.group(1).strip()
    return None


def extract_concept_graph(chat_history: List[Dict[str, str]], past_domain: Optional[str] = None) -> GraphData:
    """Extract domain graph from tool conversation history."""
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    past_domain_prompt = ""
    if past_domain:
        past_domain_prompt = (
            f"The user previously studied under the domain '{past_domain}'. "
            "If this conversation deeply aligns with that, use it. "
            "Otherwise, define a NEW ultra-specific domain.\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Psychological Profiler and Knowledge Architect analyzing a conversation.\n"
                "Extract the true learning intent into this EXACT JSON structure for a Knowledge Graph:\n"
                "{\n"
                '  "domain": "The ULTRA-SPECIFIC architectural field",\n'
                '  "core_intent": "What the user ACTUALLY wants to feel, achieve, or understand.",\n'
                '  "article_archetype": "Ideal article format.",\n'
                '  "exact_phrase_weight": "The most vital 2-4 word exact string they used.",\n'
                '  "nodes": [{"id": "topic_name", "status": "known_concept | target_concept | unknown_concept"}],\n'
                '  "edges": [{"source": "node_id_1", "target": "node_id_2", "relationship": "relationship"}]\n'
                "}\n"
                f"{past_domain_prompt}"
                "Ensure all concepts are captured as nodes. Output ONLY pure JSON."
            ),
        },
        {"role": "user", "content": f"Conversation History:\n{history_text}"},
    ]

    response = generate_completion(messages, temperature=0.1)
    try:
        clean_resp = _clean_json_payload(response)
        parsed = json.loads(clean_resp)
        if isinstance(parsed, dict):
            parsed.setdefault("domain", "General Knowledge")
            parsed.setdefault("nodes", [])
            parsed.setdefault("edges", [])
            parsed.setdefault("core_intent", "learning the basics")
            parsed.setdefault("article_archetype", "general tutorial")
            parsed.setdefault("exact_phrase_weight", "")
            return parsed
    except Exception as exc:
        print(f"Error parsing concept graph JSON: {exc}")

    return {
        "domain": "General Knowledge",
        "nodes": [],
        "edges": [],
        "core_intent": "learning the basics",
        "article_archetype": "general tutorial",
        "exact_phrase_weight": "",
    }


def evaluate_search_readiness(chat_history: List[Dict[str, str]], turn_count: int) -> bool:
    """Decides whether there is enough context to start retrieval."""
    if turn_count >= 2:
        return True

    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a conversation bouncer. "
                "Does the AI have enough specific context to fetch personalized reading links? "
                "CRITICAL: If user asks to stop questions or asks for links immediately, output YES. "
                "Output only YES or NO."
            ),
        },
        {"role": "user", "content": f"Conversation:\n{history_text}"},
    ]
    response = generate_completion(messages, temperature=0.1)
    return "YES" in response.upper()


def build_challenger_reply(
    tool_history: List[Dict[str, str]],
    current_graph: GraphData,
    active_rag_context: Optional[str],
) -> str:
    target_nodes = [
        node.get("id")
        for node in current_graph.get("nodes", [])
        if isinstance(node, dict) and node.get("status") == "target_concept"
    ]
    targets = ", ".join([node for node in target_nodes if node]) or "this new topic"

    edges = []
    for edge in current_graph.get("edges", []):
        if isinstance(edge, dict):
            source = edge.get("source", "")
            relation = edge.get("relationship", "")
            target = edge.get("target", "")
            edge_text = " ".join([str(source), str(relation), str(target)]).strip()
            if edge_text:
                edges.append(edge_text)
    edges_str = ", ".join(edges) if edges else "None"

    core_intent = current_graph.get("core_intent", "Learn the basics")

    system_prompt = (
        "You are Morarc, a sharp, witty, and slightly provocative 'Socratic Challenger'. "
        "Your job is to figure out the user's true underlying intent so you can find perfect articles. "
        f"The user wants to explore: {targets}. Their initially detected intent is: '{core_intent}'. "
        f"You mapped these conceptual relationships: {edges_str}. "
        "Ask ONE concise, conversational, high-signal question. "
        "No emojis. No filler."
    )

    if active_rag_context:
        system_prompt += f"\n\n{active_rag_context}"

    messages = [{"role": "system", "content": system_prompt}] + tool_history
    return generate_completion(messages, temperature=0.7)
