from core.memory import Session

from .intent import (
    build_challenger_reply,
    detect_past_domain,
    evaluate_search_readiness,
    extract_concept_graph,
)
from .persistence import persist_concept_graph
from .ranking import rank_results
from .response import format_articles_response
from .retrieval import execute_searches, generate_queries_from_graph, get_verified_sites_for_domain


def _get_tool_history(session: Session):
    return session.chat_history[session.tool_start_idx :]


def handle_articles_tool(session: Session, message: str) -> str:
    """Conversation entry point for the /articles tool."""
    session.add_message("user", message)

    if message.lower() == "done":
        session.pop_tool()
        return "You have exited the Articles Tool. You are back in standard chat."

    tool_history = _get_tool_history(session)
    tool_messages = len(tool_history)
    turn_count = tool_messages // 2
    print(f"[Articles Tool] Evaluating Search Readiness (Turn {turn_count})...")

    is_ready = evaluate_search_readiness(tool_history, turn_count)
    past_domain = detect_past_domain(session.active_rag_context)

    if not is_ready:
        print("[Articles Tool] Not ready. Invoking Playful Teacher persona...")
        current_graph = extract_concept_graph(tool_history, past_domain=past_domain)
        reply = build_challenger_reply(tool_history, current_graph, session.active_rag_context)
        session.add_message("assistant", reply)
        return reply

    print("[Articles Tool] Ready threshold met! Finalizing Knowledge Graph...")
    final_graph = extract_concept_graph(tool_history, past_domain=past_domain)
    domain = str(final_graph.get("domain", "General Knowledge"))

    persist_concept_graph(session, final_graph)

    try:
        core_intent = str(final_graph.get("core_intent", ""))
        article_archetype = str(final_graph.get("article_archetype", ""))

        verified_sites = get_verified_sites_for_domain(domain, core_intent, article_archetype)
        queries = generate_queries_from_graph(final_graph, verified_sites, core_intent, article_archetype)
        raw_results = execute_searches(queries)
        ranked_results = rank_results(raw_results, final_graph, limit=3)
        final_output = format_articles_response(final_graph, ranked_results, queries)

        response = "Your interested concept graph fully evoked.\n\n"
        response += final_output
        response += "\n\n_(Reply with '/stop' to end chat or 'done' to exit tool)_"

        session.pop_tool()
        return response
    except Exception as exc:
        session.pop_tool()
        print(f"An error occurred while generating articles: {exc}")
        return "An error occurred while generating articles. Please try again."
