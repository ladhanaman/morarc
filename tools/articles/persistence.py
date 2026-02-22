import json
from typing import Optional

from core.llm import generate_completion
from core.memory import Session, get_embedding
from database.models import ConceptGraph, SessionLocal, User

from .types import GraphData


def _clean_json_payload(payload: str) -> str:
    return payload.replace("```json", "").replace("```", "").strip()


def persist_concept_graph(session: Session, final_graph: GraphData) -> None:
    """Create or merge user's concept graph in DB and refresh embeddings."""
    domain = str(final_graph.get("domain", "General Knowledge"))

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.phone_number == session.phone_number).first()
        if not user:
            return

        existing_graph: Optional[ConceptGraph] = None
        if session.active_graph_id:
            existing_graph = db.query(ConceptGraph).filter(ConceptGraph.id == session.active_graph_id).first()

        if existing_graph and existing_graph.domain == domain:
            print(f"[Articles Tool] Merging into existing domain graph: {domain}")
            old_data = existing_graph.get_graph_data()

            merge_messages = [
                {
                    "role": "system",
                    "content": (
                        "Merge OLD and NEW knowledge graphs into one JSON with fields: domain, nodes, edges. "
                        "Append new nodes/edges and update node status when changed. Output JSON only."
                    ),
                },
                {"role": "user", "content": f"OLD:\n{json.dumps(old_data)}\n\nNEW:\n{json.dumps(final_graph)}"},
            ]
            merged_response = generate_completion(merge_messages, temperature=0.1)
            try:
                merged_graph = json.loads(_clean_json_payload(merged_response))
                if isinstance(merged_graph, dict):
                    final_graph = merged_graph
            except Exception as exc:
                print(f"Error during graph merge parse: {exc}")

            if len(final_graph.get("nodes", [])) > 20:
                compression_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Compress this graph by at least 40% node count while preserving core understanding. "
                            "Output JSON only with domain, nodes, edges."
                        ),
                    },
                    {"role": "user", "content": json.dumps(final_graph)},
                ]
                compressed_response = generate_completion(compression_messages, temperature=0.1)
                try:
                    compressed_graph = json.loads(_clean_json_payload(compressed_response))
                    if isinstance(compressed_graph, dict):
                        final_graph = compressed_graph
                except Exception as exc:
                    print(f"Error during graph compression parse: {exc}")

            existing_graph.set_graph_data(final_graph)
            existing_graph.set_embedding(get_embedding(domain))
            db.commit()
            session.active_graph_id = existing_graph.id
            return

        print(f"[Articles Tool] Creating new domain graph: {domain}")
        new_graph = ConceptGraph(user_id=user.id, domain=domain)
        new_graph.set_graph_data(final_graph)
        new_graph.set_embedding(get_embedding(domain))
        db.add(new_graph)
        db.commit()
        session.active_graph_id = new_graph.id
    except Exception as exc:
        print(f"Error saving concept graph to Database: {exc}")
        db.rollback()
    finally:
        db.close()
