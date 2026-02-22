import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator

import numpy as np
from sentence_transformers import SentenceTransformer

from database.models import SessionLocal, User, ConceptGraph

# Load local embedding model (free, runs locally and very fast)
print("Loading semantic embedding model for RAG...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded successfully.")


def get_embedding(text: str) -> List[float]:
    """Generate vector embedding for a given text."""
    return embedder.encode(text).tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate semantic similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
        return 0.0
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


class Session:
    def __init__(self, phone_number: str):
        self.phone_number: str = phone_number
        self.chat_history: List[Dict[str, str]] = []
        self.tool_stack: List[str] = []
        self.active_rag_context: Optional[str] = None
        self.active_graph_id: Optional[int] = None
        self.tool_start_idx: int = 0

    def add_message(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})

    def push_tool(self, tool_name: str) -> None:
        if self.tool_stack:
            raise ValueError(
                f"Cannot open '{tool_name}'. You must finish or stop '{self.tool_stack[-1]}' first."
            )
        self.tool_stack.append(tool_name)
        self.tool_start_idx = len(self.chat_history)

    def pop_tool(self) -> Optional[str]:
        if self.tool_stack:
            self.active_rag_context = None
            self.active_graph_id = None
            self.tool_start_idx = 0
            return self.tool_stack.pop()
        return None

    def get_current_tool(self) -> Optional[str]:
        if self.tool_stack:
            return self.tool_stack[-1]
        return None


@dataclass
class _LockState:
    lock: threading.RLock
    ref_count: int = 0


class MemoryManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self._session_locks: Dict[str, _LockState] = {}
        self._registry_lock = threading.Lock()

    @contextmanager
    def get_session_lock(self, phone_number: str) -> Iterator[None]:
        lock_state: _LockState
        with self._registry_lock:
            lock_state = self._session_locks.get(phone_number)
            if lock_state is None:
                lock_state = _LockState(lock=threading.RLock())
                self._session_locks[phone_number] = lock_state
            lock_state.ref_count += 1

        with lock_state.lock:
            try:
                yield
            finally:
                with self._registry_lock:
                    current = self._session_locks.get(phone_number)
                    if current is not None:
                        current.ref_count = max(0, current.ref_count - 1)
                        # Safe cleanup: delete lock only when unused and no active session exists.
                        if current.ref_count == 0 and phone_number not in self.sessions:
                            self._session_locks.pop(phone_number, None)

    def get_or_create_session(self, phone_number: str) -> Session:
        with self._registry_lock:
            if phone_number not in self.sessions:
                self.sessions[phone_number] = Session(phone_number)
            return self.sessions[phone_number]

    def clear_session(self, phone_number: str) -> None:
        with self._registry_lock:
            self.sessions.pop(phone_number, None)
            lock_state = self._session_locks.get(phone_number)
            if lock_state and lock_state.ref_count == 0:
                self._session_locks.pop(phone_number, None)

    def retrieve_and_inject_rag(self, session: Session, query: str) -> None:
        """
        Embeds the user's initial prompt, searches the database for their past
        concept graphs, and injects the highest match (>0.6) into the active session.
        """
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.phone_number == session.phone_number).first()
            if not user:
                return

            past_graphs = db.query(ConceptGraph).filter(ConceptGraph.user_id == user.id).all()
            if not past_graphs:
                return

            query_embedding = get_embedding(query)

            best_match = None
            best_score = -1.0
            for graph in past_graphs:
                graph_embedding = graph.get_embedding()
                if graph_embedding:
                    score = cosine_similarity(query_embedding, graph_embedding)
                    if score > best_score:
                        best_score = score
                        best_match = graph

            if best_match and best_score > 0.6:
                print(
                    f"[RAG] Found historical domain match: '{best_match.domain}' "
                    f"(Score: {best_score:.2f})"
                )

                graph_data = best_match.get_graph_data()
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])

                node_strs: List[str] = []
                for node in nodes:
                    if isinstance(node, dict):
                        node_strs.append(
                            f"{node.get('id', 'unknown')} ({node.get('status', 'unknown')})"
                        )
                    else:
                        node_strs.append(str(node))

                edge_strs: List[str] = []
                for edge in edges:
                    if isinstance(edge, dict):
                        edge_strs.append(
                            f"{edge.get('source')} -> {edge.get('target')} ({edge.get('relationship')})"
                        )
                    else:
                        edge_strs.append(str(edge))

                context_str = (
                    f"RAG CONTEXT - User previously explored the Domain: {best_match.domain}\n"
                    f"Historical Nodes: {', '.join(node_strs)}\n"
                    f"Historical Edges: {', '.join(edge_strs)}\n"
                    "Use this historical graph purely as context. If their new query fits within "
                    "this domain, playfully reference their past learning. Do NOT rigidly force "
                    "the conversation to conform to old nodes if their goals have changed."
                )
                session.active_rag_context = context_str
                session.active_graph_id = best_match.id

        finally:
            db.close()


# Global memory instance
memory = MemoryManager()
