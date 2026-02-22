import os
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

# Lightweight stubs to avoid heavyweight model/client initialization in tests.
class _FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, text):
        value = float((len(str(text)) % 7) + 1)
        return [value, 1.0, 0.5]


class _FakeCompletions:
    def create(self, **_kwargs):
        message = types.SimpleNamespace(content="stub")
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class _FakeGroq:
    def __init__(self, *_args, **_kwargs):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())


sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
sys.modules["groq"] = types.SimpleNamespace(Groq=_FakeGroq)

from database.models import SessionLocal, User, init_db
from main import split_message_for_whatsapp
from tools.articles.response import format_articles_response
from tools.articles.types import RetrievalHit
from tools import router
from core import llm


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def _ensure_user(self, phone: str, name: str = "tester", welcomed: bool = True):
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.phone_number == phone).first()
            if not user:
                user = User(phone_number=phone, name=name, has_been_welcomed=welcomed)
                db.add(user)
            else:
                user.has_been_welcomed = welcomed
            db.commit()
        finally:
            db.close()

    def test_same_user_messages_are_serialized(self):
        phone = "whatsapp:+15550000001"
        self._ensure_user(phone)

        active = 0
        peak_active = 0
        guard = threading.Lock()

        def fake_route_tool(session, message):
            nonlocal active, peak_active
            with guard:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return "ok"

        with patch.object(router, "route_tool", side_effect=fake_route_tool):
            threads = [
                threading.Thread(target=router.route_message, args=(phone, f"msg {index}"))
                for index in range(6)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(peak_active, 1)

    def test_different_users_can_run_in_parallel(self):
        phone_a = "whatsapp:+15550000002"
        phone_b = "whatsapp:+15550000003"
        self._ensure_user(phone_a)
        self._ensure_user(phone_b)

        active = 0
        peak_active = 0
        guard = threading.Lock()

        def fake_route_tool(session, message):
            nonlocal active, peak_active
            with guard:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return "ok"

        with patch.object(router, "route_tool", side_effect=fake_route_tool):
            threads = [
                threading.Thread(target=router.route_message, args=(phone_a, "hi")),
                threading.Thread(target=router.route_message, args=(phone_b, "hi")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertGreaterEqual(peak_active, 2)

    def test_articles_response_without_hits_has_no_fabricated_links(self):
        output = format_articles_response(
            graph={"core_intent": "learn concurrency"},
            ranked_hits=[],
            queries=["python lock tutorial"],
        )
        self.assertIn("couldn't retrieve verified article pages", output.lower())
        self.assertNotIn("Link:", output)

    def test_articles_response_with_partial_hits_uses_only_verified_urls(self):
        hits = [
            RetrievalHit(
                url="https://example.com/a",
                title="A",
                snippet="A snippet about locks and sessions.",
                source_query="locks",
            ),
            RetrievalHit(
                url="https://example.com/b",
                title="B",
                snippet="B snippet about thread safety.",
                source_query="threads",
            ),
        ]

        with patch("tools.articles.response._summarize_hit", return_value="Two useful sentences."):
            output = format_articles_response(
                graph={"core_intent": "learn locks"},
                ranked_hits=hits,
                queries=["locks"],
            )

        self.assertIn("https://example.com/a", output)
        self.assertIn("https://example.com/b", output)
        self.assertNotIn("https://example.com/c", output)

    def test_router_articles_reentry_is_safe(self):
        phone = "whatsapp:+15550000004"
        self._ensure_user(phone)

        first = router.route_message(phone, "/articles")
        second = router.route_message(phone, "/articles")

        self.assertIn("What concept would you like to explore?", first)
        self.assertIn("already in /articles", second)

    def test_chunking_hard_splits_oversized_blocks(self):
        message = "x" * 3600
        chunks = split_message_for_whatsapp(message, max_len=1000)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))
        self.assertGreaterEqual(len(chunks), 4)

    def test_llm_error_output_is_sanitized(self):
        with patch.object(
            llm.client.chat.completions,
            "create",
            side_effect=Exception("SecretTokenValue should not leak"),
        ):
            response = llm.generate_completion([{"role": "user", "content": "hi"}])

        self.assertIn("temporary AI service issue", response)
        self.assertNotIn("SecretTokenValue", response)


if __name__ == "__main__":
    unittest.main()
