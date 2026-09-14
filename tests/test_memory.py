"""Memory: the SQLite store, recall, extraction, and the remember/forget tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import jarvis.tools  # noqa: F401 - registers the built-ins
from jarvis.audit import NullAuditLog
from jarvis.config import Config
from jarvis.memory.extraction import FactExtractor, _parse
from jarvis.memory.store import SqliteMemory
from jarvis.tools.dispatch import ToolDispatcher


class StubEmbedder:
    """Deterministic embeddings from word overlap.

    Not a real semantic model, but it gives related sentences genuinely similar
    vectors, which is what the recall logic is being tested on.
    """

    VOCAB: ClassVar[list[str]] = [
        "priya",
        "sister",
        "metric",
        "units",
        "edinburgh",
        "lives",
        "coffee",
        "tea",
        "vet",
        "works",
        "brother",
        "celsius",
        "milk",
        "allergic",
    ]

    def encode(self, text: str) -> np.ndarray:
        words = set(text.lower().replace(".", "").replace(",", "").split())
        vector = np.array(
            [1.0 if term in words else 0.0 for term in self.VOCAB] + [0.0],
            dtype=np.float32,
        )
        if not vector.any():
            # A dedicated "matched nothing" dimension, orthogonal to every real
            # one, so unrelated text scores zero rather than an artefact.
            vector[-1] = 1.0
        return vector


@pytest.fixture
def memory(tmp_path: Path) -> SqliteMemory:
    store = SqliteMemory(tmp_path / "jarvis.db", embedder=StubEmbedder())
    store.initialise()
    return store


@pytest.fixture
def keyword_memory(tmp_path: Path) -> SqliteMemory:
    """No embedder: exercises the degraded path."""
    store = SqliteMemory(tmp_path / "keyword.db")
    store.initialise()
    return store


class TestSchema:
    def test_initialise_is_idempotent(self, tmp_path: Path) -> None:
        store = SqliteMemory(tmp_path / "x.db")
        store.initialise()
        store.initialise()
        store.close()

    def test_the_database_file_is_created(self, memory: SqliteMemory) -> None:
        assert memory._path.is_file()

    def test_data_survives_a_restart(self, tmp_path: Path) -> None:
        """The whole point: remembering across sessions."""
        first = SqliteMemory(tmp_path / "x.db", embedder=StubEmbedder())
        first.initialise()
        first.add_fact("the user's sister is called Priya")
        first.close()

        second = SqliteMemory(tmp_path / "x.db", embedder=StubEmbedder())
        second.initialise()
        assert [fact.text for fact in second.all_facts()] == ["the user's sister is called Priya"]


class TestTranscript:
    def test_messages_are_stored_and_returned_in_order(self, memory: SqliteMemory) -> None:
        memory.add_message("user", "hello", session_id="s1")
        memory.add_message("assistant", "good evening", session_id="s1")
        messages = memory.recent_messages(10, session_id="s1")
        assert [m.text for m in messages] == ["hello", "good evening"]

    def test_sessions_are_kept_apart(self, memory: SqliteMemory) -> None:
        memory.add_message("user", "one", session_id="s1")
        memory.add_message("user", "two", session_id="s2")
        assert len(memory.recent_messages(10, session_id="s1")) == 1

    def test_the_limit_returns_the_most_recent(self, memory: SqliteMemory) -> None:
        for index in range(10):
            memory.add_message("user", f"message {index}", session_id="s1")
        recent = memory.recent_messages(3, session_id="s1")
        assert [m.text for m in recent] == ["message 7", "message 8", "message 9"]


class TestFacts:
    def test_a_fact_is_stored(self, memory: SqliteMemory) -> None:
        stored = memory.add_fact("the user prefers metric units")
        assert stored.text == "the user prefers metric units"
        assert stored.source == "extracted"

    def test_whitespace_is_normalised(self, memory: SqliteMemory) -> None:
        stored = memory.add_fact("  the   user  likes   tea  ")
        assert stored.text == "the user likes tea"

    def test_an_exact_duplicate_is_not_stored_twice(self, memory: SqliteMemory) -> None:
        memory.add_fact("the user prefers metric units")
        memory.add_fact("the user prefers metric units")
        assert memory.count_facts() == 1

    def test_a_near_duplicate_is_merged(self, memory: SqliteMemory) -> None:
        """Otherwise one preference accumulates a dozen wordings and crowds
        everything else out of recall."""
        memory.add_fact("the user prefers metric units")
        memory.add_fact("the user prefers metric units.")
        assert memory.count_facts() == 1

    def test_explicit_facts_record_their_source(self, memory: SqliteMemory) -> None:
        stored = memory.add_fact("the user's sister is called Priya", source="explicit")
        assert stored.source == "explicit"


class TestRecall:
    @pytest.fixture
    def populated(self, memory: SqliteMemory) -> SqliteMemory:
        memory.add_fact("the user's sister is called Priya")
        memory.add_fact("the user prefers metric units")
        memory.add_fact("the user lives in Edinburgh")
        memory.add_fact("the user is allergic to milk")
        return memory

    def test_relevant_facts_come_back_first(self, populated: SqliteMemory) -> None:
        facts = populated.recall("what is my sister called", top_k=2, min_similarity=0.1)
        assert facts
        assert "Priya" in facts[0].text

    def test_irrelevant_facts_are_excluded(self, populated: SqliteMemory) -> None:
        facts = populated.recall("tell me about Edinburgh", top_k=4, min_similarity=0.3)
        assert all("Priya" not in fact.text for fact in facts)

    def test_top_k_is_respected(self, populated: SqliteMemory) -> None:
        assert len(populated.recall("the user", top_k=2, min_similarity=0.0)) <= 2

    def test_top_k_of_zero_returns_nothing(self, populated: SqliteMemory) -> None:
        assert populated.recall("sister", top_k=0, min_similarity=0.0) == []

    def test_the_similarity_floor_is_applied(self, populated: SqliteMemory) -> None:
        assert populated.recall("quantum chromodynamics", top_k=5, min_similarity=0.9) == []

    def test_recall_on_an_empty_store_is_empty(self, memory: SqliteMemory) -> None:
        assert memory.recall("anything", top_k=5, min_similarity=0.1) == []

    def test_scores_are_attached(self, populated: SqliteMemory) -> None:
        facts = populated.recall("sister Priya", top_k=1, min_similarity=0.1)
        assert facts[0].score is not None


class TestKeywordFallback:
    """Memory must work without sentence-transformers installed."""

    def test_facts_are_still_stored(self, keyword_memory: SqliteMemory) -> None:
        keyword_memory.add_fact("the user's sister is called Priya")
        assert keyword_memory.count_facts() == 1

    def test_recall_still_finds_the_right_fact(self, keyword_memory: SqliteMemory) -> None:
        keyword_memory.add_fact("the user's sister is called Priya")
        keyword_memory.add_fact("the user lives in Edinburgh")
        facts = keyword_memory.recall("sister", top_k=2, min_similarity=0.0)
        assert facts
        assert "Priya" in facts[0].text

    def test_stopwords_alone_match_nothing(self, keyword_memory: SqliteMemory) -> None:
        keyword_memory.add_fact("the user likes tea")
        assert keyword_memory.recall("the a of", top_k=3, min_similarity=0.0) == []

    def test_a_broken_embedder_degrades_rather_than_failing(self, tmp_path: Path) -> None:
        class Broken:
            def encode(self, text: str) -> np.ndarray:
                raise RuntimeError("model exploded")

        store = SqliteMemory(tmp_path / "b.db", embedder=Broken())
        store.initialise()
        store.add_fact("the user's sister is called Priya")
        facts = store.recall("sister", top_k=2, min_similarity=0.0)
        assert facts and "Priya" in facts[0].text


class TestForget:
    @pytest.fixture
    def populated(self, memory: SqliteMemory) -> SqliteMemory:
        memory.add_fact("the user's sister is called Priya")
        memory.add_fact("the user lives in Edinburgh")
        return memory

    def test_a_matching_fact_is_deleted(self, populated: SqliteMemory) -> None:
        removed = populated.forget("my sister Priya")
        assert removed
        assert "Priya" in removed[0].text
        assert all("Priya" not in fact.text for fact in populated.all_facts())

    def test_what_was_removed_is_returned(self, populated: SqliteMemory) -> None:
        """So the assistant can confirm out loud exactly what it forgot."""
        removed = populated.forget("Edinburgh")
        assert [fact.text for fact in removed] == ["the user lives in Edinburgh"]

    def test_forgetting_nothing_is_not_an_error(self, populated: SqliteMemory) -> None:
        assert populated.forget("quantum chromodynamics") == []

    def test_an_empty_query_removes_nothing(self, populated: SqliteMemory) -> None:
        assert populated.forget("  ") == []
        assert populated.count_facts() == 2

    def test_it_does_not_delete_everything_on_a_vague_query(self, populated: SqliteMemory) -> None:
        """Forgetting more than was asked is worse than forgetting less: the
        user can repeat themselves, but cannot recover what is gone."""
        populated.forget("the user")
        assert populated.count_facts() >= 1

    def test_a_substring_match_works_without_embeddings(self, keyword_memory: SqliteMemory) -> None:
        keyword_memory.add_fact("the user's sister is called Priya")
        assert keyword_memory.forget("Priya")


class TestExtraction:
    def test_it_parses_a_json_array(self) -> None:
        assert _parse('["the user prefers tea"]') == ["the user prefers tea"]

    def test_it_tolerates_surrounding_prose(self) -> None:
        """Models sometimes explain themselves despite being told not to."""
        assert _parse('Here you go:\n["a fact"]\nHope that helps') == ["a fact"]

    def test_an_empty_array_is_the_common_case(self) -> None:
        assert _parse("[]") == []

    def test_unparseable_output_yields_nothing(self) -> None:
        assert _parse("not json at all") == []
        assert _parse("[unclosed") == []

    def test_non_strings_are_dropped(self) -> None:
        assert _parse('["good", 42, null, {"a": 1}]') == ["good"]

    def test_absurdly_long_facts_are_dropped(self) -> None:
        assert _parse(f'["{"x" * 500}"]') == []

    def test_extracted_facts_are_stored(self, cfg: Config, memory: SqliteMemory) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.messages = self

            def create(self, **kwargs: Any) -> Any:
                block = type(
                    "B", (), {"type": "text", "text": '["the user prefers metric units"]'}
                )()
                return type("M", (), {"content": [block]})()

        extractor = FactExtractor(cfg, memory, client_factory=FakeClient)
        extractor.consider("I prefer metric", "Noted.")
        extractor.wait()
        assert any("metric" in fact.text for fact in memory.all_facts())

    def test_it_uses_the_cheap_model(self, cfg: Config, memory: SqliteMemory) -> None:
        """This runs after every single turn; it must not be expensive."""
        seen: dict[str, Any] = {}

        class FakeClient:
            def __init__(self) -> None:
                self.messages = self

            def create(self, **kwargs: Any) -> Any:
                seen.update(kwargs)
                return type("M", (), {"content": []})()

        FactExtractor(cfg, memory, client_factory=FakeClient).extract("x", "y")
        assert seen["model"] == cfg.memory.extraction.model
        assert seen["model"] != cfg.llm.model

    def test_a_failure_never_reaches_the_user(self, cfg: Config, memory: SqliteMemory) -> None:
        """Losing a fact is minor; a traceback over a conversation is not."""

        class ExplodingClient:
            def __init__(self) -> None:
                self.messages = self

            def create(self, **kwargs: Any) -> Any:
                raise RuntimeError("API down")

        extractor = FactExtractor(cfg, memory, client_factory=ExplodingClient)
        extractor.consider("hello", "hi")
        extractor.wait()  # must not raise

    def test_it_is_skipped_when_disabled(self, cfg: Config, memory: SqliteMemory) -> None:
        cfg.memory.extraction.enabled = False
        called = False

        class FakeClient:
            def __init__(self) -> None:
                nonlocal called
                called = True
                self.messages = self

        extractor = FactExtractor(cfg, memory, client_factory=FakeClient)
        extractor.consider("hello", "hi")
        extractor.wait()
        assert not called

    def test_an_empty_exchange_is_skipped(self, cfg: Config, memory: SqliteMemory) -> None:
        extractor = FactExtractor(cfg, memory, client_factory=lambda: None)
        extractor.consider("", "")
        extractor.wait()
        assert memory.count_facts() == 0


class TestMemoryTools:
    @pytest.fixture
    def dispatcher(self, cfg: Config, memory: SqliteMemory) -> ToolDispatcher:
        return ToolDispatcher(cfg, NullAuditLog(), memory=memory)

    def test_remember_stores_a_fact(self, dispatcher: ToolDispatcher, memory: SqliteMemory) -> None:
        outcome = dispatcher.execute("remember_this", {"fact": "the user's sister is called Priya"})
        assert not outcome.is_error
        assert memory.count_facts() == 1

    def test_remembered_facts_are_marked_explicit(
        self, dispatcher: ToolDispatcher, memory: SqliteMemory
    ) -> None:
        dispatcher.execute("remember_this", {"fact": "the user prefers tea"})
        assert memory.all_facts()[0].source == "explicit"

    def test_an_empty_fact_is_refused(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("remember_this", {"fact": "   "})
        assert outcome.is_error

    def test_forget_removes_and_confirms(
        self, dispatcher: ToolDispatcher, memory: SqliteMemory
    ) -> None:
        memory.add_fact("the user's sister is called Priya")
        outcome = dispatcher.execute("forget_this", {"query": "sister Priya"})
        assert "Forgotten" in outcome.content
        assert memory.count_facts() == 0

    def test_forgetting_something_unknown_says_so(self, dispatcher: ToolDispatcher) -> None:
        outcome = dispatcher.execute("forget_this", {"query": "nonsense"})
        assert "nothing specific enough" in outcome.content

    def test_the_tools_report_clearly_when_memory_is_absent(self, cfg: Config) -> None:
        outcome = ToolDispatcher(cfg, NullAuditLog()).execute("remember_this", {"fact": "x"})
        assert outcome.is_error
        assert "not available" in outcome.content


class TestEmbedderLoading:
    """The optional embedding model, and what happens without it."""

    def test_a_missing_package_is_reported_once_and_is_not_fatal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins
        import sys

        from jarvis.memory.embeddings import SentenceTransformerEmbedder

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("sentence_transformers"):
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", tmp_path)
        assert embedder.load() is False
        assert embedder.encode("anything") is None
        assert embedder.is_available is False

    def test_a_model_that_will_not_load_degrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        from jarvis.memory.embeddings import SentenceTransformerEmbedder

        module = type(sys)("sentence_transformers")

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("no such model")

        module.SentenceTransformer = explode  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)

        embedder = SentenceTransformerEmbedder("nonexistent", tmp_path)
        assert embedder.load() is False

    def test_a_working_model_encodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from jarvis.memory.embeddings import SentenceTransformerEmbedder

        class FakeModel:
            def __init__(self, name: str, cache_folder: str) -> None:
                self.name = name

            def encode(self, text: str, show_progress_bar: bool = False) -> Any:
                return [0.1, 0.2, 0.3]

        module = type(sys)("sentence_transformers")
        module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", tmp_path)
        vector = embedder.encode("hello")
        assert vector is not None
        assert vector.shape == (3,)
        assert vector.dtype == np.float32

    def test_loading_is_deferred_until_first_use(self, tmp_path: Path) -> None:
        """Constructing one must be free, so the health check can ask about it."""
        from jarvis.memory.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2", tmp_path)
        assert embedder._model is None
