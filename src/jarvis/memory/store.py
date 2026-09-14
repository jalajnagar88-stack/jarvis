"""SQLite conversation log and fact store.

Two tables with different jobs. ``messages`` is an append-only transcript, kept
for review and for rebuilding context after a restart. ``facts`` holds short
durable statements about the user, each with an embedding, retrieved by
relevance and injected into the system prompt.

Everything is on the local disk and nothing is ever uploaded. Embeddings are
computed locally too, by sentence-transformers when it is installed.

Recall degrades rather than fails. With no embedding model, it falls back to
keyword overlap and says so in the log -- worse recall is a far better outcome
than a memory system that refuses to work.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.interfaces.memory import Fact, MemoryStore, StoredMessage
from jarvis.logging_setup import get_logger

log = get_logger("memory")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT    NOT NULL UNIQUE,
    source      TEXT    NOT NULL DEFAULT 'extracted',
    created_at  TEXT    NOT NULL,
    embedding   BLOB
);
CREATE INDEX IF NOT EXISTS idx_facts_created ON facts(created_at DESC);
"""

MAX_SUBSTRING_DELETIONS = 1
"""The literal-match fallback may delete only an unambiguous single hit.
Two matches means the query did not say which one, and guessing wrong cannot
be undone."""

FORGET_SIMILARITY = 0.55
"""Minimum similarity before a fact may be deleted. Higher than recall on
purpose -- see :meth:`SqliteMemory.forget`."""

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "my",
        "your",
        "our",
        "their",
        "his",
        "her",
        "its",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "about",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "i",
        "me",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "them",
        "us",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
    ]
)


class SqliteMemory(MemoryStore):
    """The local memory store."""

    def __init__(
        self,
        path: Path,
        *,
        embedder: Any = None,
        min_similarity: float = 0.35,
    ) -> None:
        self._path = path
        self._embedder = embedder
        self._min_similarity = min_similarity
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------- #

    def initialise(self) -> None:
        if self._connection is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the background fact extractor and the
        # timer threads both write; every access is serialised by self._lock.
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()
        log.debug("Memory ready at %s", self._path)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self.initialise()
        assert self._connection is not None
        return self._connection

    # -- transcript --------------------------------------------------------- #

    def add_message(self, role: str, text: str, *, session_id: str) -> StoredMessage:
        now = datetime.now(UTC)
        with self._lock:
            cursor = self._conn().execute(
                "INSERT INTO messages (session_id, role, text, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, text, now.isoformat()),
            )
            self._conn().commit()
        return StoredMessage(
            id=int(cursor.lastrowid or 0),
            role=role,
            text=text,
            created_at=now,
            session_id=session_id,
        )

    def recent_messages(self, limit: int, *, session_id: str | None = None) -> list[StoredMessage]:
        query = "SELECT * FROM messages"
        params: list[Any] = []
        if session_id is not None:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn().execute(query, params).fetchall()
        return [_as_message(row) for row in reversed(rows)]

    # -- facts -------------------------------------------------------------- #

    def add_fact(self, text: str, *, source: str = "extracted") -> Fact:
        """Store a fact, merging near-duplicates rather than stacking them.

        Without this, "prefers metric" would accumulate a dozen slightly
        differently worded copies and crowd everything else out of recall.
        """
        cleaned = " ".join(text.split())
        existing = self._find_duplicate(cleaned)
        if existing is not None:
            log.debug("Fact already known, not storing again: %s", cleaned)
            return existing

        embedding = self._embed(cleaned)
        now = datetime.now(UTC)
        with self._lock:
            cursor = self._conn().execute(
                "INSERT OR IGNORE INTO facts (text, source, created_at, embedding) "
                "VALUES (?, ?, ?, ?)",
                (cleaned, source, now.isoformat(), _pack(embedding)),
            )
            self._conn().commit()
            row_id = int(cursor.lastrowid or 0)

        log.info("Remembered: %s", cleaned)
        return Fact(id=row_id, text=cleaned, source=source, created_at=now)

    def _find_duplicate(self, text: str) -> Fact | None:
        """A stored fact close enough to ``text`` to count as the same thing."""
        with self._lock:
            row = self._conn().execute("SELECT * FROM facts WHERE text = ?", (text,)).fetchone()
        if row is not None:
            return _as_fact(row)

        matches = self.recall(text, top_k=1, min_similarity=0.92)
        return matches[0] if matches else None

    def recall(self, query: str, *, top_k: int, min_similarity: float) -> list[Fact]:
        """The facts most relevant to ``query``, best first."""
        if top_k <= 0:
            return []
        with self._lock:
            rows = self._conn().execute("SELECT * FROM facts").fetchall()
        if not rows:
            return []

        query_vector = self._embed(query)
        if query_vector is None:
            return self._recall_by_keyword(rows, query, top_k)

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            stored = _unpack(row["embedding"])
            if stored is None:
                continue
            scored.append((_cosine(query_vector, stored), row))

        if not scored:
            return self._recall_by_keyword(rows, query, top_k)

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            _as_fact(row, score=score) for score, row in scored[:top_k] if score >= min_similarity
        ]

    def _recall_by_keyword(self, rows: list[sqlite3.Row], query: str, top_k: int) -> list[Fact]:
        """Fallback when embeddings are unavailable.

        Worse than semantic recall, but a memory that works badly beats one that
        refuses to work at all.
        """
        log.debug("Recalling by keyword; no embeddings available.")
        wanted = _keywords(query)
        if not wanted:
            return []

        scored = []
        for row in rows:
            overlap = wanted & _keywords(row["text"])
            if overlap:
                scored.append((len(overlap) / len(wanted), row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_as_fact(row, score=score) for score, row in scored[:top_k]]

    def forget(self, query: str) -> list[Fact]:
        """Delete facts matching ``query``, returning what was removed.

        Returning them lets the assistant confirm out loud exactly what it
        forgot, which matters when the match was not what the user meant.
        """
        cleaned = " ".join(query.split())
        if not cleaned:
            return []

        # A stricter bar than ordinary recall. Deleting is irreversible: the
        # user can repeat a fact JARVIS failed to forget, but cannot recover
        # one it forgot by mistake.
        threshold = max(self._min_similarity, FORGET_SIMILARITY)
        candidates = self.recall(cleaned, top_k=5, min_similarity=threshold)
        if not candidates:
            candidates = self._forget_by_substring(cleaned)

        if not candidates:
            return []

        # Delete only the clear matches. Forgetting more than was asked is
        # worse than forgetting less: the user can repeat themselves, but
        # cannot recover what is gone.
        best = candidates[0].score
        chosen = [
            fact
            for fact in candidates
            if fact.score is None or best is None or fact.score >= best - 0.05
        ]
        with self._lock:
            self._conn().executemany(
                "DELETE FROM facts WHERE id = ?", [(fact.id,) for fact in chosen]
            )
            self._conn().commit()

        for fact in chosen:
            log.info("Forgot: %s", fact.text)
        return chosen

    def _forget_by_substring(self, query: str) -> list[Fact]:
        """Literal-match fallback, deliberately hard to use vaguely.

        A bare "the user" would otherwise match -- and delete -- everything.
        Stopwords are stripped first, and a query that still matches a broad
        swathe is refused rather than acted on.
        """
        meaningful = _keywords(query)
        if not meaningful:
            log.info("Refusing to forget on a query with no distinctive words: %r", query)
            return []

        with self._lock:
            rows = (
                self._conn()
                .execute("SELECT * FROM facts WHERE lower(text) LIKE ?", (f"%{query.lower()}%",))
                .fetchall()
            )

        if len(rows) > MAX_SUBSTRING_DELETIONS:
            log.info("Refusing to forget %d facts on the ambiguous query %r.", len(rows), query)
            return []
        return [_as_fact(row) for row in rows]

    def all_facts(self) -> list[Fact]:
        with self._lock:
            rows = self._conn().execute("SELECT * FROM facts ORDER BY created_at DESC").fetchall()
        return [_as_fact(row) for row in rows]

    def count_facts(self) -> int:
        with self._lock:
            row = self._conn().execute("SELECT COUNT(*) AS n FROM facts").fetchone()
        return int(row["n"])

    # -- embeddings --------------------------------------------------------- #

    def _embed(self, text: str) -> np.ndarray | None:
        if self._embedder is None:
            return None
        try:
            vector = self._embedder.encode(text)
        except Exception as exc:  # a broken embedder must not lose the fact
            log.error("Could not embed text (%r); falling back to keywords.", exc)
            return None
        return np.asarray(vector, dtype=np.float32).reshape(-1)


def _as_message(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        id=int(row["id"]),
        role=str(row["role"]),
        text=str(row["text"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        session_id=str(row["session_id"]),
    )


def _as_fact(row: sqlite3.Row, *, score: float | None = None) -> Fact:
    return Fact(
        id=int(row["id"]),
        text=str(row["text"]),
        source=str(row["source"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        score=score,
    )


def _pack(vector: np.ndarray | None) -> bytes | None:
    return None if vector is None else vector.astype(np.float32).tobytes()


def _unpack(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size != a.size:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)


def _keywords(text: str) -> set[str]:
    return {
        word.strip(".,!?'\"")
        for word in text.lower().split()
        if word not in _STOPWORDS and len(word) > 2
    }
