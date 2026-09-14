"""Memory interface: the conversation log and the durable facts.

Two different things live behind one interface:

* **messages** — an append-only transcript, for review and for rebuilding
  context after a restart;
* **facts** — short durable statements ("prefers metric", "sister is called
  Priya") with embeddings, retrieved by relevance and injected into the system
  prompt each turn.

Both are SQLite on the local disk. Neither ever leaves the machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: int
    role: str
    text: str
    created_at: datetime
    session_id: str


@dataclass(frozen=True, slots=True)
class Fact:
    """One durable thing JARVIS knows about the user."""

    id: int
    text: str
    source: str
    """How it was learned: ``extracted`` from conversation, or ``explicit`` via
    the remember_this tool."""
    created_at: datetime
    score: float | None = None
    """Similarity to the query, populated only by :meth:`MemoryStore.recall`."""


class MemoryStore(ABC):
    """Persistent conversation log and fact store."""

    @abstractmethod
    def initialise(self) -> None:
        """Create tables if they do not exist. Idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Release the database connection. Must be idempotent."""

    # -- transcript --------------------------------------------------------- #

    @abstractmethod
    def add_message(self, role: str, text: str, *, session_id: str) -> StoredMessage:
        """Append one turn to the transcript and return it with its assigned id."""

    @abstractmethod
    def recent_messages(self, limit: int, *, session_id: str | None = None) -> list[StoredMessage]:
        """Return the most recent messages, oldest first.

        Args:
            limit: How many to return, counting back from the latest.
            session_id: Restrict to one session, or None for all sessions.
        """

    # -- facts -------------------------------------------------------------- #

    @abstractmethod
    def add_fact(self, text: str, *, source: str = "extracted") -> Fact:
        """Store a fact. Near-duplicates should be merged, not stacked."""

    @abstractmethod
    def recall(self, query: str, *, top_k: int, min_similarity: float) -> list[Fact]:
        """Return the facts most relevant to ``query``, best first.

        Must degrade rather than fail when embeddings are unavailable: fall back
        to keyword matching and say so in the log.
        """

    @abstractmethod
    def forget(self, query: str) -> list[Fact]:
        """Delete facts matching ``query``. Returns what was removed so the
        assistant can confirm out loud exactly what it forgot."""

    @abstractmethod
    def all_facts(self) -> list[Fact]:
        """Every stored fact, newest first. For review and for the `facts` command."""
