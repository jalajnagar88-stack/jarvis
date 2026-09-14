"""Abstract interfaces for every stage of the pipeline.

Implementations live in the sibling subpackages (``jarvis.audio``,
``jarvis.wake``, ``jarvis.stt``, ``jarvis.agent``, ``jarvis.tts``,
``jarvis.memory``) and are wired together only in ``jarvis.cli``. Importing
this module pulls in no heavy dependency and touches no hardware.
"""

from __future__ import annotations

from jarvis.interfaces.agent import (
    AgentEvent,
    AgentFailed,
    Brain,
    ConfirmationRequired,
    Role,
    SentenceComplete,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
)
from jarvis.interfaces.audio import (
    AudioClip,
    AudioInput,
    AudioOutput,
    DeviceInfo,
    Samples,
)
from jarvis.interfaces.memory import Fact, MemoryStore, StoredMessage
from jarvis.interfaces.stt import Transcriber, Transcript
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.interfaces.wake_word import WakeEvent, WakeWordDetector

__all__ = [
    "AgentEvent",
    "AgentFailed",
    "AudioClip",
    "AudioInput",
    "AudioOutput",
    "Brain",
    "ConfirmationRequired",
    "DeviceInfo",
    "Fact",
    "MemoryStore",
    "Role",
    "Samples",
    "SentenceComplete",
    "SpeechSynthesizer",
    "StoredMessage",
    "TextDelta",
    "ToolCallFinished",
    "ToolCallStarted",
    "Transcriber",
    "Transcript",
    "TurnFinished",
    "WakeEvent",
    "WakeWordDetector",
]
