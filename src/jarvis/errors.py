"""Exception hierarchy.

Every error JARVIS raises on purpose carries a plain-language explanation and,
where possible, the exact command that fixes it. The CLI prints ``remedy``
directly to the user, so write it as an instruction, not a diagnosis.
"""

from __future__ import annotations


class JarvisError(Exception):
    """Base class for every error JARVIS raises deliberately.

    Args:
        message: What went wrong, in plain language. No stack-trace jargon.
        remedy: What the user should do about it. Ideally a literal command.
    """

    def __init__(self, message: str, remedy: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def __str__(self) -> str:
        if self.remedy:
            return f"{self.message}\n  Try: {self.remedy}"
        return self.message


class ConfigError(JarvisError):
    """config.yaml is missing, malformed, or contains an invalid value."""


class DependencyMissingError(JarvisError):
    """An optional dependency is needed for the requested mode but not installed."""


class ModelMissingError(JarvisError):
    """A model file has not been downloaded yet."""


class AudioDeviceError(JarvisError):
    """No usable microphone or speaker, or the device could not be opened."""


class AuthError(JarvisError):
    """An API key is missing or was rejected."""


class ToolError(JarvisError):
    """A tool could not complete. Surfaced back to the model as a tool result."""


class SafetyViolation(ToolError):
    """A tool call was blocked by the safety layer.

    Raised for denylisted shell commands and for filesystem access outside the
    configured workspace. These are refused outright — there is no confirmation
    path that allows them through.
    """
