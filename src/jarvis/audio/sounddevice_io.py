"""Microphone capture and speaker playback via sounddevice/PortAudio.

``sounddevice`` is imported lazily, inside methods, and never at module scope.
Importing it raises ``OSError`` when the native PortAudio library is absent --
the single most common macOS setup failure -- and the health check must be able
to report that rather than die of it.

Both classes convert to and from the package's common currency on the boundary:
mono ``float32`` in ``[-1, 1]``.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Iterator
from typing import Any

import numpy as np

from jarvis.audio.resample import resample
from jarvis.errors import AudioDeviceError, DependencyMissingError
from jarvis.interfaces.audio import AudioClip, AudioInput, AudioOutput, Samples
from jarvis.logging_setup import get_logger

log = get_logger("audio")


def _import_sounddevice() -> Any:
    """Import sounddevice, translating both failure modes into our errors."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise DependencyMissingError(
            "sounddevice is not installed, so JARVIS cannot use the microphone or speakers.",
            remedy="uv sync --extra voice",
        ) from exc
    except OSError as exc:
        raise AudioDeviceError(
            f"sounddevice is installed but its native PortAudio library will not load ({exc}).",
            remedy=(
                "On macOS: brew install portaudio, then "
                "uv sync --extra voice --reinstall-package sounddevice. "
                "On Debian/Ubuntu: sudo apt install libportaudio2"
            ),
        ) from exc
    return sd


def resolve_device(selector: int | str | None, *, want_input: bool) -> int | None:
    """Turn a config selector into a PortAudio device index.

    Accepts an index, a case-insensitive substring of the device name, or None
    for the system default. Returning None means "let PortAudio choose", which
    is what sounddevice wants for the default device.
    """
    if selector is None:
        return None

    sd = _import_sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:  # PortAudio raises assorted host errors
        raise AudioDeviceError(
            f"PortAudio could not list audio devices ({exc}).",
            remedy="python -m jarvis devices",
        ) from exc

    channel_key = "max_input_channels" if want_input else "max_output_channels"
    kind = "input" if want_input else "output"

    if isinstance(selector, int):
        if 0 <= selector < len(devices) and devices[selector][channel_key] > 0:
            return selector
        raise AudioDeviceError(
            f"Device index {selector} is not a usable {kind} device.",
            remedy="python -m jarvis devices  # pick an index from the list",
        )

    needle = selector.lower()
    for index, device in enumerate(devices):
        if needle in str(device["name"]).lower() and device[channel_key] > 0:
            return index
    raise AudioDeviceError(
        f"No {kind} device matches {selector!r}.",
        remedy="python -m jarvis devices  # pick a name or index from the list",
    )


class SoundDeviceInput(AudioInput):
    """Microphone capture.

    PortAudio delivers audio on its own high-priority thread; that thread must
    never block or allocate much, so it does nothing but drop a copy into a
    bounded queue. The queue is bounded on purpose: if the consumer stalls (a
    slow transcription, a long reply), we discard the oldest audio rather than
    growing without limit and then replaying minutes of stale sound.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        device: int | str | None = None,
        max_buffered_blocks: int = 100,
    ) -> None:
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._device_selector = device
        self._queue: queue.Queue[Samples] = queue.Queue(maxsize=max_buffered_blocks)
        self._stream: Any = None
        self._running = threading.Event()
        self._dropped = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def dropped_blocks(self) -> int:
        """Blocks discarded because the consumer could not keep up."""
        return self._dropped

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _import_sounddevice()
        device = resolve_device(self._device_selector, want_input=True)

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                # Overflows are expected under load and are not worth a warning
                # per occurrence; the dropped-block counter is the real signal.
                log.debug("Input stream status: %s", status)
            block = indata[:, 0].astype(np.float32, copy=True)
            try:
                self._queue.put_nowait(block)
            except queue.Full:
                self._dropped += 1
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(block)
                except (queue.Empty, queue.Full):
                    pass

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                device=device,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioDeviceError(
                f"Could not open the microphone at {self._sample_rate} Hz ({exc}).",
                remedy=(
                    "Check that another application is not holding the microphone, and "
                    "that microphone access is granted under System Settings > Privacy & "
                    "Security > Microphone. Run `python -m jarvis devices` to pick another."
                ),
            ) from exc

        self._running.set()
        log.debug("Microphone open: %d Hz, %d-frame blocks", self._sample_rate, self._block_size)

    def stop(self) -> None:
        self._running.clear()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # closing a dead stream must not mask the real error
                log.debug("Ignoring error while closing the microphone: %r", exc)
        self.flush()

    def blocks(self) -> Iterator[Samples]:
        while self._running.is_set():
            try:
                # The timeout is what lets stop() end this loop promptly.
                yield self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

    def flush(self) -> int:
        discarded = 0
        while True:
            try:
                self._queue.get_nowait()
                discarded += 1
            except queue.Empty:
                break
        if discarded:
            log.debug("Discarded %d stale audio block(s)", discarded)
        return discarded


class SoundDeviceOutput(AudioOutput):
    """Speaker playback with immediate cancellation.

    :meth:`play` returns as soon as the clip is queued. The PortAudio callback
    pulls from that queue, so :meth:`cancel` takes effect within one callback
    (a few milliseconds) rather than at the end of the current sentence. That
    property is what milestone 6's barge-in is built on, so it is implemented
    now rather than retrofitted.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        device: int | str | None = None,
        block_size: int = 1024,
    ) -> None:
        self._sample_rate = sample_rate
        self._device_selector = device
        self._block_size = block_size
        self._stream: Any = None
        self._lock = threading.Lock()
        self._pending: deque[Samples] = deque()
        self._cursor = 0
        self._drained = threading.Event()
        self._drained.set()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _import_sounddevice()
        device = resolve_device(self._device_selector, want_input=False)

        def callback(outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                log.debug("Output stream status: %s", status)
            outdata[:] = 0.0
            written = 0
            with self._lock:
                while written < frames and self._pending:
                    chunk = self._pending[0]
                    available = len(chunk) - self._cursor
                    take = min(available, frames - written)
                    outdata[written : written + take, 0] = chunk[self._cursor : self._cursor + take]
                    written += take
                    self._cursor += take
                    if self._cursor >= len(chunk):
                        self._pending.popleft()
                        self._cursor = 0
                if not self._pending:
                    self._drained.set()

        try:
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                device=device,
                channels=1,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioDeviceError(
                f"Could not open the speaker at {self._sample_rate} Hz ({exc}).",
                remedy="python -m jarvis devices  # then set audio.output_device in config.yaml",
            ) from exc

        log.debug("Speaker open: %d Hz", self._sample_rate)

    def stop(self) -> None:
        self.cancel()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.debug("Ignoring error while closing the speaker: %r", exc)

    def play(self, clip: AudioClip) -> None:
        samples = clip.samples
        if clip.sample_rate != self._sample_rate:
            samples = resample(samples, clip.sample_rate, self._sample_rate)
        if samples.size == 0:
            return
        with self._lock:
            self._pending.append(samples.astype(np.float32, copy=False))
            self._drained.clear()

    def cancel(self) -> None:
        with self._lock:
            discarded = len(self._pending)
            self._pending.clear()
            self._cursor = 0
            self._drained.set()
        if discarded:
            log.debug("Playback cancelled, %d queued clip(s) dropped", discarded)

    def wait_until_done(self, timeout: float | None = None) -> bool:
        return self._drained.wait(timeout)
