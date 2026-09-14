"""The sounddevice backend.

PortAudio is absent here (as it is in CI), so a fake module is injected and the
audio callbacks are driven by hand. That means the queueing, the drop-oldest
policy, the mixing in the playback callback, and the error translation are all
really executed.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from jarvis.audio.sounddevice_io import (
    SoundDeviceInput,
    SoundDeviceOutput,
    resolve_device,
)
from jarvis.errors import AudioDeviceError, DependencyMissingError
from jarvis.interfaces.audio import AudioClip

from . import fake_sounddevice


@pytest.fixture
def sd(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)
    return fake_sounddevice


@pytest.fixture
def no_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the package not being installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sounddevice":
            raise ImportError("No module named 'sounddevice'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.fixture
def broken_portaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the package being installed but the native library missing."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestErrorTranslation:
    def test_missing_package_names_the_install_command(self, no_sounddevice: None) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1280)
        with pytest.raises(DependencyMissingError) as excinfo:
            mic.start()
        assert excinfo.value.remedy == "uv sync --extra voice"

    def test_missing_native_library_names_brew(self, broken_portaudio: None) -> None:
        """The classic macOS failure: pip succeeded, PortAudio was never installed."""
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1280)
        with pytest.raises(AudioDeviceError) as excinfo:
            mic.start()
        assert excinfo.value.remedy is not None
        assert "brew install portaudio" in excinfo.value.remedy

    def test_a_device_that_refuses_to_open_explains_itself(
        self, sd: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(**kwargs: Any) -> Any:
            raise sd.PortAudioError("Device unavailable")

        monkeypatch.setattr(sd, "InputStream", explode)
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1280)
        with pytest.raises(AudioDeviceError) as excinfo:
            mic.start()
        assert excinfo.value.remedy is not None
        assert "Privacy" in excinfo.value.remedy


class TestDeviceResolution:
    def test_none_defers_to_portaudio(self, sd: Any) -> None:
        assert resolve_device(None, want_input=True) is None

    def test_index_is_validated_against_direction(self, sd: Any) -> None:
        assert resolve_device(2, want_input=True) == 2
        with pytest.raises(AudioDeviceError):
            # Device 1 is output-only, so it is not a valid microphone.
            resolve_device(1, want_input=True)

    def test_name_substring_is_case_insensitive(self, sd: Any) -> None:
        assert resolve_device("jabra", want_input=True) == 2
        assert resolve_device("MacBook Pro Speakers", want_input=False) == 1

    def test_an_unmatched_name_points_at_the_devices_command(self, sd: Any) -> None:
        with pytest.raises(AudioDeviceError) as excinfo:
            resolve_device("aardvark", want_input=True)
        assert excinfo.value.remedy is not None
        assert "jarvis devices" in excinfo.value.remedy

    def test_out_of_range_index_is_rejected(self, sd: Any) -> None:
        with pytest.raises(AudioDeviceError):
            resolve_device(99, want_input=True)


class TestCapture:
    def test_delivered_audio_comes_back_as_mono_float32(self, sd: Any) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=4)
        mic.start()
        stream = mic._stream
        assert isinstance(stream, fake_sounddevice.InputStream)
        stream.deliver(np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32))

        block = next(mic.blocks())
        assert block.dtype == np.float32
        assert block.shape == (4,)
        assert np.allclose(block, [0.1, -0.2, 0.3, -0.4])
        mic.stop()

    def test_the_stream_is_opened_with_the_configured_settings(self, sd: Any) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1280, device="jabra")
        mic.start()
        stream = mic._stream
        assert stream.kwargs["samplerate"] == 16_000
        assert stream.kwargs["blocksize"] == 1280
        assert stream.kwargs["channels"] == 1
        assert stream.kwargs["dtype"] == "float32"
        assert stream.kwargs["device"] == 2
        mic.stop()

    def test_a_full_buffer_drops_the_oldest_audio(self, sd: Any) -> None:
        """Stale audio is worse than no audio: never grow without limit."""
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1, max_buffered_blocks=3)
        mic.start()
        stream = mic._stream
        for value in range(10):
            stream.deliver(np.array([float(value)], dtype=np.float32))

        assert mic.dropped_blocks == 7
        kept = [next(mic.blocks())[0] for _ in range(3)]
        assert kept == [7.0, 8.0, 9.0]
        mic.stop()

    def test_flush_discards_everything_buffered(self, sd: Any) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1)
        mic.start()
        stream = mic._stream
        for value in range(5):
            stream.deliver(np.array([float(value)], dtype=np.float32))
        assert mic.flush() == 5
        assert mic.flush() == 0
        mic.stop()

    def test_stop_is_idempotent(self, sd: Any) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1280)
        mic.start()
        mic.stop()
        mic.stop()

    def test_starting_twice_reuses_the_open_stream(self, sd: Any) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1280)
        mic.start()
        first = mic._stream
        mic.start()
        assert mic._stream is first
        mic.stop()

    def test_blocks_stops_yielding_after_stop(self, sd: Any) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1)
        mic.start()
        mic.stop()
        assert list(mic.blocks()) == []

    def test_it_works_as_a_context_manager(self, sd: Any) -> None:
        mic = SoundDeviceInput(sample_rate=16_000, block_size=1280)
        with mic:
            assert mic._stream is not None
        assert mic._stream is None


class TestPlayback:
    def test_queued_audio_reaches_the_callback(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 22_050))

        played = speaker._stream.pull(4)
        assert np.allclose(played, [1.0, 2.0, 3.0, 4.0])
        speaker.stop()

    def test_a_clip_spanning_several_callbacks_is_not_torn(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.arange(10, dtype=np.float32), 22_050))

        collected = np.concatenate([speaker._stream.pull(4) for _ in range(3)])
        assert np.allclose(collected[:10], np.arange(10))
        assert np.allclose(collected[10:], 0.0)  # padded with silence, not garbage
        speaker.stop()

    def test_consecutive_clips_play_back_to_back(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.array([1.0, 2.0], dtype=np.float32), 22_050))
        speaker.play(AudioClip(np.array([3.0, 4.0], dtype=np.float32), 22_050))
        assert np.allclose(speaker._stream.pull(4), [1.0, 2.0, 3.0, 4.0])
        speaker.stop()

    def test_silence_is_output_when_nothing_is_queued(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        assert np.allclose(speaker._stream.pull(4), 0.0)
        speaker.stop()

    def test_a_clip_at_another_rate_is_resampled_on_the_way_in(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=64)
        speaker.start()
        speaker.play(AudioClip(np.zeros(160, dtype=np.float32), 16_000))
        # 160 samples at 16 kHz is 10 ms, which is 220 samples at 22.05 kHz.
        assert speaker._pending[0].size == pytest.approx(220, abs=2)
        speaker.stop()

    def test_an_empty_clip_is_ignored(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050)
        speaker.start()
        speaker.play(AudioClip(np.zeros(0, dtype=np.float32), 22_050))
        assert not speaker.is_playing
        speaker.stop()


class TestCancellation:
    """Barge-in depends on this: cancel must take effect within one callback."""

    def test_cancel_discards_queued_audio_immediately(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.arange(1000, dtype=np.float32), 22_050))
        playing_before = speaker.is_playing

        speaker.cancel()

        assert playing_before
        assert not speaker.is_playing
        assert np.allclose(speaker._stream.pull(4), 0.0)
        speaker.stop()

    def test_cancel_mid_clip_does_not_resume_later(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.arange(100, dtype=np.float32), 22_050))
        speaker._stream.pull(4)
        speaker.cancel()
        speaker.play(AudioClip(np.array([9.0, 9.0], dtype=np.float32), 22_050))
        # The new clip starts at its own beginning, not part-way through the old one.
        assert np.allclose(speaker._stream.pull(4), [9.0, 9.0, 0.0, 0.0])
        speaker.stop()

    def test_cancel_releases_a_waiter(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.zeros(100_000, dtype=np.float32), 22_050))
        speaker.cancel()
        assert speaker.wait_until_done(timeout=0.1)
        speaker.stop()

    def test_cancel_on_an_idle_speaker_is_harmless(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050)
        speaker.start()
        speaker.cancel()
        speaker.cancel()
        speaker.stop()


class TestWaiting:
    def test_wait_returns_true_once_drained(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.arange(4, dtype=np.float32), 22_050))
        assert not speaker.wait_until_done(timeout=0.01)
        speaker._stream.pull(4)
        assert speaker.wait_until_done(timeout=0.5)
        speaker.stop()

    def test_wait_times_out_while_audio_remains(self, sd: Any) -> None:
        speaker = SoundDeviceOutput(sample_rate=22_050, block_size=4)
        speaker.start()
        speaker.play(AudioClip(np.zeros(100_000, dtype=np.float32), 22_050))
        assert not speaker.wait_until_done(timeout=0.01)
        speaker.stop()
