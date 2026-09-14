"""The concrete engines: openWakeWord, faster-whisper, Piper, ElevenLabs.

Each is tested through its interface with the underlying library mocked, so the
suite needs no models and no network. The one exception is marked ``hardware``
and runs the real wake word model when it has been downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.audio.resample import from_int16, resample, to_int16
from jarvis.config import Config
from jarvis.errors import AuthError, DependencyMissingError, JarvisError, ModelMissingError
from jarvis.interfaces.audio import AudioClip
from jarvis.stt.faster_whisper_stt import FasterWhisperTranscriber
from jarvis.tts.elevenlabs_tts import ElevenLabsSynthesizer, _decode_pcm
from jarvis.tts.piper_tts import PiperSynthesizer
from jarvis.wake.openwakeword_detector import OpenWakeWordDetector

from .conftest import speech_block

# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


class TestResample:
    def test_matching_rates_are_a_no_op(self) -> None:
        samples = speech_block()
        assert resample(samples, 16_000, 16_000) is samples

    def test_upsampling_lengthens_proportionally(self) -> None:
        out = resample(np.zeros(1600, dtype=np.float32), 16_000, 22_050)
        assert out.size == pytest.approx(2205, abs=2)

    def test_downsampling_shortens_proportionally(self) -> None:
        out = resample(np.zeros(2205, dtype=np.float32), 22_050, 16_000)
        assert out.size == pytest.approx(1600, abs=2)

    def test_output_stays_float32(self) -> None:
        assert resample(np.zeros(100, dtype=np.float32), 16_000, 8_000).dtype == np.float32

    def test_an_empty_array_survives(self) -> None:
        assert resample(np.zeros(0, dtype=np.float32), 16_000, 22_050).size == 0

    def test_a_constant_signal_stays_constant(self) -> None:
        out = resample(np.full(1000, 0.5, dtype=np.float32), 16_000, 22_050)
        assert np.allclose(out, 0.5, atol=1e-6)

    def test_invalid_rates_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            resample(np.zeros(10, dtype=np.float32), 0, 16_000)


class TestPcmConversion:
    def test_round_trip_is_lossless_enough(self) -> None:
        original = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        assert np.allclose(from_int16(to_int16(original)), original, atol=1e-4)

    def test_loud_samples_clip_rather_than_wrap(self) -> None:
        """Wrapping turns a slightly-too-loud sample into a full-scale bang."""
        converted = to_int16(np.array([2.0, -2.0], dtype=np.float32))
        assert converted[0] > 32_000 and converted[1] < -32_000

    def test_int16_output_dtype(self) -> None:
        assert to_int16(np.zeros(10, dtype=np.float32)).dtype == np.int16


# --------------------------------------------------------------------------- #
# Wake word
# --------------------------------------------------------------------------- #


class FakeOpenWakeWordModel:
    """Stands in for openwakeword.model.Model."""

    def __init__(self, scores: list[dict[str, float]] | None = None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._scores = scores or []
        self.calls: list[np.ndarray] = []
        self.resets = 0

    def predict(self, audio: np.ndarray) -> dict[str, float]:
        self.calls.append(audio)
        if self._scores:
            return self._scores.pop(0)
        return {"hey_jarvis": 0.0}

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def wake_models(cfg: Config) -> Config:
    """Zero-byte stand-ins so path resolution succeeds without a download."""
    cfg.wake_word.model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hey_jarvis_v0.1.onnx", "melspectrogram.onnx", "embedding_model.onnx"):
        (cfg.wake_word.model_dir / name).write_bytes(b"")
    return cfg


def install_fake_wake_model(
    monkeypatch: pytest.MonkeyPatch, scores: list[dict[str, float]] | None = None
) -> list[FakeOpenWakeWordModel]:
    created: list[FakeOpenWakeWordModel] = []

    def factory(**kwargs: Any) -> FakeOpenWakeWordModel:
        model = FakeOpenWakeWordModel(scores, **kwargs)
        created.append(model)
        return model

    module = type(sys)("openwakeword.model")
    module.Model = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword.model", module)
    return created


class TestWakeWordLoading:
    def test_a_missing_model_names_the_download_script(self, cfg: Config) -> None:
        detector = OpenWakeWordDetector(cfg.wake_word)
        with pytest.raises(ModelMissingError) as excinfo:
            detector.load()
        assert excinfo.value.remedy == "./scripts/download_models.sh wakeword"

    def test_missing_feature_extractors_are_caught_before_use(self, cfg: Config) -> None:
        """A wake model without them loads fine and then fails on first predict."""
        cfg.wake_word.model_dir.mkdir(parents=True, exist_ok=True)
        (cfg.wake_word.model_dir / "hey_jarvis_v0.1.onnx").write_bytes(b"")
        detector = OpenWakeWordDetector(cfg.wake_word)
        with pytest.raises(ModelMissingError) as excinfo:
            detector.load()
        assert "melspectrogram" in excinfo.value.message

    def test_feature_model_paths_are_passed_explicitly(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The library defaults to .tflite even when asked for ONNX."""
        created = install_fake_wake_model(monkeypatch)
        OpenWakeWordDetector(wake_models.wake_word).load()
        kwargs = created[0].kwargs
        assert kwargs["melspec_model_path"].endswith("melspectrogram.onnx")
        assert kwargs["embedding_model_path"].endswith("embedding_model.onnx")
        assert kwargs["inference_framework"] == "onnx"

    def test_loading_twice_reuses_the_model(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = install_fake_wake_model(monkeypatch)
        detector = OpenWakeWordDetector(wake_models.wake_word)
        detector.load()
        detector.load()
        assert len(created) == 1

    def test_a_missing_package_names_the_install_command(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("openwakeword"):
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "openwakeword.model", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DependencyMissingError):
            OpenWakeWordDetector(wake_models.wake_word).load()


class TestWakeWordDetection:
    def test_a_score_above_the_threshold_fires(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_wake_model(monkeypatch, [{"hey_jarvis": 0.9}])
        detector = OpenWakeWordDetector(wake_models.wake_word)
        event = detector.process(speech_block())
        assert event is not None
        assert event.model == "hey_jarvis"
        assert event.score == pytest.approx(0.9)

    def test_a_score_below_the_threshold_does_not(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_wake_model(monkeypatch, [{"hey_jarvis": 0.2}])
        assert OpenWakeWordDetector(wake_models.wake_word).process(speech_block()) is None

    def test_the_cooldown_suppresses_repeat_triggers(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One spoken "hey Jarvis" must produce exactly one event."""
        install_fake_wake_model(monkeypatch, [{"hey_jarvis": 0.9}] * 10)
        detector = OpenWakeWordDetector(wake_models.wake_word)
        events = [detector.process(speech_block()) for _ in range(10)]
        assert sum(event is not None for event in events) == 1

    def test_audio_keeps_flowing_during_the_cooldown(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skipping predict() would leave a gap in the model's internal buffers."""
        created = install_fake_wake_model(monkeypatch, [{"hey_jarvis": 0.9}] * 5)
        detector = OpenWakeWordDetector(wake_models.wake_word)
        for _ in range(5):
            detector.process(speech_block())
        assert len(created[0].calls) == 5

    def test_audio_is_converted_to_int16_for_the_model(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = install_fake_wake_model(monkeypatch)
        OpenWakeWordDetector(wake_models.wake_word).process(speech_block())
        assert created[0].calls[0].dtype == np.int16

    def test_reset_clears_model_state(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = install_fake_wake_model(monkeypatch)
        detector = OpenWakeWordDetector(wake_models.wake_word)
        detector.process(speech_block())
        detector.reset()
        assert created[0].resets == 1

    def test_reset_survives_a_build_without_one(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Older openWakeWord builds have no reset(); fall back to the cooldown."""

        class NoReset:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            def predict(self, audio: np.ndarray) -> dict[str, float]:
                return {"hey_jarvis": 0.0}

        assert not hasattr(NoReset(), "reset")
        module = type(sys)("openwakeword.model")
        module.Model = lambda **kwargs: NoReset(**kwargs)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openwakeword.model", module)
        detector = OpenWakeWordDetector(wake_models.wake_word)
        detector.process(speech_block())
        detector.reset()  # must not raise

    def test_the_highest_scoring_model_wins(
        self, wake_models: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_wake_model(monkeypatch, [{"alexa": 0.6, "hey_jarvis": 0.95}])
        event = OpenWakeWordDetector(wake_models.wake_word).process(speech_block())
        assert event is not None and event.model == "hey_jarvis"


@pytest.mark.hardware
class TestRealWakeWordModel:
    """Runs the genuine ONNX model. Skipped unless it has been downloaded."""

    def test_noise_does_not_trigger_the_wake_word(self, cfg: Config) -> None:
        repo_models = Path(__file__).resolve().parent.parent / "models" / "openwakeword"
        if not (repo_models / "hey_jarvis_v0.1.onnx").is_file():
            pytest.skip("wake word model not downloaded (./scripts/download_models.sh wakeword)")

        cfg.wake_word.model_dir = repo_models
        detector = OpenWakeWordDetector(cfg.wake_word)
        detector.load()

        rng = np.random.default_rng(0)
        detections = sum(
            detector.process((rng.standard_normal(1280) * 0.01).astype(np.float32)) is not None
            for _ in range(60)
        )
        assert detections == 0


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


class FakeSegment:
    def __init__(self, text: str, avg_logprob: float = -0.2) -> None:
        self.text = text
        self.avg_logprob = avg_logprob


class FakeWhisperModel:
    def __init__(self, segments: list[FakeSegment], **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._segments = segments
        self.transcribe_kwargs: dict[str, Any] = {}

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Any]:
        self.transcribe_kwargs = kwargs
        self.audio = audio
        return iter(self._segments), type("Info", (), {"language": "en"})()


def install_fake_whisper(
    monkeypatch: pytest.MonkeyPatch,
    segments: list[FakeSegment],
    *,
    fail: Exception | None = None,
) -> list[FakeWhisperModel]:
    created: list[FakeWhisperModel] = []

    def factory(*args: Any, **kwargs: Any) -> FakeWhisperModel:
        if fail is not None:
            raise fail
        model = FakeWhisperModel(segments, **kwargs)
        created.append(model)
        return model

    module = type(sys)("faster_whisper")
    module.WhisperModel = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return created


def clip(seconds: float = 1.0, rate: int = 16_000) -> AudioClip:
    return AudioClip(np.zeros(int(seconds * rate), dtype=np.float32), rate)


class TestTranscription:
    def test_segments_are_joined(self, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_whisper(monkeypatch, [FakeSegment(" What time"), FakeSegment(" is it?")])
        result = FasterWhisperTranscriber(cfg.stt).transcribe(clip())
        assert result.text == "What time is it?"

    def test_confidence_is_averaged_across_segments(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_whisper(monkeypatch, [FakeSegment("a", -0.2), FakeSegment("b", -0.4)])
        result = FasterWhisperTranscriber(cfg.stt).transcribe(clip())
        assert result.confidence == pytest.approx(-0.3)

    def test_an_empty_clip_short_circuits(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_whisper(monkeypatch, [FakeSegment("should not be reached")])
        result = FasterWhisperTranscriber(cfg.stt).transcribe(clip(seconds=0))
        assert result.is_empty

    def test_audio_is_resampled_to_16k(self, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        created = install_fake_whisper(monkeypatch, [FakeSegment("hello")])
        FasterWhisperTranscriber(cfg.stt).transcribe(clip(seconds=1.0, rate=48_000))
        assert created[0].audio.size == pytest.approx(16_000, abs=2)

    def test_previous_text_does_not_condition_the_next_utterance(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise Whisper repeats the last command when this one is unclear."""
        created = install_fake_whisper(monkeypatch, [FakeSegment("hello")])
        FasterWhisperTranscriber(cfg.stt).transcribe(clip())
        assert created[0].transcribe_kwargs["condition_on_previous_text"] is False

    def test_it_never_reaches_the_network_mid_conversation(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = install_fake_whisper(monkeypatch, [FakeSegment("hello")])
        FasterWhisperTranscriber(cfg.stt).transcribe(clip())
        assert created[0].kwargs["local_files_only"] is True

    def test_a_missing_model_names_the_download_script(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_whisper(monkeypatch, [], fail=RuntimeError("model not found"))
        with pytest.raises(ModelMissingError) as excinfo:
            FasterWhisperTranscriber(cfg.stt).load()
        assert excinfo.value.remedy == "./scripts/download_models.sh whisper"

    def test_a_missing_package_names_the_install_command(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "faster_whisper":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DependencyMissingError):
            FasterWhisperTranscriber(cfg.stt).load()


class TestHallucinationFiltering:
    """Whisper invents subtitle boilerplate out of silence. Drop it."""

    @pytest.mark.parametrize("text", ["Thank you.", "Thanks for watching!", "you", "."])
    def test_known_boilerplate_is_discarded(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch, text: str
    ) -> None:
        install_fake_whisper(monkeypatch, [FakeSegment(text)])
        assert FasterWhisperTranscriber(cfg.stt).transcribe(clip()).is_empty

    def test_very_low_confidence_output_is_discarded(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_whisper(monkeypatch, [FakeSegment("mumble mumble", avg_logprob=-3.0)])
        assert FasterWhisperTranscriber(cfg.stt).transcribe(clip()).is_empty

    def test_a_real_command_survives(self, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_whisper(monkeypatch, [FakeSegment("set a timer for ten minutes")])
        result = FasterWhisperTranscriber(cfg.stt).transcribe(clip())
        assert result.text == "set a timer for ten minutes"

    def test_thank_you_in_a_longer_sentence_is_kept(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the bare boilerplate is dropped, not any sentence containing it."""
        install_fake_whisper(monkeypatch, [FakeSegment("Thank you, that will be all.")])
        assert not FasterWhisperTranscriber(cfg.stt).transcribe(clip()).is_empty


# --------------------------------------------------------------------------- #
# Piper
# --------------------------------------------------------------------------- #


class FakePiperChunk:
    def __init__(self, samples: np.ndarray, sample_rate: int = 22_050) -> None:
        self.audio_float_array = samples
        self.sample_rate = sample_rate


class FakePiperVoice:
    def __init__(self, chunks: list[FakePiperChunk]) -> None:
        self._chunks = chunks
        self.config = type("Config", (), {"sample_rate": 22_050})()
        self.calls: list[str] = []

    def synthesize(self, text: str, syn_config: Any = None) -> list[FakePiperChunk]:
        self.calls.append(text)
        self.syn_config = syn_config
        return self._chunks


def install_fake_piper(
    monkeypatch: pytest.MonkeyPatch, chunks: list[FakePiperChunk]
) -> list[FakePiperVoice]:
    created: list[FakePiperVoice] = []

    def load(model_path: Any, config_path: Any = None, **kwargs: Any) -> FakePiperVoice:
        voice = FakePiperVoice(chunks)
        created.append(voice)
        return voice

    module = type(sys)("piper")
    module.PiperVoice = type("PiperVoice", (), {"load": staticmethod(load)})  # type: ignore[attr-defined]
    module.SynthesisConfig = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", module)
    return created


@pytest.fixture
def piper_voice(cfg: Config) -> Config:
    cfg.tts.piper.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.tts.piper.model_path.write_bytes(b"")
    cfg.tts.piper.config_path.write_text("{}")
    return cfg


class TestPiper:
    def test_it_reports_itself_as_fully_offline(self, cfg: Config) -> None:
        assert PiperSynthesizer(cfg.tts.piper).sends_text_off_machine is False

    def test_a_missing_voice_names_the_download_script(self, cfg: Config) -> None:
        with pytest.raises(ModelMissingError) as excinfo:
            PiperSynthesizer(cfg.tts.piper).load()
        assert excinfo.value.remedy == "./scripts/download_models.sh piper"

    def test_a_voice_without_its_sidecar_is_rejected(self, cfg: Config) -> None:
        cfg.tts.piper.model_dir.mkdir(parents=True, exist_ok=True)
        cfg.tts.piper.model_path.write_bytes(b"")
        with pytest.raises(ModelMissingError) as excinfo:
            PiperSynthesizer(cfg.tts.piper).load()
        assert ".onnx.json" in excinfo.value.message

    def test_synthesis_returns_a_clip(
        self, piper_voice: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_piper(monkeypatch, [FakePiperChunk(np.full(100, 0.5, dtype=np.float32))])
        result = PiperSynthesizer(piper_voice.tts.piper).synthesize("Good evening.")
        assert result.sample_rate == 22_050
        assert result.samples.size == 100

    def test_sentence_chunks_are_separated_by_the_configured_gap(
        self, piper_voice: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        piper_voice.tts.piper.sentence_silence = 0.1
        install_fake_piper(
            monkeypatch,
            [
                FakePiperChunk(np.full(100, 0.5, dtype=np.float32)),
                FakePiperChunk(np.full(100, 0.5, dtype=np.float32)),
            ],
        )
        result = PiperSynthesizer(piper_voice.tts.piper).synthesize("One. Two.")
        # 200 samples of speech plus 0.1 s of silence at 22.05 kHz.
        assert result.samples.size == 200 + int(0.1 * 22_050)

    def test_the_configured_pace_is_passed_through(
        self, piper_voice: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = install_fake_piper(monkeypatch, [FakePiperChunk(np.zeros(10, dtype=np.float32))])
        PiperSynthesizer(piper_voice.tts.piper).synthesize("Hello.")
        assert created[0].syn_config["length_scale"] == piper_voice.tts.piper.length_scale

    def test_blank_text_produces_no_audio(
        self, piper_voice: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        install_fake_piper(monkeypatch, [FakePiperChunk(np.zeros(100, dtype=np.float32))])
        assert PiperSynthesizer(piper_voice.tts.piper).synthesize("   ").samples.size == 0

    def test_a_missing_package_names_the_install_command(
        self, piper_voice: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "piper":
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "piper", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DependencyMissingError):
            PiperSynthesizer(piper_voice.tts.piper).load()


# --------------------------------------------------------------------------- #
# ElevenLabs
# --------------------------------------------------------------------------- #


class TestElevenLabs:
    def test_it_declares_that_text_leaves_the_machine(self, cfg: Config) -> None:
        """The privacy difference from Piper must be visible through the interface."""
        synth = ElevenLabsSynthesizer(cfg.tts.elevenlabs, "key")
        assert synth.sends_text_off_machine is True

    def test_no_key_is_an_auth_error_that_offers_piper(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The package is present in this scenario; only the key is missing.
        module = type(sys)("elevenlabs.client")
        module.ElevenLabs = lambda **kwargs: object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "elevenlabs.client", module)

        cfg.tts.elevenlabs.voice_id = "abc"
        with pytest.raises(AuthError) as excinfo:
            ElevenLabsSynthesizer(cfg.tts.elevenlabs, None).load()
        assert excinfo.value.remedy is not None
        assert "piper" in excinfo.value.remedy

    def test_no_voice_id_is_reported(self, cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
        module = type(sys)("elevenlabs.client")
        module.ElevenLabs = lambda **kwargs: object()  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "elevenlabs.client", module)
        with pytest.raises(JarvisError, match="voice_id"):
            ElevenLabsSynthesizer(cfg.tts.elevenlabs, "key").load()

    def test_a_missing_package_offers_piper_as_the_alternative(
        self, cfg: Config, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("elevenlabs"):
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "elevenlabs.client", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(DependencyMissingError) as excinfo:
            ElevenLabsSynthesizer(cfg.tts.elevenlabs, "key").load()
        assert excinfo.value.remedy is not None
        assert "piper" in excinfo.value.remedy


class TestPcmDecoding:
    def test_raw_little_endian_int16(self) -> None:
        payload = np.array([0, 16384, -16384], dtype="<i2").tobytes()
        decoded = _decode_pcm(payload)
        assert np.allclose(decoded, [0.0, 0.5, -0.5], atol=1e-3)

    def test_an_odd_trailing_byte_is_dropped(self) -> None:
        """A truncated response should lose a sample, not raise."""
        payload = np.array([0, 16384], dtype="<i2").tobytes() + b"\x01"
        assert _decode_pcm(payload).size == 2

    def test_a_wav_header_is_handled(self) -> None:
        """Asking for PCM usually gives raw bytes, but sometimes a RIFF header."""
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(22_050)
            handle.writeframes(np.array([0, 16384], dtype="<i2").tobytes())
        decoded = _decode_pcm(buffer.getvalue())
        assert decoded.size == 2
        assert np.allclose(decoded, [0.0, 0.5], atol=1e-3)
