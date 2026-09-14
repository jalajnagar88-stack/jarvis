"""Building subsystems from configuration.

The one place that knows which concrete class implements which interface. The
loop depends only on the interfaces; the CLI calls these functions. Swapping an
implementation means editing one function here.

Every builder is cheap: it constructs the object without loading a model. Call
``load()`` on the result when you actually need it, or :func:`preload_all` to
pay all the loading costs up front with a progress message.
"""

from __future__ import annotations

from jarvis.audio.sounddevice_io import SoundDeviceInput, SoundDeviceOutput
from jarvis.config import Config
from jarvis.interfaces.audio import AudioInput, AudioOutput
from jarvis.interfaces.stt import Transcriber
from jarvis.interfaces.tts import SpeechSynthesizer
from jarvis.interfaces.wake_word import WakeWordDetector
from jarvis.logging_setup import get_logger
from jarvis.stt.faster_whisper_stt import FasterWhisperTranscriber
from jarvis.tts.elevenlabs_tts import ElevenLabsSynthesizer
from jarvis.tts.piper_tts import PiperSynthesizer
from jarvis.wake.openwakeword_detector import OpenWakeWordDetector

log = get_logger("factory")


def build_audio_input(cfg: Config) -> AudioInput:
    return SoundDeviceInput(
        sample_rate=cfg.audio.sample_rate,
        block_size=cfg.audio.block_size,
        device=cfg.audio.input_device,
    )


def build_audio_output(cfg: Config, *, sample_rate: int) -> AudioOutput:
    """Build the speaker.

    ``sample_rate`` is the synthesiser's native rate. Opening the stream at that
    rate means the common path never resamples.
    """
    return SoundDeviceOutput(sample_rate=sample_rate, device=cfg.audio.output_device)


def build_wake_word(cfg: Config) -> WakeWordDetector:
    return OpenWakeWordDetector(cfg.wake_word)


def build_transcriber(cfg: Config) -> Transcriber:
    return FasterWhisperTranscriber(cfg.stt)


def build_synthesizer(cfg: Config) -> SpeechSynthesizer:
    if cfg.tts.engine == "elevenlabs":
        return ElevenLabsSynthesizer(cfg.tts.elevenlabs, cfg.secrets.elevenlabs_api_key)
    return PiperSynthesizer(cfg.tts.piper)
