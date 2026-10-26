"""Audio subsystem exports; Vosk native bindings are loaded only when a model is initialized."""
from app.audio.audio_source import MicrophoneAudioSource, convert_and_resample_pcm, find_logitech_c270_device_index
from app.audio.vosk_recognizer import REVIEWED_COMMAND_VOCABULARY, VoskSpeechRecognizer, build_grammar_json

__all__ = ["MicrophoneAudioSource", "convert_and_resample_pcm", "find_logitech_c270_device_index", "VoskSpeechRecognizer", "REVIEWED_COMMAND_VOCABULARY", "build_grammar_json"]
