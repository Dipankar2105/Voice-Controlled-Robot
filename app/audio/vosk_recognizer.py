"""Vosk Offline Speech Recognizer for SS_PROJECT_VCR_FRESH.

Features:
- Loads the cryptographically preserved vosk-model-small-en-us-0.15 directory.
- Constrains decoding via a reviewed robotics command grammar to maximize accuracy
  and minimize CPU utilization on Raspberry Pi 4.
- Parses both streaming partial results and finalized utterance transcriptions.
- Keeps recognition decoupled from motor actuation and state management.
- Handles model loading errors with clear exceptions and zero silent downloads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from app.config import AudioConfig, get_project_root
from app.constants import AUDIO_SAMPLE_RATE, REL_PATH_VOSK_MODEL_DIR
from app.hardware.interfaces import SpeechRecognizerInterface
from app.logger import get_logger

logger = get_logger("vosk")


# Authoritative command phrases covering verified movement, safety, and 26 targets
REVIEWED_COMMAND_VOCABULARY = [
    # Safety & Stops
    "stop",
    "halt",
    "wait",
    "emergency stop",
    "cancel",
    "cancel search",
    "stop scanning",
    "resume listening",
    # Directional Movement
    "forward",
    "move forward",
    "go forward",
    "straight",
    "backward",
    "move backward",
    "back",
    "go back",
    "reverse",
    "turn left",
    "turn right",
    "left",
    "right",
    "rotate left",
    "rotate right",
    "move",
    "go",
    # Queries & Actions
    "find",
    "locate",
    "search for",
    "look for",
    "what do you see",
    "status",
    "help",
    # 26 Verified YOLO Target Classes & Verified Aliases
    "person",
    "chair",
    "table",
    "door",
    "red target",
    "blue target",
    "red",
    "blue",
    "laptop",
    "monitor",
    "keyboard",
    "mouse",
    "backpack",
    "bag",
    "seat",
    "notebook",
    "mug",
    "book",
    "bottle",
    "water bottle",
    "trash can",
    "whiteboard",
    "projector",
    "clock",
    "fan",
    "fire extinguisher",
    "bookshelf",
    "cup",
    "cell phone",
    "phone",
    "remote control",
    "remote",
    "potted plant",
    "plant",
    "window",
    "scissors",
    "scissor",
    "shears",
    # Optional Numbers & Modifiers
    "the",
    "a",
    "an",
    "please",
    "speed",
    "slow",
    "medium",
    "normal",
    "fast",
    "target",
    "object",
    "[unk]",
]


def build_grammar_json(vocabulary: Optional[List[str]] = None) -> str:
    """Build Vosk phrase grammar including complete target-command combinations.

    Vosk grammar entries are phrases, not a bag of independent words. Include
    full ``find bottle``-style phrases explicitly; listing ``find`` and ``bottle``
    separately is not sufficient for reliable constrained decoding.
    """
    vocab = list(vocabulary if vocabulary is not None else REVIEWED_COMMAND_VOCABULARY)
    if vocabulary is None:
        targets = [
            "person", "chair", "seat", "table", "door", "red target", "red object", "red",
            "blue target", "blue object", "blue", "laptop", "monitor", "screen", "display",
            "keyboard", "mouse", "backpack", "bag", "pack", "schoolbag", "book", "bottle", "notebook", "mug",
            "water bottle", "trash can", "trash", "trashcan", "garbage can", "bin",
            "whiteboard", "board", "projector", "clock", "fan", "fire extinguisher",
            "extinguisher", "bookshelf", "cup", "cell phone", "phone", "mobile phone",
            "mobile", "remote control", "remote", "potted plant", "plant", "houseplant",
            "window", "scissors", "scissor", "shears", "pair of scissors",
        ]
        prefixes = ("find", "locate", "search for", "look for", "search")
        for prefix in prefixes:
            for target in targets:
                vocab.append(f"{prefix} {target}")
                for article in ("the", "a", "an"):
                    vocab.append(f"{prefix} {article} {target}")
        vocab.extend(("speed slow", "speed medium", "speed normal", "speed fast"))

    tokens = sorted({str(w).lower().strip() for w in vocab if str(w).strip()})
    return json.dumps(tokens)


class VoskSpeechRecognizer(SpeechRecognizerInterface):
    """Authoritative speech recognition engine using offline Vosk Kaldi models."""

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        model_path: Optional[Path | str] = None,
        use_grammar: bool = True,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ):
        self.config = config or AudioConfig()
        self.sample_rate = sample_rate
        self.use_grammar = use_grammar

        root = get_project_root()
        if model_path is None:
            self.model_dir = root / self.config.model_dir
        else:
            p = Path(model_path)
            self.model_dir = p if p.is_absolute() else (root / p)

        self._vosk = None
        self._model = None
        self._recognizer = None
        self._latest_partial: str = ""
        self._latest_final: str = ""

    def load_model(self) -> None:
        """Load offline Kaldi model files from disk. Fails closed if corrupt or missing."""
        if self._model is not None:
            return

        logger.info("Loading offline Vosk speech model from '%s'...", self.model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Offline Vosk model directory missing: {self.model_dir}. "
                "Silent replacement or downloading is strictly prohibited."
            )

        # Verify key Kaldi model files are present
        required_files = ["am/final.mdl", "conf/model.conf", "graph/Gr.fst", "graph/HCLr.fst"]
        for rf in required_files:
            target_f = self.model_dir / rf
            if not target_f.is_file() or target_f.stat().st_size == 0:
                raise FileNotFoundError(f"Vosk model component corrupt or missing: {target_f}")

        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("The 'vosk' package is required for physical/offline speech recognition. Install requirements.txt.") from exc
        self._vosk = vosk
        vosk.SetLogLevel(-1)
        try:
            self._model = vosk.Model(str(self.model_dir))
        except Exception as exc:
            logger.error("Failed to load Vosk Kaldi model: %s", exc)
            raise RuntimeError(f"Vosk Kaldi model initialization failed: {exc}") from exc

        # Initialize KaldiRecognizer
        if self.use_grammar:
            grammar_str = build_grammar_json()
            try:
                grammar_phrase_count = len(json.loads(grammar_str))
            except Exception:
                grammar_phrase_count = len(REVIEWED_COMMAND_VOCABULARY)
            logger.info("Initializing Vosk KaldiRecognizer with reviewed robotics grammar (%d phrases)...", grammar_phrase_count)
            self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate, grammar_str)
        else:
            logger.info("Initializing Vosk KaldiRecognizer in open-vocabulary mode...")
            self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)

        logger.info("Vosk speech recognition engine ready.")

    def accept_waveform(self, pcm_data: bytes) -> bool:
        """Feed a 16-bit mono 16 kHz PCM chunk.
        Returns True if a complete utterance end is detected.
        """
        if self._recognizer is None:
            self.load_model()

        if not pcm_data:
            return False

        is_complete = self._recognizer.AcceptWaveform(pcm_data)

        if is_complete:
            res_json = json.loads(self._recognizer.Result())
            self._latest_final = res_json.get("text", "").strip()
            self._latest_partial = ""
        else:
            part_json = json.loads(self._recognizer.PartialResult())
            self._latest_partial = part_json.get("partial", "").strip()

        return is_complete

    def get_result(self) -> str:
        """Retrieve and clear the latest finalized transcription."""
        text = self._latest_final
        self._latest_final = ""
        return text

    def get_partial_result(self) -> str:
        """Retrieve in-progress partial transcription."""
        return self._latest_partial

    def reset(self) -> None:
        """Reset internal Kaldi decoder lattice and clear transcripts."""
        if self._recognizer:
            try:
                self._recognizer.Reset()
            except Exception:
                pass
        self._latest_partial = ""
        self._latest_final = ""
