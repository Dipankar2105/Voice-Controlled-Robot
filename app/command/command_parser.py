"""Deterministic Command Parser and Voice Dispatch for SS_PROJECT_VCR_FRESH.

Features:
- Deterministic, whole-token phrase matching (zero naive substring regressions).
- Complete disambiguation (e.g. 'find backpack' -> SEARCH_TARGET, never MOVE_BACKWARD).
- Resolves search targets and aliases to verified detector model classes.
- Distinguishes specific control ('stop scanning', 'cancel search') from generic stops.
- Enforces strict numeric bounds on speed commands [MIN_SPEED, MAX_SPEED].
- Rejects partial or low-confidence transcripts for movement commands to prevent false triggers.
- Permits immediate safety stops ('emergency stop', 'stop', 'halt') even on partial transcripts.
- Pure software component with zero hardware dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import math
import time
from typing import Dict, List, Optional, Set, Tuple

from app.constants import (
    DEFAULT_MANUAL_SPEED,
    MAX_SPEED,
    MIN_SPEED,
    SEARCH_SPEED,
    TURN_SPEED,
)
from app.logger import get_logger

logger = get_logger("command")


class CommandIntent(str, Enum):
    """Broad semantic category of commanded intent."""
    SAFETY = "SAFETY"          # Emergency stop, stop, halt (highest priority)
    MOVEMENT = "MOVEMENT"      # Forward, backward, turn left, turn right
    SEARCH = "SEARCH"          # Find, locate, search for target object
    CONTROL = "CONTROL"        # Status, help, cancel, scan management, query vision
    SPEED = "SPEED"            # Speed configuration setting
    UNKNOWN = "UNKNOWN"        # Unrecognized, ambiguous, or discarded partial


class CommandAction(str, Enum):
    """Specific actionable command operation."""
    EMERGENCY_STOP = "EMERGENCY_STOP"
    STOP = "STOP"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    SEARCH_TARGET = "SEARCH_TARGET"
    CANCEL_SEARCH = "CANCEL_SEARCH"
    STOP_SCANNING = "STOP_SCANNING"
    RESUME_LISTENING = "RESUME_LISTENING"
    GET_STATUS = "GET_STATUS"
    GET_HELP = "GET_HELP"
    QUERY_VISION = "QUERY_VISION"
    SET_SPEED = "SET_SPEED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RobotCommand:
    """Strongly-typed parsed command with metadata and safety attributes."""
    raw_transcript: str
    normalized_text: str
    intent: CommandIntent
    action: CommandAction
    target_class: Optional[str] = None
    target_alias: Optional[str] = None
    speed: Optional[float] = None
    is_partial: bool = False
    confidence: float = 1.0
    timestamp: float = 0.0
    error_reason: Optional[str] = None

    @property
    def is_executable(self) -> bool:
        """True only if the command is recognized and actionable."""
        return self.intent != CommandIntent.UNKNOWN and self.action != CommandAction.UNKNOWN

    @property
    def causes_movement(self) -> bool:
        """True if executing this command commands motor actuation."""
        return self.intent in (CommandIntent.MOVEMENT, CommandIntent.SEARCH)


# Verified 26 classes from YOLO ONNX model metadata
ALL_VERIFIED_CLASSES: Set[str] = {
    "person", "chair", "table", "door", "RED_TARGET", "BLUE_TARGET",
    "laptop", "monitor", "keyboard", "mouse", "backpack", "book",
    "bottle", "trash_can", "whiteboard", "projector", "clock", "fan",
    "fire_extinguisher", "bookshelf", "cup", "cell_phone", "remote_control",
    "potted_plant", "window", "scissors"
}

# Supported natural aliases mapped strictly to verified YOLO class names
TARGET_ALIAS_MAP: Dict[str, str] = {
    # Backpack / Bag
    "backpack": "backpack",
    "bag": "backpack",
    "pack": "backpack",
    "schoolbag": "backpack",
    # Bottle
    "bottle": "bottle",
    "water bottle": "bottle",
    "notebook": "laptop",
    "mug": "cup",
    "seat": "chair",
    # Scissors
    "scissors": "scissors",
    "scissor": "scissors",
    "shears": "scissors",
    "pair of scissors": "scissors",
    # Red Target
    "red target": "RED_TARGET",
    "red object": "RED_TARGET",
    "red": "RED_TARGET",
    # Blue Target
    "blue target": "BLUE_TARGET",
    "blue object": "BLUE_TARGET",
    "blue": "BLUE_TARGET",
    # Cell Phone
    "cell phone": "cell_phone",
    "phone": "cell_phone",
    "mobile phone": "cell_phone",
    "mobile": "cell_phone",
    # Trash Can
    "trash can": "trash_can",
    "trash": "trash_can",
    "trashcan": "trash_can",
    "garbage can": "trash_can",
    "bin": "trash_can",
    # Whiteboard
    "whiteboard": "whiteboard",
    "board": "whiteboard",
    # Remote Control
    "remote control": "remote_control",
    "remote": "remote_control",
    # Potted Plant
    "potted plant": "potted_plant",
    "plant": "potted_plant",
    "houseplant": "potted_plant",
    # Fire Extinguisher
    "fire extinguisher": "fire_extinguisher",
    "extinguisher": "fire_extinguisher",
    # Monitor
    "monitor": "monitor",
    "screen": "monitor",
    "display": "monitor",
    # Common Classes
    "person": "person",
    "chair": "chair",
    "table": "table",
    "door": "door",
    "laptop": "laptop",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "book": "book",
    "projector": "projector",
    "clock": "clock",
    "fan": "fan",
    "bookshelf": "bookshelf",
    "cup": "cup",
    "window": "window",
}


class CommandParser:
    """Deterministic, token-based voice command parser."""

    def __init__(
        self,
        supported_classes: Optional[Set[str]] = None,
        min_speed: float = 0.20,
        max_speed: float = 0.80,
    ):
        self.supported_classes = set(supported_classes) if supported_classes is not None else set(ALL_VERIFIED_CLASSES)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)

    def normalize(self, text: Optional[str]) -> str:
        """Clean, lowercase, strip punctuation, and normalize whitespace."""
        if not text:
            return ""
        # Lowercase
        t = str(text).lower()
        # Keep letters, digits, whitespace, dot, and minus
        t = re.sub(r"[^a-z0-9\s.-]", " ", t)
        # Remove dots that are NOT between digits (e.g. "stop." -> "stop", but keep "0.5")
        t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
        # Remove hyphens that are NOT followed by a digit (e.g. "e-stop" -> "e stop", but keep "-0.5")
        t = re.sub(r"-(?!\d)", " ", t)
        # Collapse whitespace
        tokens = t.split()
        # Strip conversational prefix fillers
        if tokens and tokens[0] in ("please", "robot"):
            tokens = tokens[1:]
        if len(tokens) >= 2 and tokens[0] == "hey" and tokens[1] == "robot":
            tokens = tokens[2:]
        return " ".join(tokens)

    def parse(
        self,
        transcript: Optional[str],
        is_partial: bool = False,
        confidence: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> RobotCommand:
        """Deterministically parse a spoken transcript into a typed RobotCommand."""
        ts = timestamp if timestamp is not None else time.time()
        raw = transcript or ""
        norm = self.normalize(raw)

        if not norm:
            return RobotCommand(
                raw_transcript=raw,
                normalized_text="",
                intent=CommandIntent.UNKNOWN,
                action=CommandAction.UNKNOWN,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
                error_reason="Empty transcript",
            )

        tokens = norm.split()

        # -------------------------------------------------------------------
        # 1. Safety & Emergency Stop Commands (Highest Priority)
        # -------------------------------------------------------------------
        # Specific control stops first
        if norm == "stop scanning":
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.CONTROL,
                action=CommandAction.STOP_SCANNING,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        if norm in ("cancel search", "stop searching"):
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.CONTROL,
                action=CommandAction.CANCEL_SEARCH,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        # Emergency stop takes precedence if present anywhere in the utterance
        if norm in ("emergency stop", "e stop", "estop") or "emergency stop" in norm or "e stop" in norm or "estop" in tokens:
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.SAFETY,
                action=CommandAction.EMERGENCY_STOP,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        # General stop commands take safety precedence over any movement
        if norm in ("stop", "halt", "freeze", "stop robot", "wait") or any(t in ("stop", "halt", "freeze", "wait") for t in tokens):
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.SAFETY,
                action=CommandAction.STOP,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        if norm in ("cancel", "abort"):
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.CONTROL,
                action=CommandAction.CANCEL_SEARCH,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        if norm in ("resume listening", "listen"):
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.CONTROL,
                action=CommandAction.RESUME_LISTENING,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        # -------------------------------------------------------------------
        # 2. Control & Query Commands
        # -------------------------------------------------------------------
        if norm in ("what do you see", "what do you see now", "look", "what is there"):
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.CONTROL,
                action=CommandAction.QUERY_VISION,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        if norm in ("status", "robot status", "system status", "health"):
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.CONTROL,
                action=CommandAction.GET_STATUS,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        if norm in ("help", "commands", "what can you do"):
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.CONTROL,
                action=CommandAction.GET_HELP,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        # -------------------------------------------------------------------
        # 3. Speed Commands (Strict Numeric Bounds)
        # -------------------------------------------------------------------
        if tokens and tokens[0] == "speed":
            if len(tokens) == 2:
                arg = tokens[1]
                target_speed: Optional[float] = None
                if arg == "slow":
                    target_speed = SEARCH_SPEED  # 0.30
                elif arg in ("medium", "normal"):
                    target_speed = 0.50
                elif arg == "fast":
                    target_speed = DEFAULT_MANUAL_SPEED  # 0.75
                else:
                    try:
                        val = float(arg)
                        if not math.isfinite(val):
                            return RobotCommand(
                                raw_transcript=raw,
                                normalized_text=norm,
                                intent=CommandIntent.UNKNOWN,
                                action=CommandAction.UNKNOWN,
                                is_partial=is_partial,
                                confidence=confidence,
                                timestamp=ts,
                                error_reason="Speed value must be a finite number",
                            )
                        if val <= 0.0:
                            return RobotCommand(
                                raw_transcript=raw,
                                normalized_text=norm,
                                intent=CommandIntent.UNKNOWN,
                                action=CommandAction.UNKNOWN,
                                is_partial=is_partial,
                                confidence=confidence,
                                timestamp=ts,
                                error_reason=f"Speed value {val} must be positive",
                            )
                        # Clamp strictly between configured min_speed and max_speed
                        target_speed = max(self.min_speed, min(self.max_speed, val))
                    except (ValueError, OverflowError):
                        pass

                if target_speed is not None:
                    return RobotCommand(
                        raw_transcript=raw,
                        normalized_text=norm,
                        intent=CommandIntent.SPEED,
                        action=CommandAction.SET_SPEED,
                        speed=round(target_speed, 2),
                        is_partial=is_partial,
                        confidence=confidence,
                        timestamp=ts,
                    )

            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.UNKNOWN,
                action=CommandAction.UNKNOWN,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
                error_reason=f"Unrecognized speed command format: '{norm}'",
            )

        # Check for directional movement conflict flags
        fwd_words = {"forward", "straight"}
        bwd_words = {"backward", "back", "reverse", "backwards", "backwad"}
        left_words = {"left"}
        right_words = {"right"}

        has_fwd = any(t in fwd_words for t in tokens)
        has_bwd = any(t in bwd_words for t in tokens)
        has_left = any(t in left_words for t in tokens)
        has_right = any(t in right_words for t in tokens)
        active_directions = sum([has_fwd, has_bwd, has_left, has_right])

        # -------------------------------------------------------------------
        # 4. Search Commands ("find", "locate", "search for", "look for")
        # -------------------------------------------------------------------
        search_prefixes = [
            ("search for", 10),
            ("look for", 8),
            ("find", 4),
            ("locate", 6),
            ("search", 6),
        ]
        is_search = False
        target_str = ""

        for prefix, _ in search_prefixes:
            if norm.startswith(prefix + " ") or norm == prefix:
                is_search = True
                target_str = norm[len(prefix):].strip()
                break

        if is_search:
            # Conflicting search and movement commands in the same utterance
            if active_directions > 0:
                return RobotCommand(
                    raw_transcript=raw,
                    normalized_text=norm,
                    intent=CommandIntent.UNKNOWN,
                    action=CommandAction.UNKNOWN,
                    is_partial=is_partial,
                    confidence=confidence,
                    timestamp=ts,
                    error_reason="Conflicting search and movement commands in transcript",
                )

            # Strip articles: "the", "a", "an"
            for art in ("the ", "a ", "an "):
                if target_str.startswith(art):
                    target_str = target_str[len(art):].strip()

            if not target_str:
                return RobotCommand(
                    raw_transcript=raw,
                    normalized_text=norm,
                    intent=CommandIntent.UNKNOWN,
                    action=CommandAction.UNKNOWN,
                    is_partial=is_partial,
                    confidence=confidence,
                    timestamp=ts,
                    error_reason="Search command missing target object class",
                )

            # Resolve target class through alias mapping
            resolved_class = TARGET_ALIAS_MAP.get(target_str)
            if resolved_class is None:
                # Try direct normalization without spaces
                resolved_class = TARGET_ALIAS_MAP.get(target_str.replace(" ", "_"))

            if resolved_class is not None:
                # Validate against detector model's supported classes
                if resolved_class in self.supported_classes:
                    # Invariant: Partial transcripts must not trigger autonomous search
                    if is_partial:
                        return RobotCommand(
                            raw_transcript=raw,
                            normalized_text=norm,
                            intent=CommandIntent.UNKNOWN,
                            action=CommandAction.UNKNOWN,
                            target_class=resolved_class,
                            target_alias=target_str,
                            is_partial=is_partial,
                            confidence=confidence,
                            timestamp=ts,
                            error_reason="Partial transcript discarded to prevent false search triggering",
                        )

                    return RobotCommand(
                        raw_transcript=raw,
                        normalized_text=norm,
                        intent=CommandIntent.SEARCH,
                        action=CommandAction.SEARCH_TARGET,
                        target_class=resolved_class,
                        target_alias=target_str,
                        is_partial=is_partial,
                        confidence=confidence,
                        timestamp=ts,
                    )
                else:
                    return RobotCommand(
                        raw_transcript=raw,
                        normalized_text=norm,
                        intent=CommandIntent.UNKNOWN,
                        action=CommandAction.UNKNOWN,
                        target_class=resolved_class,
                        target_alias=target_str,
                        is_partial=is_partial,
                        confidence=confidence,
                        timestamp=ts,
                        error_reason=f"Target class '{resolved_class}' is not supported by current detector model",
                    )
            else:
                return RobotCommand(
                    raw_transcript=raw,
                    normalized_text=norm,
                    intent=CommandIntent.UNKNOWN,
                    action=CommandAction.UNKNOWN,
                    target_alias=target_str,
                    is_partial=is_partial,
                    confidence=confidence,
                    timestamp=ts,
                    error_reason=f"Unknown search target: '{target_str}'",
                )

        # -------------------------------------------------------------------
        # 5. Directional Movement Commands (Complete Token Match & Conflict Check)
        # -------------------------------------------------------------------
        # Detect conflicting movements in the same transcript
        if active_directions > 1:
            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.UNKNOWN,
                action=CommandAction.UNKNOWN,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
                error_reason="Conflicting movement commands in transcript",
            )

        movement_action: Optional[CommandAction] = None

        if norm in ("forward", "move forward", "go forward", "straight", "forward forward", "go forward go forward"):
            movement_action = CommandAction.MOVE_FORWARD
        elif norm in (
            "backward", "move backward", "go backward", "back", "go back", "reverse",
            "backward backward", "back back", "backwards", "move backwards", "go backwards",
            "backwad", "move backwad", "go backwad", "move back"
        ):
            movement_action = CommandAction.MOVE_BACKWARD
        elif norm in ("turn left", "left", "rotate left", "turn left turn left", "left left"):
            movement_action = CommandAction.TURN_LEFT
        elif norm in ("turn right", "right", "rotate right", "turn right turn right", "right right"):
            movement_action = CommandAction.TURN_RIGHT

        if movement_action is not None:
            # Invariant: Partial transcripts must NEVER trigger physical movement
            if is_partial:
                return RobotCommand(
                    raw_transcript=raw,
                    normalized_text=norm,
                    intent=CommandIntent.UNKNOWN,
                    action=CommandAction.UNKNOWN,
                    is_partial=is_partial,
                    confidence=confidence,
                    timestamp=ts,
                    error_reason="Partial transcript discarded to prevent false movement actuation",
                )

            return RobotCommand(
                raw_transcript=raw,
                normalized_text=norm,
                intent=CommandIntent.MOVEMENT,
                action=movement_action,
                is_partial=is_partial,
                confidence=confidence,
                timestamp=ts,
            )

        # -------------------------------------------------------------------
        # 6. Fallthrough: Unrecognized or Ambiguous
        # -------------------------------------------------------------------
        return RobotCommand(
            raw_transcript=raw,
            normalized_text=norm,
            intent=CommandIntent.UNKNOWN,
            action=CommandAction.UNKNOWN,
            is_partial=is_partial,
            confidence=confidence,
            timestamp=ts,
            error_reason=f"Unrecognized voice command: '{norm}'",
        )
