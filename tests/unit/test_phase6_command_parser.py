"""Phase 6 unit tests: Deterministic command parser, token matching,
safety priorities, object aliases, speed bounds, and partial transcript filtering.
"""
from __future__ import annotations

import pytest

from app.command.command_parser import (
    ALL_VERIFIED_CLASSES,
    TARGET_ALIAS_MAP,
    CommandAction,
    CommandIntent,
    CommandParser,
    RobotCommand,
)


@pytest.fixture
def parser() -> CommandParser:
    return CommandParser()


# ---------------------------------------------------------------------------
# 1. Table-Driven Movement Commands Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase, expected_action",
    [
        ("forward", CommandAction.MOVE_FORWARD),
        ("move forward", CommandAction.MOVE_FORWARD),
        ("go forward", CommandAction.MOVE_FORWARD),
        ("backward", CommandAction.MOVE_BACKWARD),
        ("move backward", CommandAction.MOVE_BACKWARD),
        ("back", CommandAction.MOVE_BACKWARD),
        ("go back", CommandAction.MOVE_BACKWARD),
        ("go backward", CommandAction.MOVE_BACKWARD),
        ("turn left", CommandAction.TURN_LEFT),
        ("left", CommandAction.TURN_LEFT),
        ("turn right", CommandAction.TURN_RIGHT),
        ("right", CommandAction.TURN_RIGHT),
    ],
)
def test_movement_commands_table_driven(parser: CommandParser, phrase: str, expected_action: CommandAction):
    cmd = parser.parse(phrase)
    assert cmd.intent == CommandIntent.MOVEMENT
    assert cmd.action == expected_action
    assert cmd.is_executable is True
    assert cmd.causes_movement is True


# ---------------------------------------------------------------------------
# 2. Table-Driven Safety & Stop Priority Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase, expected_intent, expected_action",
    [
        ("emergency stop", CommandIntent.SAFETY, CommandAction.EMERGENCY_STOP),
        ("e stop", CommandIntent.SAFETY, CommandAction.EMERGENCY_STOP),
        ("stop", CommandIntent.SAFETY, CommandAction.STOP),
        ("halt", CommandIntent.SAFETY, CommandAction.STOP),
        ("freeze", CommandIntent.SAFETY, CommandAction.STOP),
        ("stop scanning", CommandIntent.CONTROL, CommandAction.STOP_SCANNING),
        ("cancel search", CommandIntent.CONTROL, CommandAction.CANCEL_SEARCH),
        ("stop searching", CommandIntent.CONTROL, CommandAction.CANCEL_SEARCH),
        ("cancel", CommandIntent.CONTROL, CommandAction.CANCEL_SEARCH),
        ("resume listening", CommandIntent.CONTROL, CommandAction.RESUME_LISTENING),
    ],
)
def test_safety_and_stop_priority_table_driven(
    parser: CommandParser, phrase: str, expected_intent: CommandIntent, expected_action: CommandAction
):
    cmd = parser.parse(phrase)
    assert cmd.intent == expected_intent
    assert cmd.action == expected_action
    assert cmd.causes_movement is False


# ---------------------------------------------------------------------------
# 3. Table-Driven Control & Query Commands Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase, expected_action",
    [
        ("status", CommandAction.GET_STATUS),
        ("robot status", CommandAction.GET_STATUS),
        ("system status", CommandAction.GET_STATUS),
        ("help", CommandAction.GET_HELP),
        ("commands", CommandAction.GET_HELP),
        ("what do you see", CommandAction.QUERY_VISION),
        ("what do you see now", CommandAction.QUERY_VISION),
    ],
)
def test_control_commands_table_driven(parser: CommandParser, phrase: str, expected_action: CommandAction):
    cmd = parser.parse(phrase)
    assert cmd.intent == CommandIntent.CONTROL
    assert cmd.action == expected_action
    assert cmd.causes_movement is False


# ---------------------------------------------------------------------------
# 4. Critical Regression: Backpack vs Backward Disambiguation
# ---------------------------------------------------------------------------

def test_find_backpack_never_parses_as_backward(parser: CommandParser):
    """'find backpack' must parse as SEARCH_TARGET, never MOVE_BACKWARD."""
    cmd = parser.parse("find backpack")
    assert cmd.intent == CommandIntent.SEARCH
    assert cmd.action == CommandAction.SEARCH_TARGET
    assert cmd.target_class == "backpack"
    assert cmd.action != CommandAction.MOVE_BACKWARD

    # Also test 'bag' alias
    cmd_bag = parser.parse("locate the bag")
    assert cmd_bag.intent == CommandIntent.SEARCH
    assert cmd_bag.action == CommandAction.SEARCH_TARGET
    assert cmd_bag.target_class == "backpack"
    assert cmd_bag.action != CommandAction.MOVE_BACKWARD


# ---------------------------------------------------------------------------
# 5. Table-Driven Search Commands and Target Aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase, expected_class, expected_alias",
    [
        ("find backpack", "backpack", "backpack"),
        ("find bag", "backpack", "bag"),
        ("look for scissors", "scissors", "scissors"),
        ("search for a scissor", "scissors", "scissor"),
        ("locate shears", "scissors", "shears"),
        ("search for a pair of scissors", "scissors", "pair of scissors"),
        ("find bottle", "bottle", "bottle"),
        ("look for the water bottle", "bottle", "water bottle"),
        ("find red target", "RED_TARGET", "red target"),
        ("search for red object", "RED_TARGET", "red object"),
        ("find red", "RED_TARGET", "red"),
        ("locate blue target", "BLUE_TARGET", "blue target"),
        ("find blue", "BLUE_TARGET", "blue"),
        ("find phone", "cell_phone", "phone"),
        ("search for cell phone", "cell_phone", "cell phone"),
        ("find trash can", "trash_can", "trash can"),
        ("find trash", "trash_can", "trash"),
        ("locate the plant", "potted_plant", "plant"),
        ("find person", "person", "person"),
        ("search for a chair", "chair", "chair"),
        ("find whiteboard", "whiteboard", "whiteboard"),
        ("look for board", "whiteboard", "board"),
        ("find remote", "remote_control", "remote"),
        ("find fire extinguisher", "fire_extinguisher", "fire extinguisher"),
    ],
)
def test_search_commands_and_aliases(
    parser: CommandParser, phrase: str, expected_class: str, expected_alias: str
):
    cmd = parser.parse(phrase)
    assert cmd.intent == CommandIntent.SEARCH
    assert cmd.action == CommandAction.SEARCH_TARGET
    assert cmd.target_class == expected_class
    assert cmd.target_alias == expected_alias


# ---------------------------------------------------------------------------
# 6. Speed Command Bounds and Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase, expected_speed",
    [
        ("speed slow", 0.30),
        ("speed medium", 0.50),
        ("speed normal", 0.50),
        ("speed fast", 0.75),
        ("speed 0.4", 0.40),
        ("speed 0.8", 0.80),
        ("speed 1.5", 0.80),   # Clamped to max_speed (0.80)
        ("speed 0.1", 0.20),   # Clamped to min_speed (0.20)
    ],
)
def test_speed_commands_valid(parser: CommandParser, phrase: str, expected_speed: float):
    cmd = parser.parse(phrase)
    assert cmd.intent == CommandIntent.SPEED
    assert cmd.action == CommandAction.SET_SPEED
    assert cmd.speed == expected_speed


def test_speed_commands_invalid(parser: CommandParser):
    # Negative speed outside [0.0, 1.0]
    cmd_neg = parser.parse("speed -0.5")
    assert cmd_neg.intent == CommandIntent.UNKNOWN
    assert cmd_neg.is_executable is False

    # Non-numeric garbage
    cmd_str = parser.parse("speed warp")
    assert cmd_str.intent == CommandIntent.UNKNOWN
    assert cmd_str.is_executable is False


# ---------------------------------------------------------------------------
# 7. Unsupported Objects Do Not Cause Movement
# ---------------------------------------------------------------------------

def test_unsupported_objects_rejected(parser: CommandParser):
    """Unknown classes must result in UNKNOWN intent and zero movement."""
    cmd_alien = parser.parse("find alien")
    assert cmd_alien.intent == CommandIntent.UNKNOWN
    assert cmd_alien.action == CommandAction.UNKNOWN
    assert cmd_alien.is_executable is False
    assert cmd_alien.causes_movement is False

    cmd_cat = parser.parse("locate cat")
    assert cmd_cat.intent == CommandIntent.UNKNOWN
    assert cmd_cat.is_executable is False


def test_unsupported_class_by_model_rejected():
    """If detector model lacks scissors (e.g. best_v1), search for scissors is rejected."""
    # Restrict supported classes to exclude 'scissors'
    classes_v1 = set(ALL_VERIFIED_CLASSES) - {"scissors"}
    parser_v1 = CommandParser(supported_classes=classes_v1)

    cmd = parser_v1.parse("find scissors")
    assert cmd.intent == CommandIntent.UNKNOWN
    assert cmd.is_executable is False
    assert "not supported by current detector model" in (cmd.error_reason or "")


# ---------------------------------------------------------------------------
# 8. Partial Transcript Filtering & Safety Invariants
# ---------------------------------------------------------------------------

def test_partial_transcripts_do_not_cause_movement(parser: CommandParser):
    """Partial transcripts must be discarded for movement to avoid false movement."""
    # Movement partials rejected
    cmd_fwd = parser.parse("forward", is_partial=True)
    assert cmd_fwd.intent == CommandIntent.UNKNOWN
    assert cmd_fwd.is_executable is False
    assert cmd_fwd.causes_movement is False

    cmd_left = parser.parse("turn left", is_partial=True)
    assert cmd_left.intent == CommandIntent.UNKNOWN
    assert cmd_left.is_executable is False

    # Search partials rejected
    cmd_search = parser.parse("find bottle", is_partial=True)
    assert cmd_search.intent == CommandIntent.UNKNOWN
    assert cmd_search.is_executable is False


def test_partial_transcripts_honor_immediate_stops(parser: CommandParser):
    """Partial transcripts MUST execute emergency stops and cancellation immediately."""
    # Stop partials allowed for safety responsiveness
    cmd_stop = parser.parse("stop", is_partial=True)
    assert cmd_stop.intent == CommandIntent.SAFETY
    assert cmd_stop.action == CommandAction.STOP
    assert cmd_stop.is_executable is True

    cmd_estop = parser.parse("emergency stop", is_partial=True)
    assert cmd_estop.intent == CommandIntent.SAFETY
    assert cmd_estop.action == CommandAction.EMERGENCY_STOP

    cmd_cancel = parser.parse("cancel", is_partial=True)
    assert cmd_cancel.intent == CommandIntent.CONTROL
    assert cmd_cancel.action == CommandAction.CANCEL_SEARCH


# ---------------------------------------------------------------------------
# 9. Normalization, Punctuation and Whitespace Tests
# ---------------------------------------------------------------------------

def test_normalization_and_punctuation(parser: CommandParser):
    # Conversational fillers stripped
    cmd1 = parser.parse("  PLEASE,  move   FORWARD! ")
    assert cmd1.intent == CommandIntent.MOVEMENT
    assert cmd1.action == CommandAction.MOVE_FORWARD

    cmd2 = parser.parse("Hey robot, what do you see?")
    assert cmd2.intent == CommandIntent.CONTROL
    assert cmd2.action == CommandAction.QUERY_VISION

    cmd3 = parser.parse("Robot, stop.")
    assert cmd3.intent == CommandIntent.SAFETY
    assert cmd3.action == CommandAction.STOP

    cmd4 = parser.parse("FIND... THE... BOTTLE???")
    assert cmd4.intent == CommandIntent.SEARCH
    assert cmd4.action == CommandAction.SEARCH_TARGET
    assert cmd4.target_class == "bottle"


def test_documented_stop_and_object_aliases_are_supported():
    """Aliases documented in the user guide must map to verified classes/actions."""
    parser = CommandParser()
    assert parser.parse("wait").action == CommandAction.STOP
    assert parser.parse("find seat").target_class == "chair"
    assert parser.parse("find notebook").target_class == "laptop"
    assert parser.parse("find mug").target_class == "cup"


def test_nonfinite_speed_commands_are_rejected():
    """NaN and infinity are invalid speed commands, not clamped motor requests."""
    parser = CommandParser()
    for phrase in ("speed nan", "speed inf", "speed -inf"):
        command = parser.parse(phrase)
        assert command.action == CommandAction.UNKNOWN
        assert command.is_executable is False


def test_voice_search_intent_parsed_without_starting_mission():
    """Parsing search commands strictly extracts intent and target without side effects."""
    parser = CommandParser()
    for phrase, expected_target in [
        ("find bottle", "bottle"),
        ("locate person", "person"),
        ("search for chair", "chair"),
        ("look for red target", "RED_TARGET"),
    ]:
        cmd = parser.parse(phrase)
        assert cmd.intent == CommandIntent.SEARCH
        assert cmd.action == CommandAction.SEARCH_TARGET
        assert cmd.target_class == expected_target
        assert cmd.is_executable is True
        # Pure parser has zero state machine or motor side-effects
        assert isinstance(cmd, RobotCommand)


def test_conflicting_movement_commands_rejected():
    """Conflicting directional commands in the same phrase must fail closed as UNKNOWN."""
    parser = CommandParser()
    for phrase in (
        "forward backward",
        "go forward move back",
        "turn left turn right",
        "left right",
        "go forward and turn left",
        "move backward and turn right",
        "find bottle and move forward",
        "search for chair turn left",
    ):
        cmd = parser.parse(phrase)
        assert cmd.intent == CommandIntent.UNKNOWN, f"Phrase '{phrase}' unexpectedly parsed intent: {cmd.intent}"
        assert cmd.action == CommandAction.UNKNOWN, f"Phrase '{phrase}' unexpectedly parsed action: {cmd.action}"
        assert cmd.is_executable is False
        assert cmd.causes_movement is False


def test_safety_stop_preempts_conflicting_movement_words():
    """Emergency stop and stop commands must take immediate precedence even if movement words are present."""
    parser = CommandParser()
    # General stop overrides movement words
    cmd1 = parser.parse("forward stop")
    assert cmd1.intent == CommandIntent.SAFETY
    assert cmd1.action == CommandAction.STOP
    assert cmd1.causes_movement is False

    cmd2 = parser.parse("stop forward")
    assert cmd2.intent == CommandIntent.SAFETY
    assert cmd2.action == CommandAction.STOP

    # Emergency stop overrides movement words
    cmd3 = parser.parse("emergency stop move forward")
    assert cmd3.intent == CommandIntent.SAFETY
    assert cmd3.action == CommandAction.EMERGENCY_STOP

    cmd4 = parser.parse("go back and halt")
    assert cmd4.intent == CommandIntent.SAFETY
    assert cmd4.action == CommandAction.STOP


def test_repeated_consistent_commands_supported():
    """Repeated single-action phrases parse safely without conflict."""
    parser = CommandParser()
    assert parser.parse("stop stop").action == CommandAction.STOP
    assert parser.parse("forward forward").action == CommandAction.MOVE_FORWARD
    assert parser.parse("left left").action == CommandAction.TURN_LEFT
    assert parser.parse("right right").action == CommandAction.TURN_RIGHT


def test_duplicate_transcript_suppression_in_runtime():
    """Runtime suppresses duplicate non-safety transcripts in debounce window, but never stops."""
    from app.runtime import RobotRuntime
    from app.hardware.mock_hardware import (
        MockMotorController,
        MockDistanceSensor,
        MockCamera,
        MockAudioSource,
        MockSpeechRecognizer,
        MockDetector,
    )

    runtime = RobotRuntime(
        motor_controller=MockMotorController(),
        distance_sensor=MockDistanceSensor(),
        camera=MockCamera(),
        audio_source=MockAudioSource(),
        speech_recognizer=MockSpeechRecognizer(),
        detector=MockDetector(),
    )
    runtime.start()
    try:
        # 1. First forward command accepted
        ok1, cmd1, _ = runtime.dispatch_text_command("forward")
        assert ok1 is True
        assert cmd1.action == CommandAction.MOVE_FORWARD

        # 2. Immediate duplicate forward command suppressed within debounce window
        ok2, cmd2, msg2 = runtime.dispatch_text_command("forward")
        assert ok2 is False
        assert "Duplicate transcript suppressed" in msg2

        # 3. Stop command immediately after is NEVER suppressed
        ok3, cmd3, _ = runtime.dispatch_text_command("stop")
        assert ok3 is True
        assert cmd3.action == CommandAction.STOP

        # 4. Repeated stop is NEVER suppressed (safety-critical)
        ok4, cmd4, _ = runtime.dispatch_text_command("stop")
        assert ok4 is True
        assert cmd4.action == CommandAction.STOP
    finally:
        runtime.stop()
