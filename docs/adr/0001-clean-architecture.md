# ADR 0001: Clean Zero-Legacy Architecture & Strict Abstraction

## Context
The legacy implementation of `SS_Project_VCR` suffered from architectural debt:
- Proliferation of broken backup files (`.bak`, `.bak_task`, `.phase1_bak`).
- Tight coupling between business logic, hardware drivers, and tests, leading to risk of accidental GPIO toggling during testing.
- Severe substring parsing defects (e.g. "backpack" matching "back" -> `BACKWARD`).
- Discrepancies between configuration files (`config.yaml` vs `config.py`).
- Subclassing standard data types (`DistanceFloat(float)`) with non-standard comparison logic.

## Decisions
1. **Zero-Legacy Implementation:** All new code is built from scratch within `SS_Project_VCR_FRESH` without copying legacy application files wholesale.
2. **Protocol/Interface Driven:** Define abstract interfaces for Motors, Ultrasonic Sensor, Camera, and Microphone using Python `typing.Protocol` or `abc.ABC`.
3. **Deterministic Mocks:** Provide production-grade, zero-GPIO mock implementations of all hardware interfaces for unit and integration testing.
4. **Unified Configuration:** Consolidate all system parameters into a single validated configuration system with fail-closed validation rules.
5. **Preserved Weights:** Preserve original trained YOLO ONNX models (`best_v2_target.onnx`, `best_v1.onnx`) and Vosk offline speech model without modifying a single byte.
6. **Token-Level Intent Parsing:** Implement voice command parsing using full word tokens and strict precedence: E-Stop/Halt > Stop/Cancel > Navigation > Object Find.

## Consequences
- Testing can be performed on developer workstations or CI environments without Raspberry Pi hardware or root privileges.
- Hardware drivers are pluggable adapters isolated behind clear contracts.
- Safety invariants (E-Stop latch, sensor fail-closed) are guaranteed by architectural boundaries.
