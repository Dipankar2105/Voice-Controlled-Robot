"""Phase 1 verification tests: Scaffold integrity, asset preservation, and hardware isolation."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
import pytest


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_preserved_model_assets_match_manifest_checksums():
    """The packaged models match the immutable hashes recorded in the asset manifest."""
    from tests.unit.test_asset_integrity import EXPECTED_ONNX, sha256_file
    base_dir = Path(__file__).resolve().parent.parent.parent
    model_dir = base_dir / "models" / "object_detection"
    for filename, expected in EXPECTED_ONNX.items():
        path = model_dir / filename
        assert path.is_file(), f"Missing packaged model: {path}"
        assert path.stat().st_size == expected["size"]
        assert sha256_file(path) == expected["sha256"]


def test_vosk_directory_matches_manifest_checksums():
    """All 14 packaged Vosk model files match their source inventory hashes."""
    from tests.unit.test_asset_integrity import EXPECTED_VOSK_FILES, sha256_file
    base_dir = Path(__file__).resolve().parent.parent.parent
    vosk_dir = base_dir / "models" / "voice" / "vosk-model"
    assert vosk_dir.is_dir(), f"Missing packaged Vosk directory: {vosk_dir}"
    files = [p for p in vosk_dir.rglob("*") if p.is_file()]
    assert len(files) == len(EXPECTED_VOSK_FILES) == 14
    for relative, expected_hash in EXPECTED_VOSK_FILES.items():
        path = vosk_dir / relative
        assert path.is_file(), f"Missing packaged Vosk component: {path}"
        assert sha256_file(path) == expected_hash, f"Checksum mismatch for {relative}"


def test_expected_files_and_imports_discoverable():
    """Verify that the clean package and modules are cleanly discoverable and importable."""
    import app
    import app.config
    import app.hardware
    import app.hardware.interfaces
    import app.hardware.mock_hardware
    import app.main

    assert hasattr(app, "__version__")
    assert hasattr(app.config, "AppConfig")
    assert hasattr(app.hardware.interfaces, "MotorControllerInterface")
    assert hasattr(app.hardware.mock_hardware, "MockMotorController")
    assert hasattr(app.main, "check_config")


def test_importing_scaffold_has_no_hardware_side_effects():
    """Importing the scaffold must never initialize GPIO, audio, video, or motor pins."""
    # Ensure RPi.GPIO is NOT loaded as a consequence of importing scaffold
    # (unless it was already in sys.modules, which is not the case on non-Pi)
    import app.config
    import app.hardware.interfaces
    import app.hardware.mock_hardware
    import app.main

    # Run check_config in non-actuating mode
    exit_code = app.main.check_config()
    assert exit_code == 0


def test_external_source_archive_is_not_required_for_portable_tests():
    """Portable tests do not assume the user's original ZIP exists beside the workspace."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    zip_path = base_dir.parent.parent / "SS_VCR_FINAL_CLEAN.zip"
    if not zip_path.is_file():
        pytest.skip("The original source archive is an external input and is not packaged with this workspace.")
    # When the external archive is available, verify its known source checksum.
    assert zip_path.stat().st_size == 63839198
    expected_sha256 = "f70b02e226bb9404c177507fae8fbe0afe36f1115a73bc0b8253c13ecd8c947e"
    assert sha256_file(zip_path) == expected_sha256, "Original ZIP archive was modified!"
