"""Cryptographic integrity verification tests for preserved models in SS_PROJECT_VCR_FRESH."""
import hashlib
from pathlib import Path
import pytest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


EXPECTED_ONNX = {
    "best_v2_target.onnx": {
        "size": 12132618,
        "sha256": "138acd11750d31eb06a5b0458d21de85034a5782b9236633915ffa66fe635c8c",
    },
    "best_v1.onnx": {
        "size": 12131894,
        "sha256": "867fbc41052890fd733656f4da4a73c9e37bd1bd503b7ce58dc0eaf7e53c67d9",
    },
}

EXPECTED_VOSK_FILES = {
    "README": "c0cf286e4f7783306c5f6469b37db69228fb16803b03cae661edb2d7bba64ebb",
    "am/final.mdl": "75370a0137f9daf8f469dedd7daa4513ae7a621f03240c6e512e2b50b656a7b6",
    "conf/mfcc.conf": "1e2228006d01d805ad1c267fee9f79709ca87ac51bd82b0e3f5c69ba543f0fc4",
    "conf/model.conf": "8f14cb1eeb07c762c371db648c6be688d347236155ca0f64fb13b6567a8ce81f",
    "graph/disambig_tid.int": "9ad87cc166d0998f08f758f47a6223a120dfffcbee805c4849c3aa5e6bb3c0fc",
    "graph/Gr.fst": "023c8b7e30704a9e37765c635c252e608a02f361235bf94abdcf2a5225d85b20",
    "graph/HCLr.fst": "5caafba3081e1646545ac6bff0dd7a318e53dcbdc86f237909ce1d2ac1293d34",
    "graph/phones/word_boundary.int": "da199d9c991e0e84681ddbb34627b915b26302d50a8fdaa23c51e2bc3a50b5c3",
    "ivector/final.dubm": "8c5d7dd69d2122313baaf19f61f35dd3fa18b70c62ac0687e311e1c46e6daca7",
    "ivector/final.ie": "3f37faf90c375b9e4740b569398b5829ed9cc07d19be6d441f72c3b71d7efcc6",
    "ivector/final.mat": "ddd83586dc5f928cda8738b922c85ffe38fc789cb5f9151a712ca12f37265382",
    "ivector/global_cmvn.stats": "33be09afcc80059847a275c3d043b51f1ab954c7c2438ddbbf4745e8ba144ff9",
    "ivector/online_cmvn.conf": "a2f3571754b64297cb7efb2e7ca3df61995c5a45fcbb97188f90613552bb2dfe",
    "ivector/splice.conf": "9f0c5f7c82d18eaf25d8bce470efa9f7741f88411fe428774bc0a9bb69a24756",
}


def test_onnx_models_cryptographic_integrity():
    """Trained ONNX weights must match exact preserved checksums and byte lengths."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    onnx_dir = base_dir / "models" / "object_detection"

    for filename, expected in EXPECTED_ONNX.items():
        file_path = onnx_dir / filename
        assert file_path.is_file(), f"Missing ONNX model: {file_path}"
        assert file_path.stat().st_size == expected["size"], f"Size mismatch for {filename}"
        calculated_hash = sha256_file(file_path)
        assert calculated_hash == expected["sha256"], f"Checksum mismatch for {filename}: {calculated_hash}"


def test_vosk_model_files_match_checksums():
    """All preserved Vosk model files must match exact cryptographic checksums."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    vosk_dir = base_dir / "models" / "voice" / "vosk-model"

    assert vosk_dir.is_dir(), f"Missing Vosk directory: {vosk_dir}"

    for rel_path, expected_hash in EXPECTED_VOSK_FILES.items():
        file_path = vosk_dir / rel_path
        assert file_path.is_file(), f"Missing Vosk model component: {file_path}"
        calculated_hash = sha256_file(file_path)
        assert calculated_hash == expected_hash, f"Checksum mismatch for Vosk component {rel_path}"



def test_vosk_model_loadability():
    """Load the preserved model when the optional native Vosk package is installed."""
    vosk = pytest.importorskip("vosk", reason="Vosk native package is not installed in this test environment")
    base_dir = Path(__file__).resolve().parent.parent.parent
    vosk_dir = base_dir / "models" / "voice" / "vosk-model"
    model = vosk.Model(str(vosk_dir))
    assert model is not None
