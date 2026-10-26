# ASSET PRESERVATION MANIFEST
**Project:** SS_PROJECT_VCR_FRESH  
**Creation Date:** 2026-10-21  
**Verification Standard:** Cryptographic SHA-256 bit-for-bit equivalence  

This manifest catalogs every external binary asset preserved from the baseline archive. No model has been retrained, converted, modified, or altered in byte sequence.

---

## 1. Object Detection Models (ONNX)

| Asset Name | Source Path | Destination Path | Size (Bytes) | SHA-256 Checksum | Verified Input/Output Contract |
|---|---|---|---|---|---|
| **best_v2_target.onnx** | `../models/object_detection/best_v2_target.onnx` | `models/object_detection/best_v2_target.onnx` | 12,132,618 | `138acd11750d31eb06a5b0458d21de85034a5782b9236633915ffa66fe635c8c` | Input: `images` [1, 3, 320, 320] Float32<br>Output: `output0` [1, 30, 2100] Float32<br>Classes: 26 classes |
| **best_v1.onnx** | `../models/object_detection/best_v1.onnx` | `models/object_detection/best_v1.onnx` | 12,131,894 | `867fbc41052890fd733656f4da4a73c9e37bd1bd503b7ce58dc0eaf7e53c67d9` | Input: `images` [1, 3, 320, 320] Float32<br>Output: `output0` [1, 29, 2100] Float32<br>Classes: 25 classes |

---

## 2. Offline Speech Recognition (Vosk Small English Model)

**Model Release:** `vosk-model-small-en-us-0.15`  
**Destination Root:** `models/voice/vosk-model/`  
**Total File Count:** 14 files  
**Total Directory Size:** 70,898,873 bytes  

| Relative File Path | Size (Bytes) | Cryptographic SHA-256 Checksum | Subsystem Function |
|---|---|---|---|
| `README` | 199 | `c0cf286e4f7783306c5f6469b37db69228fb16803b03cae661edb2d7bba64ebb` | Upstream Vosk release metadata |
| `am/final.mdl` | 15,962,575 | `75370a0137f9daf8f469dedd7daa4513ae7a621f03240c6e512e2b50b656a7b6` | Kaldi acoustic model parameters |
| `conf/mfcc.conf` | 131 | `1e2228006d01d805ad1c267fee9f79709ca87ac51bd82b0e3f5c69ba543f0fc4` | 16 kHz MFCC extraction settings |
| `conf/model.conf` | 290 | `8f14cb1eeb07c762c371db648c6be688d347236155ca0f64fb13b6567a8ce81f` | Neural network graph & decoder config |
| `graph/disambig_tid.int` | 102 | `9ad87cc166d0998f08f758f47a6223a120dfffcbee805c4849c3aa5e6bb3c0fc` | Kaldi disambiguation table |
| `graph/Gr.fst` | 24,013,795 | `023c8b7e30704a9e37765c635c252e608a02f361235bf94abdcf2a5225d85b20` | Grammar Finite State Transducer |
| `graph/HCLr.fst` | 22,416,994 | `5caafba3081e1646545ac6bff0dd7a318e53dcbdc86f237909ce1d2ac1293d34` | Composite acoustic-lexicon FST |
| `graph/phones/word_boundary.int` | 1,761 | `da199d9c991e0e84681ddbb34627b915b26302d50a8fdaa23c51e2bc3a50b5c3` | Word boundary phone table |
| `ivector/final.dubm` | 168,048 | `8c5d7dd69d2122313baaf19f61f35dd3fa18b70c62ac0687e311e1c46e6daca7` | Universal Background Model |
| `ivector/final.ie` | 8,288,887 | `3f37faf90c375b9e4740b569398b5829ed9cc07d19be6d441f72c3b71d7efcc6` | i-vector extraction matrix |
| `ivector/final.mat` | 44,975 | `ddd83586dc5f928cda8738b922c85ffe38fc789cb5f9151a712ca12f37265382` | Projection transform matrix |
| `ivector/global_cmvn.stats` | 1,080 | `33be09afcc80059847a275c3d043b51f1ab954c7c2438ddbbf4745e8ba144ff9` | Cepstral Mean/Variance statistics |
| `ivector/online_cmvn.conf` | 95 | `a2f3571754b64297cb7efb2e7ca3df61995c5a45fcbb97188f90613552bb2dfe` | Online sliding-window CMVN settings |
| `ivector/splice.conf` | 35 | `9f0c5f7c82d18eaf25d8bce470efa9f7741f88411fe428774bc0a9bb69a24756` | Feature frame splicing config |
