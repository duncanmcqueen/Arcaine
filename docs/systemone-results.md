# SystemOne validation results and provenance

This file records the evidence for `/v1/systemone` on the named checkpoint. It
separates tests that exist, tests that were executed, and tests that were
independently reproduced. It is the "final report" referenced by
[systemone.md](systemone.md).

## Environment

| Item | Value |
|---|---|
| Target | `arcaine_server` (executable) |
| Checkpoint | `diffusiongemma-26B-A4B-it-AWQ-INT4` |
| GPU | Intel Arc Pro B70 (Battlemage G31), one device |
| Driver | Level-Zero V2, `20.2.0 [1.15.38646+7]` |
| Host kernel | `7.2.6-1-cachyos` |
| Toolchain | oneAPI `icpx`, `-fsycl-targets=intel_gpu_bmg_g31`, oneDNN |
| Build | `cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DARCAINE_SYCL_TARGETS=intel_gpu_bmg_g31 && cmake --build build --target arcaine_server -j4` |
| Commit | branch `fix/systemone-spec-gaps` (see the PR) |

## Executed in this work

Commands were run against a live server on port 7461 with `ARCAINE_API_KEY=local`.

| Check | Command | Result |
|---|---|---|
| HTTP shapes, validation, isolation, concurrency, lifecycle, graph bypass | `python3 scripts/test_systemone.py --base http://127.0.0.1:7461 --key local` | all checks passed |
| Direct API bounds before GPU | `decision_validation_check` | all checks passed |
| Scoring micro-harness | `slot_score_check`, 560 runs | all numerical checks passed |
| Error recovery, `after_prefill` | `test_systemone_recovery.py` | fault 500, next reads 200, chat 200 |
| Error recovery, `after_first_decode_submit` | `test_systemone_recovery.py` | fault 500, next reads 200, chat 200 |
| Error recovery, `in_decode_layer` | `test_systemone_recovery.py` | fault 500, next reads 200, chat 200 |
| Error recovery + graph bypass, `in_decode_layer` | `test_systemone_recovery.py --graph-mode` | structured reads 0 captures / 0 replays |
| Unsupported allocation mode | server with `DIFF_ARENA=off` | structured read rejected 500; chat 200 |
| Internal seed parsing | `arcaine_server --seed <value> --help` | `4294967296`, `1junk`, `-1`, `+5`, `99x` rejected; `42` accepted |
| Structured-read stress | `scripts/stress_systemone.py --iterations 150` | 150/150, no hang, 0 forcewake errors |
| OneDNN SDPA decode leg | same, server with `DIFF_ONEDNN_SDPA=decode` | 100/100, no hang |

## Implemented, not yet executed here

| Check | Command | Note |
|---|---|---|
| Prompt-length sweep + throughput | `python3 scripts/benchmark_systemone.py -p 512,1024,2048,4096 --iters 10 --warmup 2` | harness updated; run per server read policy (1 and 4) |
| Rendered artifacts vs checkpoint tokenizer | see `check_tokenizer_parity.py` | raw-encoding comparison only; compiled-artifact export is not implemented |
| Official Jev SDK request/response | `check_sdk_compat.py` | needs the `typesafe_sdk` package; response parsing only |

## Incomplete gates

- **Compiled-artifact tokenizer parity (V1):** the harness compares a
  Hugging-Face-rendered prompt and slot fixtures against the native encoder. It
  does not export `TokenizerBridge::build_prompt` prompt IDs, canvas tokens, slot
  position, or the full 255-label codebook and compare them to an independent
  checkpoint-tokenizer construction. Release gate not established.
- **Official SDK request construction (V3):** response parsing is covered when
  the SDK is installed; official request serialization and structured value
  preservation are not.
- **Behavior and reference comparison (V4):** no retained labeled evaluation and
  no pinned-vLLM comparison artifact. Accuracy, Brier, threshold coverage, and
  repeatability are unverified. A hosted comparison is separate quality
  evidence; local labeled evaluation does not need the hosted service.
- **Confidence:** local `normalized-entropy-v1` approximation, not exact Jev
  confidence. Exact parity is unresolved.
- **Graph-mode chat:** chat under the experimental `DIFF_NVFP4_SYCL_GRAPH=1`
  fails in this tree independently of `/v1/systemone` (reproduced on the base
  commit). Structured reads bypass capture and report 0/0 in the same mode.
