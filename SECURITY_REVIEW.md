# Security Review — Last 7 Merged PRs

**Scope:** Every merged PR on `main` at the time of review (`#1`, `#2`, `#3`, `#4`, `#5`, `#7`, `#8` — PR `#6` was never merged). The repository has only 7 merged PRs total; this review covers all of them.

**Method:** For each PR, the merge commit was checked out and the introduced diff was audited line-by-line for security and data-leak issues. No code was modified.

**Reviewed by:** Cursor Agent
**Date:** 2026-05-24

---

## Summary of findings

| Severity | PR  | Location                                                                       | Issue                                                                                                |
| -------- | --- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| HIGH     | #5  | `recsys_ranking/train.py`, `recsys_retrieval/train.py`                         | `torch.load(cfg.resume_from, weights_only=False, ...)` enables pickle RCE on resume-from checkpoint  |
| HIGH     | #1  | `recsys_serving/service.py`                                                    | gRPC server uses `add_insecure_port("[::]:...")` with no TLS and no authentication                   |
| MEDIUM   | #1  | `recsys_serving/service.py` + `recsys_serving/pipeline.py`                     | Unauthenticated callers can enumerate per-user historic categories / clicks via `user_id`            |
| MEDIUM   | #1  | `recsys_serving/service.py`                                                    | `top_n` / `retrieval_k` unbounded → memory-allocation DoS                                            |
| MEDIUM   | #1  | `recsys_serving/pipeline.py`                                                   | `self._pp_config.top_n = top_n` mutates shared state across concurrent gRPC worker threads           |
| MEDIUM   | #1  | `recsys_serving/feature_store.py`                                              | Single SQLite `Connection` shared across gRPC worker threads                                         |
| MEDIUM   | #5  | `recsys_ranking/train.py`, `recsys_retrieval/train.py`                         | `mlflow.log_params(asdict(cfg))` leaks `mlflow_tracking_uri` (may contain `user:token@…` credentials) |
| LOW      | #1  | `recsys_serving/feature_store.py`                                              | `IN (...)` clause sized by unbounded `retrieval_k` (compounds DoS)                                   |
| LOW      | #1  | `recsys_serving/pipeline.py`                                                   | Silent fabricated default user enables user-existence probing                                        |

**Clean PRs (no findings):** #2, #3, #4, #7, #8.

**Explicitly verified safe:**
- All SQL is parameterized via `?` placeholders, including the dynamic `IN (...)` clause in `get_item_features` — no SQL injection.
- YAML loading uses `yaml.safe_load` — no YAML deserialization RCE.
- No hard-coded credentials, tokens, API keys, or private URLs in any merged commit.
- CI workflows pin major versions of all third-party actions and do not use `pull_request_target`, so PR-supplied code cannot exfiltrate secrets.

---

## Per-PR findings

### PR #8 — `feat: add DCN cross-layers to MMoE-DeepFM ranking experts` (`a96ad91`)

Adds a pure-PyTorch `CrossLayer` / `DCNExpert` and three new `TrainConfig` fields (`use_dcn`, `num_cross_layers`, `dcn_rank`). No I/O, no auth surface, no serialization changes.

**Findings:** None.

---

### PR #7 — `feat: CI, pre-commit, coverage, metadata for public release` (`aeb180f`)

Adds `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, pyproject metadata, and ruff/mypy reformatting.

The workflow:
- Pins major versions for all third-party actions (`actions/checkout@v4`, `actions/setup-python@v5`, `astral-sh/setup-uv@v4`).
- Uses `uv sync --frozen`, so the lockfile is enforced.
- Triggers only on `push` to `main` and `pull_request` to `main` (not `pull_request_target`), so untrusted PR code cannot read repo secrets.
- Does not reference any `secrets.*` values.

**Findings:** None.

---

### PR #5 — `feat: metrics, MLflow tracking, checkpoint/resume, YAML config` (`1ff7515`)

#### HIGH — Insecure checkpoint deserialization

Both `packages/recsys-ranking/src/recsys_ranking/train.py` and `packages/recsys-retrieval/src/recsys_retrieval/train.py` execute:

```
ckpt = torch.load(cfg.resume_from, weights_only=False, map_location=device)
```

Explicitly setting `weights_only=False` opts out of PyTorch's pickle hardening. Loading a malicious `.pt` file results in arbitrary Python code execution at module-import time inside the trainer process. The same files correctly use `weights_only=True` when loading `best_model.pt` a few lines down — the unsafe mode is only used on the externally-controllable resume path (the path is read from CLI / YAML config).

**Recommendation:** Use `weights_only=True`. Optimizer state can be saved/restored via `torch.save` of plain tensors / state-dicts. If `weights_only=False` is unavoidable, restrict the resume path to a trusted directory and require explicit operator confirmation.

#### MEDIUM — Credential leakage via MLflow params

Both train scripts now run:

```
mlflow.log_params(dataclasses.asdict(cfg))
```

`TrainConfig` contains `mlflow_tracking_uri`. If an operator passes a URI of the form `https://user:token@host/...` (a common pattern for HTTP-basic-auth–backed MLflow servers), the credentials are persisted as a run parameter and become visible to anyone with read access to the tracking server — including the very server they authenticate against.

**Recommendation:** Redact / strip the userinfo component of `mlflow_tracking_uri` (and any other URI-shaped config field) before `log_params`. A simple `urllib.parse.urlparse(...)._replace(netloc=host).geturl()` is sufficient.

#### LOW — YAML loader

`recsys_common.config.load_config` uses `yaml.safe_load` — correct, not vulnerable. Listed only for completeness.

---

### PR #4 — `refactor: codebase review — bug fixes, DRY refactoring, doc fixes, changelog` (`1666d80`)

Doc-comment removal, DRY helpers (`pad_and_mask`, `data_utils.load_user_map/load_item_map`), and a *positive* security-relevant fix: `postprocess()` now defensively copies each candidate dict before mutating its `score`:

```
candidates = [dict(c) for c in candidates]
```

This prevents inadvertent caller-side state mutation.

**Findings:** None new.

---

### PR #3 — `docs: use generic Feature Store label in architecture diagrams` (`fb11e7f`)

README-only text change ("SQLite" → "Feature Store" in diagrams).

**Findings:** None.

---

### PR #2 — `docs: add README with architecture diagrams and CLAUDE.md` (`e163267`)

Pure markdown. Scanned for embedded tokens / API keys / credentials / private URLs — none present.

**Findings:** None.

---

### PR #1 — `feat: two-stage recommender system with MMoE-DeepFM ranking` (`e07f47a`)

#### HIGH — gRPC server bound on all interfaces with no transport security and no authentication

`packages/recsys-serving/src/recsys_serving/service.py`:

```
server.add_insecure_port(f"[::]:{port}")
```

Combined with:
- No `ServerInterceptor` for auth/authorization.
- No token check, no mTLS.
- Binds to `[::]` (all IPv4/IPv6 addresses), not loopback.

If the port is ever exposed beyond a private cluster network, any caller can query the service. The RPC also returns model-derived information about arbitrary `user_id`s (see next finding).

**Recommendation:** Switch to `server.add_secure_port(..., grpc.ssl_server_credentials(...))` and install an authentication interceptor (mTLS or a bearer-token check).

#### MEDIUM — PII / behavior-data exposure via unauthenticated `user_id` lookup

The `Recommend` RPC accepts an arbitrary `user_id` and returns recommendations derived from that user's stored historic categories and last-3-clicks (`SqliteFeatureStore.get_user_features` / `get_user_session`). With no authentication (previous finding), any caller can enumerate `user_id`s and infer user behavior from the returned scored item list and category multipliers.

**Recommendation:** Require the caller to prove they are the requested user (or an authorized service) before returning per-user results. Pair with rate limiting to slow enumeration even when authn is in place.

#### MEDIUM — Unbounded request parameters → DoS

`service.py` only validates `top_n > 0` and `retrieval_k > 0`; both are otherwise used directly:

```
top_n = request.top_n if request.top_n > 0 else 10
retrieval_k = request.retrieval_k if request.retrieval_k > 0 else 200
```

A client can send `retrieval_k = 2**31 - 1`, causing `RecommendationPipeline.recommend` to allocate huge numpy arrays (e.g. `np.tile(...)` of `(n, …)` rows, plus the corresponding ONNX inference). There is no rate limiting and no per-call resource budget.

**Recommendation:** Cap `top_n` and `retrieval_k` to documented maxima (e.g. 100 and 1000 respectively) and reject otherwise with `INVALID_ARGUMENT`.

#### MEDIUM — Cross-request state mutation in pipeline

`RecommendationPipeline.recommend` does:

```
self._pp_config.top_n = top_n
```

…on a shared instance. The server uses `ThreadPoolExecutor(max_workers=4)`, so concurrent gRPC requests can interleave: request A writes `top_n=50`, request B writes `top_n=5`, and request A then reads `5`. Net effect: requests can receive a wrong-length result list — a correctness/integrity issue with mild information-exposure flavor (a low-`top_n` caller can be made to receive results sized for another caller). PR #4 fixed the analogous candidate-dict mutation but missed this one.

**Recommendation:** Pass `top_n` as an argument to `postprocess()` rather than mutating `self._pp_config`; or construct a fresh `PostprocessConfig` per request.

#### MEDIUM — SQLite connection shared across threads

`SqliteFeatureStore.__init__` calls `sqlite3.connect(str(db_path))` once and stores it on `self`. Python's `sqlite3` defaults to `check_same_thread=True`, so today concurrent gRPC worker threads will raise an error on access. If a future change sets `check_same_thread=False` without adding a lock, this becomes a race-condition / data-integrity issue.

SQL itself is fully parameterized (`?` placeholders, including the dynamic `IN (...)` in `get_item_features`) — **no SQL injection**.

**Recommendation:** Use a thread-local connection (or a connection pool) and document the threading model on `FeatureStore`.

#### LOW — Unbounded SQL `IN` clause

`get_item_features` builds an `IN (?, ?, …)` placeholder list from the caller's `candidate_ids`. Since `retrieval_k` is unbounded (see DoS finding), a huge `retrieval_k` can exceed SQLite's default ~999-parameter limit and raise. The effect is a noisy error rather than data corruption, but it compounds the DoS finding.

**Recommendation:** Cap `retrieval_k` (also fixes the DoS) and/or chunk the `IN (...)` lookup.

#### LOW — Silent default user fabrication enables existence probing

In `pipeline.py`, if the feature store has no record for `user_id`, the pipeline silently fabricates:

```
raw_user = RawUserFeatures(user_id=user_id, user_age=30.0)
```

…and continues. Combined with no authentication, this lets an attacker probe whether a `user_id` exists by comparing response shape/scores against the synthetic-user response.

**Recommendation:** Return an explicit error (e.g. `NOT_FOUND`) for unknown users, or ensure synthetic responses are statistically indistinguishable from real ones (much harder; prefer the explicit error).

---

## Methodology notes

- Each merge commit (`a96ad91`, `aeb180f`, `1ff7515`, `1666d80`, `fb11e7f`, `e163267`, `e07f47a`) was checked out individually with `git checkout <sha>`.
- For each PR, `git diff <sha>^1 <sha>` was inspected at the file level, focusing on:
  - I/O surfaces: gRPC service, SQLite store, file loaders, YAML loaders, checkpoint loaders.
  - Deserialization: `torch.load`, `pickle`, `yaml.load`, `json.loads`.
  - Auth / transport: `add_insecure_port`, interceptors, TLS configuration.
  - Logging: `mlflow.log_params`, `logger.info`, `print` — for credential / PII leakage.
  - Input handling: bounds, validation, parameterized SQL.
- Repo-wide grep was used to confirm absence of risky patterns (`pickle`, `exec(`, `eval(`, `os.system`, `subprocess`, `shell=True`, hard-coded URLs/secrets).
