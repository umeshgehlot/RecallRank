# CLAUDE.md

## Project Overview

RecallRank is a two-stage neural recommender system: retrieval (Two-Tower) → ranking (MMoE-DeepFM), served via gRPC with ONNX Runtime.

Monorepo with 5 uv workspace packages under `packages/`:

- **recsys-common** — Pydantic schemas, stateless transforms, VocabStore, FeatureProcessor. No ML dependencies. Every other package depends on this.
- **recsys-data** — Synthetic data generation and dataset compilation (freezes vocab/stats/popularity into artifacts).
- **recsys-retrieval** — UserTower + ItemTower (PyTorch), InfoNCE+LogQ loss, FaissIndex, training loop with ONNX export.
- **recsys-ranking** — MMoE-DeepFM (PyTorch) with CTR+CVR heads, optional DCN cross-layers in experts, multi-task training with gradient masking, ONNX export.
- **recsys-serving** — gRPC service, RecommendationPipeline (ONNX Runtime), SqliteFeatureStore, post-processing.

## Build & Test

```bash
uv sync                    # Install all workspace packages + dev deps
uv run pytest              # Run full test suite
uv run pytest --cov        # Tests with coverage report
uv run pytest tests/integration/  # Integration tests only
uv run ruff check .        # Lint
uv run ruff format --check .  # Format check
uv run mypy packages/      # Type check
```

## CI

GitHub Actions runs on every PR and push to main (`.github/workflows/ci.yml`):
- **lint** job: ruff check, ruff format --check, mypy
- **test** job: pytest with coverage, uploads XML report as artifact

## Pre-commit

Pre-commit hooks run `ruff --fix` and `ruff-format` on staged files. Install with `pre-commit install` after `uv sync`.

## Package Dependency Graph

```
recsys-common (no ML deps)
    ↑
    ├── recsys-data (pyarrow)
    ├── recsys-retrieval (torch, faiss-cpu, onnxruntime)
    ├── recsys-ranking (torch, onnxruntime)
    └── recsys-serving (grpcio, onnxruntime, recsys-retrieval, recsys-ranking)
```

recsys-common is the only cross-cutting dependency. Retrieval and ranking do not depend on each other.

## Key Conventions

- **Schemas** live in `recsys_common.schemas` (Pydantic models). Raw features go in, processed features come out.
- **FeatureProcessor** (`recsys_common.processor`) is the single source of truth for feature transforms. Used identically offline (in `compile.py`) and online (in `pipeline.py`) to guarantee zero skew.
- **Frozen artifacts** (vocab.json, stats.json, popularity.json) are produced by `recsys_data.compile` and consumed read-only everywhere else.
- **Protocol classes** define swappable interfaces: `VectorIndex` (Faiss↔Milvus) and `FeatureStore` (SQLite↔Redis) in their respective packages.
- **ONNX export** happens at the end of each training script. Serving uses ONNX Runtime, not PyTorch.
- **Multi-task ranking** returns `(ctr_logits, cvr_logits)`. CVR loss uses gradient masking (only computed on clicked rows).

## Testing Patterns

- Root `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` and pre-imports torch before faiss to avoid OpenMP library conflicts on macOS.
- Integration tests in `tests/integration/` generate small synthetic datasets via fixtures, run the full pipeline, and verify outputs.
- `test_skew.py` validates that FeatureProcessor produces identical results across independent instances (the zero-skew guarantee).
- Test fixtures use `tmp_path_factory` with `scope="module"` for compiled artifacts shared across tests.

## Common Workflows

**Adding a new feature:**
1. Add the raw field to the appropriate schema in `recsys_common/schemas.py`
2. Add the transform in `recsys_common/transforms.py` if it's a new transform type
3. Update `FeatureProcessor` in `recsys_common/processor.py` to apply the transform
4. Update `compile.py` in recsys-data if the feature needs new frozen artifacts
5. Update the model's `forward()` to consume the new processed feature
6. Update the dataset class to include the feature in its output dict
7. Update tests at each layer

**Swapping an infrastructure component:**
The `VectorIndex` protocol (`recsys_retrieval.index`) and `FeatureStore` protocol (`recsys_serving.feature_store`) define the interfaces. Implement the protocol, then pass the new implementation to `RecommendationPipeline`.

## Experiment Tracking (MLflow)

Both training scripts log to MLflow automatically: params, per-epoch metrics, and ONNX artifacts. Default tracking URI is local `./mlruns`. Override via `mlflow_tracking_uri` in config YAML or `TrainConfig`.

```bash
uv run python scripts/train_retrieval.py --config configs/retrieval.yaml
mlflow ui  # View runs at http://localhost:5000
```

## Config & Checkpoint

- YAML config: `configs/retrieval.yaml`, `configs/ranking.yaml` — loaded via `--config` flag on training scripts, with CLI args as overrides.
- Checkpoint/resume: both training scripts save `checkpoint.pt` (model + optimizer + scheduler + epoch + best metric). Resume via `--resume-from path/to/checkpoint.pt`.
- `load_config()` in `recsys_common.config` handles YAML loading with dict override merging.

## Scripts

All CLI entry points are in `scripts/`. They are thin wrappers around package functions:

| Script | What it does |
|--------|-------------|
| `generate_data.py` | Generates synthetic data → compiles to artifacts |
| `train_retrieval.py` | Trains Two-Tower model → exports ONNX (supports `--config`, `--resume-from`) |
| `train_ranking.py` | Trains MMoE-DeepFM → exports ONNX (supports `--config`, `--resume-from`) |
| `build_index.py` | Runs items through Item Tower → builds Faiss index |
| `populate_feature_store.py` | Loads user/item parquet data into SQLite feature store |
| `example_client.py` | Sends a test recommendation request to the gRPC server |
| `serve.py` | Starts gRPC server on port 50051 |
