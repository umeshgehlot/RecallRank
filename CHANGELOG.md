# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] - 2026-05-24

### Added
- `scripts/populate_feature_store.py` — populates SQLite feature store from parquet files (fixes broken Quick Start)
- `scripts/example_client.py` — runnable gRPC client with CLI args for testing recommendations
- `tests/integration/test_serving_pipeline.py` — end-to-end serving integration test: train → build index → populate → recommend

### Fixed
- Quick Start pipeline was broken: no step populated the feature store, so serve.py returned empty results
- `MAX_RETRIEVAL_K` reduced from 1000 to 900 to stay within SQLite's default 999-parameter limit
- CI badge and pyproject.toml URLs updated for repo rename to `mmoe-recsys`
- MLflow metric names used `@` (rejected by MLflow validator): `recall@K` → `recall_at_K`, etc.
- `recsys_pb2_grpc.py` used flat `import recsys_pb2` instead of relative import — crashed at runtime
- `RecommendationPipeline` now accepts separate `retrieval_model_dir` and `ranking_model_dir` — eliminates need to manually copy ranking model into retrieval dir

## [0.4.0] - 2026-05-24

### Added
- **DCN (Deep & Cross Network) cross-layers** in MMoE-DeepFM ranking model experts
  - `CrossLayer` module: explicit feature crossing via `x_{l+1} = x_0 * (W * x_l + b) + x_l`
  - `DCNExpert` module: cross-then-deep architecture (cross-layers followed by deep MLP)
  - Full-rank and low-rank DCN variants via `dcn_rank` config parameter
  - Configurable via `use_dcn`, `num_cross_layers`, `dcn_rank` in training config
  - ONNX-compatible: all DCN ops map to standard ONNX operators
  - Inspired by Pinterest's MMOE-DCN architecture:
    - [GPU Serving Two-Tower Models for Lightweight Ads Engagement Prediction](https://medium.com/pinterest-engineering/gpu-serving-two-tower-models-for-lightweight-ads-engagement-prediction-5a0ffb442f3b)
    - [Beyond Two Towers: Re-architecting the Serving Stack](https://medium.com/pinterest-engineering/beyond-two-towers-re-architecting-the-serving-stack-for-next-gen-ads-lightweight-ranking-models-1992f2b76cbb)
- 12 new tests: CrossLayer math/shapes, DCNExpert, full model DCN integration, backward compatibility

### Changed
- `MMoEDeepFM` constructor accepts `use_dcn`, `num_cross_layers`, `dcn_rank` (backward-compatible defaults)
- `TrainConfig` supports DCN parameters from YAML config
- `configs/ranking.yaml` includes commented-out DCN parameters with Pinterest references

## [0.3.0] - 2026-05-24

### Added
- **GitHub Actions CI** with two parallel jobs: lint/typecheck (ruff + mypy) and tests with coverage
- **Pre-commit hooks** via `astral-sh/ruff-pre-commit` — auto-fixes lint and enforces format on commit
- **Test coverage** with `pytest-cov` — branch coverage across all 5 packages, XML report for CI artifacts
- **mypy configuration** with `ignore_missing_imports` for third-party ML libraries, `ignore_errors` for generated protobuf files
- Package metadata (description, authors, license) to all 6 `pyproject.toml` files

### Changed
- Applied `ruff format` across entire codebase (18 files reformatted, no logic changes)
- Excluded generated protobuf files (`*_pb2.py`, `*_pb2_grpc.py`) from ruff lint/format
- Fixed E741 ambiguous variable names and E501 line-length violations in tests

## [0.2.0] - 2026-05-24

### Added
- **MLflow experiment tracking** for both retrieval and ranking training — logs params, per-epoch metrics, and ONNX artifacts
- **Checkpoint/resume** for both training scripts — saves full training state (model, optimizer, scheduler, epoch, best metric) and resumes via `--resume-from`
- **Richer retrieval evaluation**: `_eval_retrieval` returns Recall@K, NDCG@K, MRR@K (was Recall@K only)
- **Richer ranking evaluation**: `_eval_ranking` returns val_loss, CTR AUC, CVR AUC, CTR log-loss, CVR log-loss (was val_loss only)
- `compute_auc()` in ranking metrics — global AUC with edge-case handling for single-class labels
- Retrieval metrics module (`recsys_retrieval.metrics`): `hit_rate_at_k`, `mrr_at_k`, `ndcg_at_k`
- YAML config loading via `load_config()` in `recsys_common.config` — reads YAML files with CLI override support
- Example config files: `configs/retrieval.yaml`, `configs/ranking.yaml`
- `--config` flag on `train_retrieval.py` and `train_ranking.py` scripts
- `pyyaml>=6.0` dependency in recsys-common, `mlflow>=2.12` in root dev-dependencies

## [0.1.1] - 2026-05-23

### Fixed
- `torch.load` missing `map_location=device` in retrieval and ranking training — caused failures when loading GPU-trained models on CPU
- ONNX export functions now run under `torch.no_grad()` — prevents unnecessary computation graph allocation
- `np.log(0)` in popularity loading — zero-count items now treated as frequency 1 instead of producing `-inf`
- `postprocess()` no longer mutates the caller's candidate dicts in-place
- Removed unused `compute_log_loss` import from ranking training

### Changed
- Extracted `pad_and_mask()` helper into `recsys_common.transforms` — replaces duplicated padding/masking logic in retrieval dataset, ranking dataset, and serving pipeline
- Extracted `load_user_map()` and `load_item_map()` into `recsys_common.data_utils` — replaces duplicated parquet loading in both dataset classes
- Removed verbose docstrings from `recsys_common.transforms` (function signatures are self-documenting)
- CLAUDE.md: fixed references to nonexistent stub implementations, now points to Protocol definitions
- README/CLAUDE.md: removed hardcoded test counts

## [0.1.0] - 2026-05-22

### Added
- **recsys-common**: Pydantic feature schemas, stateless transforms (log_compress, z_score, quantile_bucket), VocabStore, FeatureProcessor with zero online/offline skew guarantee
- **recsys-data**: Synthetic data generation (configurable users, items, interactions, categories) and dataset compilation with frozen vocab/stats/popularity artifacts
- **recsys-retrieval**: Two-Tower model (UserTower + ItemTower), InfoNCE loss with LogQ popularity correction, FaissIndex for MIPS, training loop with cosine warmup schedule and ONNX export
- **recsys-ranking**: MMoE-DeepFM with 3 shared expert MLPs, per-task softmax gates, parallel FM component, dual CTR+CVR heads, gradient masking for CVR loss, ONNX export
- **recsys-serving**: gRPC service, RecommendationPipeline with ONNX Runtime inference, SqliteFeatureStore, Expected Value scoring (CTR x CVR x price), post-processing (dedup, diversity cap, business multipliers)
- CLI scripts: generate_data, train_retrieval, train_ranking, build_index, serve
- Integration tests: end-to-end pipeline validation, online/offline skew detection
- README with Mermaid architecture diagrams
- CLAUDE.md with project conventions and workflows
