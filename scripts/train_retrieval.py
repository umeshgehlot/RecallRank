import argparse
import logging
from pathlib import Path

from recsys_common.config import load_config
from recsys_retrieval.train import TrainConfig, train_retrieval

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Train Two-Tower retrieval model")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/compiled"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models/retrieval"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    if args.config:
        cli_overrides = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        }
        overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        cfg_dict = load_config(args.config, overrides)
        fields = TrainConfig.__dataclass_fields__
        cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
    else:
        cfg = TrainConfig(
            **({"epochs": args.epochs} if args.epochs else {}),
            **({"batch_size": args.batch_size} if args.batch_size else {}),
            **({"lr": args.lr} if args.lr else {}),
        )

    metrics = train_retrieval(args.artifacts_dir, args.output_dir, cfg)
    logging.info(f"Training complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
