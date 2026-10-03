import argparse
import logging
from pathlib import Path

from recsys_data.compile import compile_dataset
from recsys_data.synthetic import SyntheticConfig, generate_synthetic_data

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data and compile dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--num-users", type=int, default=10_000)
    parser.add_argument("--num-items", type=int, default=50_000)
    parser.add_argument("--num-interactions", type=int, default=500_000)
    parser.add_argument("--num-categories", type=int, default=20)
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw"
    compiled_dir = args.output_dir / "compiled"

    cfg = SyntheticConfig(
        num_users=args.num_users,
        num_items=args.num_items,
        num_interactions=args.num_interactions,
        num_categories=args.num_categories,
    )

    logging.info("Generating synthetic data...")
    generate_synthetic_data(cfg, raw_dir)
    logging.info(f"Raw data written to {raw_dir}")

    logging.info("Compiling dataset...")
    compile_dataset(raw_dir, compiled_dir)
    logging.info(f"Compiled data written to {compiled_dir}")


if __name__ == "__main__":
    main()
