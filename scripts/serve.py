import argparse
import logging
from pathlib import Path

from recsys_serving.service import serve

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Start gRPC recommendation server")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/compiled"))
    parser.add_argument(
        "--retrieval-model-dir", type=Path, default=Path("artifacts/models/retrieval")
    )
    parser.add_argument("--ranking-model-dir", type=Path, default=Path("artifacts/models/ranking"))
    parser.add_argument("--db-path", type=Path, default=Path("artifacts/features.db"))
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    serve(
        artifacts_dir=args.artifacts_dir,
        retrieval_model_dir=args.retrieval_model_dir,
        ranking_model_dir=args.ranking_model_dir,
        db_path=args.db_path,
        port=args.port,
    )


if __name__ == "__main__":
    main()
