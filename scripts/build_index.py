import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from recsys_common.processor import FeatureProcessor
from recsys_common.schemas import RawItemFeatures
from recsys_retrieval.index import FaissIndex

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Build Faiss vector index from item embeddings")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/compiled"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/models/retrieval"))
    args = parser.parse_args()

    processor = FeatureProcessor(args.artifacts_dir)
    item_tower = ort.InferenceSession(str(args.model_dir / "item_tower.onnx"))

    items = pq.read_table(args.artifacts_dir.parent / "raw" / "items.parquet")
    item_ids = items.column("item_id").to_pylist()

    logging.info(f"Processing {len(item_ids)} items...")
    vectors = []
    batch_size = 512
    for start in range(0, len(item_ids), batch_size):
        end = min(start + batch_size, len(item_ids))
        batch_items = []
        for i in range(start, end):
            item = RawItemFeatures(
                item_id=items.column("item_id")[i].as_py(),
                item_category=items.column("item_category")[i].as_py(),
                item_historical_ctr=items.column("item_historical_ctr")[i].as_py(),
                item_price=items.column("item_price")[i].as_py(),
            )
            batch_items.append(processor.process_item(item))

        inputs = {
            "item_id_idx": np.array([pi.item_id_idx for pi in batch_items], dtype=np.int64),
            "category_idx": np.array([pi.category_idx for pi in batch_items], dtype=np.int64),
            "ctr_scaled": np.array([pi.ctr_scaled for pi in batch_items], dtype=np.float32),
            "price_bucket": np.array([pi.price_bucket for pi in batch_items], dtype=np.int64),
        }
        vecs = item_tower.run(None, inputs)[0]
        vectors.append(vecs)

    all_vectors = np.vstack(vectors).astype(np.float32)
    logging.info(f"Built {all_vectors.shape[0]} vectors of dim {all_vectors.shape[1]}")

    index = FaissIndex(all_vectors.shape[1])
    index.build(all_vectors, item_ids)
    index_path = args.model_dir / "faiss_index"
    index.save(index_path)
    logging.info(f"Faiss index saved to {index_path}")


if __name__ == "__main__":
    main()
