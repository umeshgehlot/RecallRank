from __future__ import annotations

import logging
from concurrent import futures
from pathlib import Path

import grpc

from recsys_serving import recsys_pb2, recsys_pb2_grpc
from recsys_serving.feature_store import SqliteFeatureStore
from recsys_serving.pipeline import RecommendationPipeline

logger = logging.getLogger(__name__)


MAX_TOP_N = 100
MAX_RETRIEVAL_K = 900


class RecommendationServicer(recsys_pb2_grpc.RecommendationServiceServicer):
    def __init__(self, pipeline: RecommendationPipeline) -> None:
        self._pipeline = pipeline

    def Recommend(self, request, context):
        top_n = min(request.top_n if request.top_n > 0 else 10, MAX_TOP_N)
        retrieval_k = min(request.retrieval_k if request.retrieval_k > 0 else 200, MAX_RETRIEVAL_K)

        results = self._pipeline.recommend(
            user_id=request.user_id,
            device_type=request.device_type,
            hour_of_day=request.hour_of_day,
            top_n=top_n,
            retrieval_k=retrieval_k,
        )

        items = [
            recsys_pb2.RecommendedItem(
                item_id=r["item_id"],
                score=r["score"],
                rank=r["rank"],
                ctr_score=r["ctr_score"],
                cvr_score=r["cvr_score"],
            )
            for r in results
        ]
        return recsys_pb2.RecommendResponse(items=items)


def serve(
    artifacts_dir: Path,
    retrieval_model_dir: Path,
    ranking_model_dir: Path,
    db_path: Path,
    port: int = 50051,
) -> None:
    feature_store = SqliteFeatureStore(db_path)
    pipeline = RecommendationPipeline(
        artifacts_dir=artifacts_dir,
        retrieval_model_dir=retrieval_model_dir,
        ranking_model_dir=ranking_model_dir,
        feature_store=feature_store,
    )
    servicer = RecommendationServicer(pipeline)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    recsys_pb2_grpc.add_RecommendationServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"gRPC server started on port {port}")
    server.wait_for_termination()
