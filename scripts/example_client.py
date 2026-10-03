import argparse

import grpc
from recsys_serving import recsys_pb2, recsys_pb2_grpc


def main():
    parser = argparse.ArgumentParser(description="Send a recommendation request to the gRPC server")
    parser.add_argument("--user-id", type=str, default="user_42")
    parser.add_argument("--device-type", type=str, default="mobile")
    parser.add_argument("--hour", type=int, default=14)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--retrieval-k", type=int, default=100)
    parser.add_argument("--host", type=str, default="localhost:50051")
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.host)
    stub = recsys_pb2_grpc.RecommendationServiceStub(channel)

    response = stub.Recommend(
        recsys_pb2.RecommendRequest(
            user_id=args.user_id,
            device_type=args.device_type,
            hour_of_day=args.hour,
            top_n=args.top_n,
            retrieval_k=args.retrieval_k,
        )
    )

    print(f"Recommendations for {args.user_id}:")
    for item in response.items:
        print(
            f"  {item.rank}. {item.item_id}"
            f"  score={item.score:.4f}"
            f"  ctr={item.ctr_score:.4f}"
            f"  cvr={item.cvr_score:.4f}"
        )


if __name__ == "__main__":
    main()
