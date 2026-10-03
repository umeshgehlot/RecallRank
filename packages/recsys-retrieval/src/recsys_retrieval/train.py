from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from recsys_retrieval.dataset import RetrievalDataset, retrieval_collate_fn
from recsys_retrieval.index import FaissIndex
from recsys_retrieval.loss import info_nce_loss_with_logq
from recsys_retrieval.metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k
from recsys_retrieval.model import ItemTower, TwoTowerModel, UserTower

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    eval_top_k: int = 100
    mlflow_tracking_uri: str | None = None
    resume_from: Path | None = None


def train_retrieval(
    artifacts_dir: Path,
    output_dir: Path,
    config: TrainConfig | None = None,
) -> dict[str, float]:
    cfg = config or TrainConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    train_ds = RetrievalDataset(artifacts_dir, split="train")
    val_ds = RetrievalDataset(artifacts_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=retrieval_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=retrieval_collate_fn,
    )

    processor = train_ds._processor
    model = TwoTowerModel(
        user_tower=UserTower(
            user_id_vocab_size=processor.user_id_vocab_size,
            item_id_vocab_size=processor.item_id_vocab_size,
            category_vocab_size=processor.category_vocab_size,
            device_type_vocab_size=processor.device_type_vocab_size,
            embedding_dim=64,
            mlp_dims=[256, 128, 64],
        ),
        item_tower=ItemTower(
            item_id_vocab_size=processor.item_id_vocab_size,
            category_vocab_size=processor.category_vocab_size,
            embedding_dim=64,
            mlp_dims=[256, 128, 64],
            price_buckets=10,
        ),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=min(cfg.warmup_steps, total_steps),
    )
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - cfg.warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[cfg.warmup_steps])

    start_epoch = 0
    best_recall = 0.0
    best_metrics: dict[str, float] = {}

    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt["best_recall"]
        logger.info(f"Resumed from checkpoint epoch {ckpt['epoch']}, best_recall={best_recall:.4f}")

    with mlflow.start_run(run_name="retrieval"):
        params = dataclasses.asdict(cfg)
        if params.get("mlflow_tracking_uri"):
            parsed = urlparse(params["mlflow_tracking_uri"])
            if parsed.username or parsed.password:
                redacted = parsed._replace(netloc=parsed.hostname or "")
                params["mlflow_tracking_uri"] = urlunparse(redacted)
        mlflow.log_params(params)

        for epoch in range(start_epoch, cfg.epochs):
            model.train()
            total_loss = 0.0
            for user_batch, item_batch, log_q in train_loader:
                user_batch = {k: v.to(device) for k, v in user_batch.items()}
                item_batch = {k: v.to(device) for k, v in item_batch.items()}
                log_q = log_q.to(device)

                u, v = model(user_batch, item_batch)
                loss = info_nce_loss_with_logq(u, v, log_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(train_loader), 1)
            logger.info(f"Epoch {epoch + 1}/{cfg.epochs} — Loss: {avg_loss:.4f}")
            mlflow.log_metrics({"train_loss": avg_loss}, step=epoch)

            metrics = _eval_retrieval(model, val_loader, device, cfg.eval_top_k)
            mlflow.log_metrics(metrics, step=epoch)
            recall = metrics[f"recall_at_{cfg.eval_top_k}"]
            logger.info(f"Epoch {epoch + 1}/{cfg.epochs} — Recall@{cfg.eval_top_k}: {recall:.4f}")

            if recall > best_recall:
                best_recall = recall
                best_metrics = metrics
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_recall": best_recall,
                }
                torch.save(checkpoint, output_dir / "checkpoint.pt")

        best_path = output_dir / "best_model.pt"
        model.load_state_dict(torch.load(best_path, weights_only=True, map_location=device))
        _export_onnx(model, output_dir, device)

        mlflow.log_artifact(str(output_dir / "user_tower.onnx"))
        mlflow.log_artifact(str(output_dir / "item_tower.onnx"))

    return best_metrics


def _eval_retrieval(
    model: TwoTowerModel,
    val_loader: DataLoader,
    device: torch.device,
    top_k: int,
) -> dict[str, float]:
    model.eval()
    all_user_vecs = []
    all_item_vecs = []
    with torch.no_grad():
        for user_batch, item_batch, _ in val_loader:
            user_batch = {k: v.to(device) for k, v in user_batch.items()}
            item_batch = {k: v.to(device) for k, v in item_batch.items()}
            u, v = model(user_batch, item_batch)
            all_user_vecs.append(u.cpu())
            all_item_vecs.append(v.cpu())

    user_vecs = torch.cat(all_user_vecs, dim=0).numpy()
    item_vecs = torch.cat(all_item_vecs, dim=0).numpy()

    index = FaissIndex(item_vecs.shape[1])
    ids = list(range(len(item_vecs)))
    index.build(item_vecs.astype(np.float32), [str(i) for i in ids])

    total_recall = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    n = len(user_vecs)

    for i in range(n):
        results = index.query(user_vecs[i].astype(np.float32), top_k=top_k)
        ranked_list = [int(r[0]) for r in results]
        total_recall += hit_rate_at_k(i, ranked_list)
        total_mrr += mrr_at_k(i, ranked_list)
        total_ndcg += ndcg_at_k(i, ranked_list)

    denom = max(n, 1)
    return {
        f"recall_at_{top_k}": total_recall / denom,
        f"ndcg_at_{top_k}": total_ndcg / denom,
        f"mrr_at_{top_k}": total_mrr / denom,
    }


def _export_onnx(model: TwoTowerModel, output_dir: Path, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        _export_onnx_inner(model, output_dir, device)


def _export_onnx_inner(model: TwoTowerModel, output_dir: Path, device: torch.device) -> None:
    user_dummy = {
        "user_id_idx": torch.tensor([1], device=device),
        "user_age_scaled": torch.tensor([0.0], device=device),
        "historic_category_indices": torch.zeros(1, 5, dtype=torch.long, device=device),
        "historic_category_mask": torch.ones(1, 5, dtype=torch.long, device=device),
        "session_click_indices": torch.zeros(1, 3, dtype=torch.long, device=device),
        "session_click_mask": torch.ones(1, 3, dtype=torch.long, device=device),
        "device_type_idx": torch.tensor([1], device=device),
        "hour_of_day_scaled": torch.tensor([0.0], device=device),
    }
    item_dummy = {
        "item_id_idx": torch.tensor([1], device=device),
        "category_idx": torch.tensor([1], device=device),
        "ctr_scaled": torch.tensor([0.0], device=device),
        "price_bucket": torch.tensor([5], device=device),
    }

    class UserTowerWrapper(nn.Module):
        def __init__(self, tower):
            super().__init__()
            self.tower = tower

        def forward(
            self,
            user_id_idx,
            user_age_scaled,
            historic_category_indices,
            historic_category_mask,
            session_click_indices,
            session_click_mask,
            device_type_idx,
            hour_of_day_scaled,
        ):
            batch = {
                "user_id_idx": user_id_idx,
                "user_age_scaled": user_age_scaled,
                "historic_category_indices": historic_category_indices,
                "historic_category_mask": historic_category_mask,
                "session_click_indices": session_click_indices,
                "session_click_mask": session_click_mask,
                "device_type_idx": device_type_idx,
                "hour_of_day_scaled": hour_of_day_scaled,
            }
            return self.tower(batch)

    class ItemTowerWrapper(nn.Module):
        def __init__(self, tower):
            super().__init__()
            self.tower = tower

        def forward(self, item_id_idx, category_idx, ctr_scaled, price_bucket):
            batch = {
                "item_id_idx": item_id_idx,
                "category_idx": category_idx,
                "ctr_scaled": ctr_scaled,
                "price_bucket": price_bucket,
            }
            return self.tower(batch)

    user_wrapper = UserTowerWrapper(model.user_tower)
    torch.onnx.export(
        user_wrapper,
        tuple(user_dummy.values()),
        str(output_dir / "user_tower.onnx"),
        input_names=list(user_dummy.keys()),
        output_names=["user_vector"],
        dynamic_axes={k: {0: "batch"} for k in user_dummy},
    )

    item_wrapper = ItemTowerWrapper(model.item_tower)
    torch.onnx.export(
        item_wrapper,
        tuple(item_dummy.values()),
        str(output_dir / "item_tower.onnx"),
        input_names=list(item_dummy.keys()),
        output_names=["item_vector"],
        dynamic_axes={k: {0: "batch"} for k in item_dummy},
    )
