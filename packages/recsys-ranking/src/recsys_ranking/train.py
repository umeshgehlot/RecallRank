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
from torch.utils.data import DataLoader

from recsys_ranking.dataset import RankingDataset, ranking_collate_fn
from recsys_ranking.metrics import compute_auc, compute_log_loss
from recsys_ranking.model import MMoEDeepFM

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    alpha_ctr: float = 1.0
    alpha_cvr: float = 1.0
    use_dcn: bool = False
    num_cross_layers: int = 2
    dcn_rank: int = 0
    mlflow_tracking_uri: str | None = None
    resume_from: Path | None = None


def train_ranking(
    artifacts_dir: Path,
    output_dir: Path,
    config: TrainConfig | None = None,
) -> dict[str, float]:
    cfg = config or TrainConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    train_ds = RankingDataset(artifacts_dir, split="train")
    val_ds = RankingDataset(artifacts_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=ranking_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=ranking_collate_fn,
    )

    processor = train_ds._processor
    model = MMoEDeepFM(
        user_id_vocab_size=processor.user_id_vocab_size,
        item_id_vocab_size=processor.item_id_vocab_size,
        category_vocab_size=processor.category_vocab_size,
        device_type_vocab_size=processor.device_type_vocab_size,
        embedding_dim=64,
        price_buckets=10,
        use_dcn=cfg.use_dcn,
        num_cross_layers=cfg.num_cross_layers,
        dcn_rank=cfg.dcn_rank,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ctr_loss_fn = nn.BCEWithLogitsLoss()
    cvr_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_metrics: dict[str, float] = {}
    patience_counter = 0
    start_epoch = 0

    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    with mlflow.start_run(run_name="ranking"):
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
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                clicked = batch.pop("clicked")
                converted = batch.pop("converted")

                ctr_logits, cvr_logits = model(batch)

                ctr_loss = ctr_loss_fn(ctr_logits, clicked)
                cvr_per_sample = cvr_loss_fn(cvr_logits, converted)
                cvr_loss = (cvr_per_sample * clicked).sum() / clicked.sum().clamp(min=1)

                loss = cfg.alpha_ctr * ctr_loss + cfg.alpha_cvr * cvr_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train = total_loss / max(len(train_loader), 1)

            val_metrics = _eval_ranking(model, val_loader, ctr_loss_fn, cvr_loss_fn, device, cfg)
            val_loss = val_metrics["val_loss"]
            logger.info(
                f"Epoch {epoch + 1}/{cfg.epochs} — Train: {avg_train:.4f}, Val: {val_loss:.4f}"
            )

            mlflow.log_metrics({"train_loss": avg_train}, step=epoch)
            mlflow.log_metrics(val_metrics, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                }
                torch.save(checkpoint, output_dir / "checkpoint.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        best_path = output_dir / "best_model.pt"
        model.load_state_dict(torch.load(best_path, weights_only=True, map_location=device))
        _export_onnx(model, output_dir, device)
        mlflow.log_artifact(str(output_dir / "mmoe_deepfm.onnx"))

    return best_metrics


def _eval_ranking(
    model: MMoEDeepFM,
    loader: DataLoader,
    ctr_loss_fn: nn.Module,
    cvr_loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    count = 0
    all_ctr_logits: list[np.ndarray] = []
    all_cvr_logits: list[np.ndarray] = []
    all_clicked: list[np.ndarray] = []
    all_converted: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            clicked = batch.pop("clicked")
            converted = batch.pop("converted")

            ctr_logits, cvr_logits = model(batch)

            ctr_loss = ctr_loss_fn(ctr_logits, clicked)
            cvr_per_sample = cvr_loss_fn(cvr_logits, converted)
            cvr_loss = (cvr_per_sample * clicked).sum() / clicked.sum().clamp(min=1)

            loss = cfg.alpha_ctr * ctr_loss + cfg.alpha_cvr * cvr_loss
            total += loss.item()
            count += 1

            all_ctr_logits.append(ctr_logits.cpu().numpy())
            all_cvr_logits.append(cvr_logits.cpu().numpy())
            all_clicked.append(clicked.cpu().numpy())
            all_converted.append(converted.cpu().numpy())

    val_loss = total / max(count, 1)

    ctr_logits_np = np.concatenate(all_ctr_logits)
    cvr_logits_np = np.concatenate(all_cvr_logits)
    clicked_np = np.concatenate(all_clicked)
    converted_np = np.concatenate(all_converted)

    ctr_probs = 1.0 / (1.0 + np.exp(-ctr_logits_np))
    cvr_probs = 1.0 / (1.0 + np.exp(-cvr_logits_np))

    ctr_auc = compute_auc(clicked_np, ctr_probs)
    ctr_log_loss = compute_log_loss(clicked_np, ctr_probs)

    click_mask = clicked_np == 1
    if click_mask.sum() > 0:
        cvr_auc = compute_auc(converted_np[click_mask], cvr_probs[click_mask])
        cvr_log_loss = compute_log_loss(converted_np[click_mask], cvr_probs[click_mask])
    else:
        cvr_auc = 0.0
        cvr_log_loss = 0.0

    return {
        "val_loss": val_loss,
        "ctr_auc": ctr_auc,
        "cvr_auc": cvr_auc,
        "ctr_log_loss": ctr_log_loss,
        "cvr_log_loss": cvr_log_loss,
    }


def _export_onnx(model: MMoEDeepFM, output_dir: Path, device: torch.device) -> None:
    model.eval()
    with torch.no_grad():
        _export_onnx_inner(model, output_dir, device)


def _export_onnx_inner(model: MMoEDeepFM, output_dir: Path, device: torch.device) -> None:
    dummy = {
        "user_id_idx": torch.tensor([1], device=device),
        "item_id_idx": torch.tensor([1], device=device),
        "category_idx": torch.tensor([1], device=device),
        "device_type_idx": torch.tensor([1], device=device),
        "price_bucket": torch.tensor([5], device=device),
        "user_age_scaled": torch.tensor([0.0], device=device),
        "hour_of_day_scaled": torch.tensor([0.0], device=device),
        "ctr_scaled": torch.tensor([0.0], device=device),
        "historic_category_indices": torch.zeros(1, 5, dtype=torch.long, device=device),
        "historic_category_mask": torch.ones(1, 5, dtype=torch.long, device=device),
        "session_click_indices": torch.zeros(1, 3, dtype=torch.long, device=device),
        "session_click_mask": torch.ones(1, 3, dtype=torch.long, device=device),
    }

    class MMoEWrapper(nn.Module):
        def __init__(self, mmoe: MMoEDeepFM):
            super().__init__()
            self.mmoe = mmoe

        def forward(
            self,
            user_id_idx,
            item_id_idx,
            category_idx,
            device_type_idx,
            price_bucket,
            user_age_scaled,
            hour_of_day_scaled,
            ctr_scaled,
            historic_category_indices,
            historic_category_mask,
            session_click_indices,
            session_click_mask,
        ):
            batch = {
                "user_id_idx": user_id_idx,
                "item_id_idx": item_id_idx,
                "category_idx": category_idx,
                "device_type_idx": device_type_idx,
                "price_bucket": price_bucket,
                "user_age_scaled": user_age_scaled,
                "hour_of_day_scaled": hour_of_day_scaled,
                "ctr_scaled": ctr_scaled,
                "historic_category_indices": historic_category_indices,
                "historic_category_mask": historic_category_mask,
                "session_click_indices": session_click_indices,
                "session_click_mask": session_click_mask,
            }
            ctr_logits, cvr_logits = self.mmoe(batch)
            return ctr_logits, cvr_logits

    wrapper = MMoEWrapper(model)
    torch.onnx.export(
        wrapper,
        tuple(dummy.values()),
        str(output_dir / "mmoe_deepfm.onnx"),
        input_names=list(dummy.keys()),
        output_names=["ctr_logits", "cvr_logits"],
        dynamic_axes={k: {0: "batch"} for k in dummy},
    )
