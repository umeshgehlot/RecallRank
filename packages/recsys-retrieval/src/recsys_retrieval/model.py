from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev, dim))
        layers.append(nn.ReLU())
        prev = dim
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    def __init__(
        self,
        user_id_vocab_size: int,
        item_id_vocab_size: int,
        category_vocab_size: int,
        device_type_vocab_size: int,
        embedding_dim: int,
        mlp_dims: list[int],
    ) -> None:
        super().__init__()
        self.user_id_emb = nn.Embedding(user_id_vocab_size, embedding_dim, padding_idx=0)
        self.item_emb_for_session = nn.Embedding(item_id_vocab_size, embedding_dim, padding_idx=0)
        self.category_emb = nn.Embedding(category_vocab_size, embedding_dim, padding_idx=0)
        self.device_type_emb = nn.Embedding(device_type_vocab_size, embedding_dim, padding_idx=0)
        mlp_input_dim = embedding_dim * 4 + 2
        self.mlp = _build_mlp(mlp_input_dim, mlp_dims)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_emb = self.user_id_emb(batch["user_id_idx"])
        cat_emb = self.category_emb(batch["historic_category_indices"])
        cat_mask = batch["historic_category_mask"].unsqueeze(-1).float()
        cat_pooled = (cat_emb * cat_mask).sum(dim=1) / cat_mask.sum(dim=1).clamp(min=1)
        sess_emb = self.item_emb_for_session(batch["session_click_indices"])
        sess_mask = batch["session_click_mask"].unsqueeze(-1).float()
        sess_pooled = (sess_emb * sess_mask).sum(dim=1) / sess_mask.sum(dim=1).clamp(min=1)
        device_emb = self.device_type_emb(batch["device_type_idx"])
        numericals = torch.stack([batch["user_age_scaled"], batch["hour_of_day_scaled"]], dim=1)
        combined = torch.cat([user_emb, cat_pooled, sess_pooled, device_emb, numericals], dim=1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=1)


class ItemTower(nn.Module):
    def __init__(
        self,
        item_id_vocab_size: int,
        category_vocab_size: int,
        embedding_dim: int,
        mlp_dims: list[int],
        price_buckets: int,
    ) -> None:
        super().__init__()
        self.item_id_emb = nn.Embedding(item_id_vocab_size, embedding_dim, padding_idx=0)
        self.category_emb = nn.Embedding(category_vocab_size, embedding_dim, padding_idx=0)
        self.price_bucket_emb = nn.Embedding(price_buckets, embedding_dim)
        mlp_input_dim = embedding_dim * 3 + 1
        self.mlp = _build_mlp(mlp_input_dim, mlp_dims)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item_emb = self.item_id_emb(batch["item_id_idx"])
        cat_emb = self.category_emb(batch["category_idx"])
        price_emb = self.price_bucket_emb(batch["price_bucket"])
        ctr = batch["ctr_scaled"].unsqueeze(1)
        combined = torch.cat([item_emb, cat_emb, price_emb, ctr], dim=1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_tower: UserTower, item_tower: ItemTower) -> None:
        super().__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(
        self, user_batch: dict[str, torch.Tensor], item_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.user_tower(user_batch)
        v = self.item_tower(item_batch)
        return u, v
