from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayer(nn.Module):
    """DCN cross-layer: x_{l+1} = x_0 * (W * x_l + b) + x_l"""

    def __init__(self, in_features: int, rank: int = 0) -> None:
        super().__init__()
        self.rank = rank
        if rank == 0:
            self.linear = nn.Linear(in_features, in_features)
        else:
            self.U = nn.Linear(in_features, rank, bias=False)
            self.V = nn.Linear(rank, in_features)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return x0 * self.linear(xl) + xl  # type: ignore[no-any-return]
        return x0 * self.V(self.U(xl)) + xl  # type: ignore[no-any-return]


class DCNExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_cross_layers: int,
        deep_dims: list[int],
        dropout: float,
        rank: int = 0,
    ) -> None:
        super().__init__()
        self.cross_layers = nn.ModuleList(
            [CrossLayer(in_features, rank) for _ in range(num_cross_layers)]
        )
        layers: list[nn.Module] = []
        prev = in_features
        for dim in deep_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ELU())
            layers.append(nn.Dropout(dropout))
            prev = dim
        self.deep = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        xl = x
        for cross in self.cross_layers:
            xl = cross(x0, xl)
        return self.deep(xl)  # type: ignore[no-any-return]


class MMoEDeepFM(nn.Module):
    def __init__(
        self,
        user_id_vocab_size: int,
        item_id_vocab_size: int,
        category_vocab_size: int,
        device_type_vocab_size: int,
        embedding_dim: int,
        price_buckets: int,
        num_experts: int = 3,
        expert_dims: list[int] | None = None,
        tower_dim: int = 64,
        expert_dropout: float = 0.1,
        max_historic_categories: int = 5,
        max_session_clicks: int = 3,
        use_dcn: bool = False,
        num_cross_layers: int = 2,
        dcn_rank: int = 0,
    ) -> None:
        super().__init__()
        if expert_dims is None:
            expert_dims = [256, 128]

        self.embedding_dim = embedding_dim
        self.max_historic_categories = max_historic_categories
        self.max_session_clicks = max_session_clicks

        self.user_id_emb = nn.Embedding(user_id_vocab_size, embedding_dim, padding_idx=0)
        self.item_id_emb = nn.Embedding(item_id_vocab_size, embedding_dim, padding_idx=0)
        self.category_emb = nn.Embedding(category_vocab_size, embedding_dim, padding_idx=0)
        self.device_type_emb = nn.Embedding(device_type_vocab_size, embedding_dim, padding_idx=0)
        self.price_bucket_emb = nn.Embedding(price_buckets, embedding_dim)
        self.session_item_emb = nn.Embedding(item_id_vocab_size, embedding_dim, padding_idx=0)
        self.hist_category_emb = nn.Embedding(category_vocab_size, embedding_dim, padding_idx=0)

        num_categorical = 5
        num_numerical = 3
        num_pooled = 2

        self.fm_first_order_weights = nn.Parameter(
            torch.zeros(
                num_categorical * embedding_dim + num_pooled * embedding_dim + num_numerical
            )
        )
        self.fm_bias = nn.Parameter(torch.zeros(1))

        concat_dim = (num_categorical + num_pooled) * embedding_dim + num_numerical

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            if use_dcn:
                self.experts.append(
                    DCNExpert(concat_dim, num_cross_layers, expert_dims, expert_dropout, dcn_rank)
                )
            else:
                layers: list[nn.Module] = []
                prev = concat_dim
                for dim in expert_dims:
                    layers.append(nn.Linear(prev, dim))
                    layers.append(nn.ELU())
                    layers.append(nn.Dropout(expert_dropout))
                    prev = dim
                self.experts.append(nn.Sequential(*layers))

        expert_out_dim = expert_dims[-1]

        self.gate_ctr = nn.Linear(concat_dim, num_experts)
        self.gate_cvr = nn.Linear(concat_dim, num_experts)

        self.tower_ctr = nn.Sequential(
            nn.Linear(expert_out_dim, tower_dim),
            nn.ELU(),
            nn.Linear(tower_dim, 1),
        )
        self.tower_cvr = nn.Sequential(
            nn.Linear(expert_out_dim, tower_dim),
            nn.ELU(),
            nn.Linear(tower_dim, 1),
        )

        self._fm_field_count = num_categorical + num_pooled

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_id_emb(batch["user_id_idx"])
        item_emb = self.item_id_emb(batch["item_id_idx"])
        cat_emb = self.category_emb(batch["category_idx"])
        dev_emb = self.device_type_emb(batch["device_type_idx"])
        price_emb = self.price_bucket_emb(batch["price_bucket"])

        hist_emb = self.hist_category_emb(batch["historic_category_indices"])
        hist_mask = batch["historic_category_mask"].unsqueeze(-1).float()
        hist_pooled = (hist_emb * hist_mask).sum(dim=1) / hist_mask.sum(dim=1).clamp(min=1)

        sess_emb = self.session_item_emb(batch["session_click_indices"])
        sess_mask = batch["session_click_mask"].unsqueeze(-1).float()
        sess_pooled = (sess_emb * sess_mask).sum(dim=1) / sess_mask.sum(dim=1).clamp(min=1)

        numericals = torch.stack(
            [batch["user_age_scaled"], batch["hour_of_day_scaled"], batch["ctr_scaled"]], dim=1
        )

        all_field_embs = torch.stack(
            [user_emb, item_emb, cat_emb, dev_emb, price_emb, hist_pooled, sess_pooled], dim=1
        )

        flat_embs = all_field_embs.reshape(all_field_embs.size(0), -1)
        fm_input = torch.cat([flat_embs, numericals], dim=1)

        y_1st = self.fm_bias + (fm_input * self.fm_first_order_weights).sum(dim=1)

        sum_of_embs = all_field_embs.sum(dim=1)
        sum_of_squares = (all_field_embs**2).sum(dim=1)
        square_of_sums = sum_of_embs**2
        y_fm = 0.5 * (square_of_sums - sum_of_squares).sum(dim=1)

        x = torch.cat([flat_embs, numericals], dim=1)

        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)

        gate_ctr_weights = F.softmax(self.gate_ctr(x), dim=1)
        gate_cvr_weights = F.softmax(self.gate_cvr(x), dim=1)

        ctr_expert_mix = (gate_ctr_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        cvr_expert_mix = (gate_cvr_weights.unsqueeze(-1) * expert_outs).sum(dim=1)

        ctr_tower_out = self.tower_ctr(ctr_expert_mix).squeeze(-1)
        cvr_tower_out = self.tower_cvr(cvr_expert_mix).squeeze(-1)

        ctr_logits = y_1st + y_fm + ctr_tower_out
        cvr_logits = y_1st + y_fm + cvr_tower_out

        return ctr_logits, cvr_logits
