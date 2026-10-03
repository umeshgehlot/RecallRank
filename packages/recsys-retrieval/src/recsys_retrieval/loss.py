import torch
import torch.nn.functional as F


def info_nce_loss_with_logq(
    user_vectors: torch.Tensor,
    item_vectors: torch.Tensor,
    log_q: torch.Tensor,
) -> torch.Tensor:
    sim_matrix = user_vectors @ item_vectors.T
    corrected = sim_matrix - log_q.unsqueeze(0)
    labels = torch.arange(user_vectors.size(0), device=user_vectors.device)
    return F.cross_entropy(corrected, labels)
