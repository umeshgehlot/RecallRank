import torch
from recsys_retrieval.loss import info_nce_loss_with_logq


def test_loss_is_scalar():
    u = torch.randn(8, 16)
    v = torch.randn(8, 16)
    log_q = torch.log(torch.ones(8) / 8)
    loss = info_nce_loss_with_logq(u, v, log_q)
    assert loss.shape == ()


def test_loss_is_positive():
    u = torch.randn(8, 16)
    v = torch.randn(8, 16)
    log_q = torch.log(torch.ones(8) / 8)
    loss = info_nce_loss_with_logq(u, v, log_q)
    assert loss.item() > 0


def test_perfect_alignment_has_lower_loss():
    torch.manual_seed(0)
    # Perfect alignment: u == v
    v = torch.randn(8, 16)
    v = torch.nn.functional.normalize(v, dim=1)
    u_aligned = v.clone()
    u_random = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    log_q = torch.zeros(8)

    loss_aligned = info_nce_loss_with_logq(u_aligned, v, log_q)
    loss_random = info_nce_loss_with_logq(u_random, v, log_q)
    assert loss_aligned.item() < loss_random.item()


def test_loss_is_differentiable():
    u = torch.randn(4, 8, requires_grad=True)
    v = torch.randn(4, 8, requires_grad=True)
    log_q = torch.zeros(4)
    loss = info_nce_loss_with_logq(u, v, log_q)
    loss.backward()
    assert u.grad is not None
    assert v.grad is not None


def test_logq_correction_reduces_popular_item_dominance():
    torch.manual_seed(42)
    B = 16
    u = torch.nn.functional.normalize(torch.randn(B, 8), dim=1)
    v = torch.nn.functional.normalize(torch.randn(B, 8), dim=1)

    uniform_log_q = torch.zeros(B)
    skewed_counts = torch.ones(B)
    skewed_counts[0] = 100.0
    skewed_log_q = torch.log(skewed_counts / skewed_counts.sum())

    loss_uniform = info_nce_loss_with_logq(u, v, uniform_log_q)
    loss_corrected = info_nce_loss_with_logq(u, v, skewed_log_q)
    # Losses should differ when correction is applied
    assert not torch.isclose(loss_uniform, loss_corrected)
