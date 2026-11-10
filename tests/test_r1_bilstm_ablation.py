"""Smoke tests for the R1 recurrent-layer ablation (no external datasets required)."""
import torch
from ablations.models import CNNWNNBiLSTMAM
from paper_reproduction.models import PaperCNNWNNWBiLSTMAM


def test_r1_changes_only_recurrent_block():
    r0 = PaperCNNWNNWBiLSTMAM()
    r1 = CNNWNNBiLSTMAM()
    assert isinstance(r1.wbilstm, torch.nn.LSTM)
    assert not isinstance(r0.wbilstm, torch.nn.LSTM)
    assert r1.conv.weight.shape == r0.conv.weight.shape
    assert r1.wnn_projection.weight.shape == r0.wnn_projection.weight.shape
    assert r1.attention.scorer[0].weight.shape == r0.attention.scorer[0].weight.shape


def test_r1_forward_and_backward():
    torch.manual_seed(7)
    model = CNNWNNBiLSTMAM()
    x = torch.randn(3, 64, 4)
    pred, weights = model(x, return_attention=True)
    assert pred.shape == (3,)
    assert weights.shape == (3, 32)
    assert torch.isfinite(pred).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), atol=1e-5, rtol=1e-5)
    loss = ((pred - torch.tensor([0.7, 0.8, 0.9])) ** 2).mean()
    loss.backward()
    assert model.wbilstm.weight_ih_l0.grad is not None
    assert torch.isfinite(model.wbilstm.weight_ih_l0.grad).all()
