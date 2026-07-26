import pytest

torch = pytest.importorskip("torch")

from inference.distillation.train import _CpuGradientAccumulator


def test_flush_repopulates_grads_and_allows_repeated_steps():
    """Minibatch stepping needs flush() to hand grads back without dropping hooks."""
    linear = torch.nn.Linear(4, 2, bias=False)
    accumulator = _CpuGradientAccumulator([("weight", linear.weight)])
    optimizer = torch.optim.SGD([linear.weight], lr=0.1)
    before = linear.weight.detach().clone()

    for _ in range(2):
        for _ in range(2):
            linear(torch.ones(1, 4)).sum().backward()
        # The hook parks gradients on CPU and clears .grad as they arrive.
        assert linear.weight.grad is None
        accumulator.flush()
        assert linear.weight.grad is not None
        assert torch.allclose(linear.weight.grad, torch.full((2, 4), 2.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    accumulator.restore()
    moved = (linear.weight.detach() - before).abs().max().item()
    assert moved == pytest.approx(0.4, abs=1e-5)


def test_restore_removes_hooks():
    linear = torch.nn.Linear(3, 1, bias=False)
    accumulator = _CpuGradientAccumulator([("weight", linear.weight)])
    accumulator.restore()
    linear(torch.ones(1, 3)).sum().backward()
    # With hooks gone, gradients stay on the parameter as usual.
    assert linear.weight.grad is not None
