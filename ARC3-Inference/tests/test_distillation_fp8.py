import pytest

torch = pytest.importorskip("torch")

from inference.distillation.fp8 import StaticW8A8Linear  # noqa: E402


def test_static_fp8_layer_propagates_input_gradients() -> None:
    source = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    layer = StaticW8A8Linear(source)
    scale = torch.tensor([0.01], dtype=torch.bfloat16)
    layer.weight_scale.copy_(scale)
    layer.input_scale.copy_(scale)
    layer.weight.copy_(
        (source.weight.detach() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    inputs = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)

    layer(inputs).float().sum().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert layer.weight.grad is None
