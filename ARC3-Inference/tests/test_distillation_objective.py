import pytest

torch = pytest.importorskip("torch")

objective = pytest.importorskip("inference.distillation.objective")
chosen_token_logprobs = objective.chosen_token_logprobs
reverse_cumsum = objective.reverse_cumsum
reverse_kl_policy_loss = objective.reverse_kl_policy_loss
train = pytest.importorskip("inference.distillation.train")
processed_logprobs = train._processed_logprobs
CpuGradientAccumulator = train._CpuGradientAccumulator


def test_chosen_token_logprobs_selects_shifted_targets():
    logits = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]])
    token_ids = torch.tensor([[0, 1, 2]])
    actual = chosen_token_logprobs(logits, token_ids)
    expected = (
        torch.log_softmax(logits[:, :-1], dim=-1)
        .gather(-1, token_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.allclose(actual, expected)


def test_reverse_cumsum_respects_padding():
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    result = reverse_cumsum(values, mask)
    assert torch.equal(result, torch.tensor([[6.0, 5.0, 3.0], [9.0, 5.0, 0.0]]))


def test_policy_loss_pushes_toward_higher_teacher_probability():
    new = torch.tensor([[-1.0]], requires_grad=True)
    old = new.detach().clone()
    teacher = torch.tensor([[-0.5]])
    mask = torch.tensor([[True]])
    loss, metrics = reverse_kl_policy_loss(new, old, teacher, mask)
    loss.backward()
    assert new.grad.item() < 0.0
    assert metrics["reverse_kl_sample"].item() == pytest.approx(-0.5)
    assert metrics["reward"].item() == pytest.approx(0.5)


def test_policy_loss_averages_sequence_objective_not_sampled_token_count():
    new = torch.tensor([[-1.0, -1.0], [-1.0, 0.0]], requires_grad=True)
    old = new.detach().clone()
    teacher = torch.tensor([[-0.5, -0.5], [-0.5, 0.0]])
    mask = torch.tensor([[True, True], [True, False]])
    loss, metrics = reverse_kl_policy_loss(new, old, teacher, mask)
    assert loss.item() == pytest.approx(1.0)
    assert metrics["reverse_kl_sample"].item() == pytest.approx(-0.75)
    assert metrics["reverse_kl_per_token"].item() == pytest.approx(-0.5)


def test_processed_logprobs_preserve_sampled_action_at_truncation_boundary():
    logits = torch.tensor([[[10.0, 9.0, 1.0]]], requires_grad=True)
    targets = torch.tensor([[2]])

    sampled = processed_logprobs(
        logits,
        targets,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
    )

    assert torch.isfinite(sampled).all()
    sampled.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_sparse_processed_logprobs_match_dense_truncation():
    logits = torch.tensor([[[3.0, 2.0, 1.0, 0.0]]])
    targets = torch.tensor([[1]])

    actual = processed_logprobs(
        logits,
        targets,
        temperature=1.0,
        top_k=3,
        top_p=0.8,
    )

    dense = logits.clone()
    cutoff = dense.topk(3, dim=-1).values[..., -1:]
    dense = dense.masked_fill(dense < cutoff, -torch.inf)
    sorted_scores, sorted_indices = dense.sort(dim=-1, descending=True)
    remove = sorted_scores.softmax(dim=-1).cumsum(dim=-1) > 0.8
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
    dense = torch.full_like(dense, -torch.inf).scatter(
        -1, sorted_indices, sorted_scores
    )
    expected = dense.log_softmax(dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)

    assert torch.allclose(actual, expected)


def test_cpu_gradient_accumulator_matches_normal_accumulation():
    reference = torch.nn.Linear(3, 2, bias=False)
    offloaded = torch.nn.Linear(3, 2, bias=False)
    offloaded.load_state_dict(reference.state_dict())
    inputs = [torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[3.0, 2.0, 1.0]])]

    for value in inputs:
        reference(value).sum().backward()

    accumulator = CpuGradientAccumulator(list(offloaded.named_parameters()))
    for value in inputs:
        offloaded(value).sum().backward()
        assert all(parameter.grad is None for parameter in offloaded.parameters())
    accumulator.restore()

    assert torch.equal(offloaded.weight.grad, reference.weight.grad)
