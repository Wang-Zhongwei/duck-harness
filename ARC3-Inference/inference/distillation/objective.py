from __future__ import annotations

import torch


def token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Select target-token log probabilities from aligned logits."""
    if logits.ndim != 3 or target_ids.ndim != 2:
        raise ValueError(
            "expected logits [batch, seq, vocab] and target_ids [batch, seq]"
        )
    if logits.shape[:2] != target_ids.shape:
        raise ValueError(
            "logits and target_ids must have matching batch and sequence dimensions"
        )
    return logits.log_softmax(dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def chosen_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Return log p(x[t] | x[:t]) for every token after the first."""
    if logits.shape[:2] != token_ids.shape:
        raise ValueError(
            "logits and token_ids must have the same batch and sequence dimensions"
        )
    return token_logprobs(logits[:, :-1], token_ids[:, 1:])


def reverse_cumsum(
    values: torch.Tensor, mask: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Reward-to-go for a right-padded token reward tensor."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have identical shapes")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1]")
    running = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
    result = torch.zeros_like(values)
    for index in range(values.shape[1] - 1, -1, -1):
        active = mask[:, index].to(values.dtype)
        running = values[:, index] * active + gamma * running
        running = running * active
        result[:, index] = running
    return result


def reverse_kl_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    max_abs_advantage: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Score-function estimator for KL(student || teacher) on student rollouts.

    The sampled reverse-KL cost is ``old_logprobs - teacher_logprobs``.
    Its negative is the token reward. Reward-to-go accounts for an earlier
    sampled token changing the distribution of all later tokens.
    """
    if not (
        new_logprobs.shape == old_logprobs.shape == teacher_logprobs.shape == mask.shape
    ):
        raise ValueError("all token tensors must have identical shapes")
    active = mask.to(new_logprobs.dtype)
    token_reward = (teacher_logprobs - old_logprobs).detach() * active
    advantages = reverse_cumsum(token_reward, mask, gamma=gamma)
    if max_abs_advantage is not None:
        advantages = advantages.clamp(min=-max_abs_advantage, max=max_abs_advantage)
    active_tokens = active.sum().clamp_min(1)
    loss = -(new_logprobs * advantages.detach() * active).sum(dim=1).mean()
    reverse_kl_per_sequence = (
        ((old_logprobs - teacher_logprobs) * active).sum(dim=1).mean()
    )
    reverse_kl_per_token = (
        (old_logprobs - teacher_logprobs) * active
    ).sum() / active_tokens
    metrics = {
        "reverse_kl_sample": reverse_kl_per_sequence.detach(),
        "reverse_kl_per_token": reverse_kl_per_token.detach(),
        "reward": (-reverse_kl_per_sequence).detach(),
        "mean_advantage": (advantages * active).sum().detach() / active_tokens,
        "tokens": active.sum().detach(),
    }
    return loss, metrics
