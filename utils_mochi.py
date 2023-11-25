from typing import Optional

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor


@jaxtyped
@beartype
def mixing_hardest_negative(
    q: Float[Tensor, "b dim"],
    n: Float[Tensor, "queue_dim dim"],
    s: int,
    tau: float,
    N: int,
    tilde_Q: Optional[Float[Tensor, "b K"]] = None,
) -> tuple[Float[Tensor, "b s dim"], Float[Tensor, "b N dim"]]:
    batch_size = q.shape[0]
    assert N <= n.shape[0], f"N={N} must be smaller than the queue size={n.shape[0]}"
    assert s <= N, f"s={s} must be smaller than N={N}"

    # find the N hardest for each query
    tilde_Q = tilde_Q if tilde_Q is not None else (q @ n.T)
    _, indices = torch.topk(tilde_Q.clone().detach(), dim=-1, k=N)
    tilde_Q_N = n[indices]

    total_couple_of_indices = batch_size * s
    all_indices = torch.cat(
        [torch.randperm(N)[:2].unsqueeze(0) for _ in range(total_couple_of_indices)]
    ).reshape([batch_size, s, 2])

    alpha = torch.rand([batch_size, s], device=q.device).unsqueeze(-1)
    N_i = tilde_Q_N.reshape(batch_size * N, -1)[all_indices[:, :, 0]].clone().detach()
    N_j = tilde_Q_N.reshape(batch_size * N, -1)[all_indices[:, :, 1]].clone().detach()
    tilde_h = alpha * N_i + (1 - alpha) * N_j
    H = F.normalize(tilde_h, dim=-1)
    return H.clone().detach(), tilde_Q_N


@jaxtyped
@beartype
def mixing_for_even_harder_negatives(
    q: Float[Tensor, "b dim"], tilde_Q_N: Float[Tensor, "b N dim"], s_prime: int
) -> Float[Tensor, "b s_prime dim"]:
    batch_size = q.shape[0]
    N = tilde_Q_N.shape[1]
    assert s_prime <= N, f"s_prime={s_prime} must be smaller than N={N}"

    total_indices = batch_size * s_prime
    all_indices = torch.cat(
        [torch.randperm(N)[:1] for _ in range(total_indices)]
    ).reshape([batch_size, s_prime])

    beta = torch.rand([batch_size, s_prime], device=q.device).unsqueeze(-1) / 2
    N_j = tilde_Q_N.reshape(batch_size * N, -1)[all_indices].clone().detach()
    tilde_h_prime = (
        beta * q.unsqueeze(1).repeat(1, s_prime, 1).clone().detach() + (1 - beta) * N_j
    )
    H_prime = F.normalize(tilde_h_prime, dim=-1, p=2)

    return H_prime.clone().detach()
