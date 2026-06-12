"""Geometry router for curvature-aware attention routing.

Estimates, per example, how hyperbolic-like (four-point delta condition) and
how sphere-like (constant-curvature Gram fit) the current layer's hidden
states are, and converts the scores into soft gates over three attention
operators: [Euclidean, Hyperbolic, Sphere].

Design constraints (v1):
- Scores are mini-batch / token-subsample estimates, not exact geometric
  measures; they are routing signals only.
- All geometry statistics run in fp32 (and, by default, under no_grad) so
  the router is stable in bf16 autocast and adds no gradient paths.
- No randomness: token subsampling and quadruple selection are deterministic
  so training stays reproducible.
- The score computation is wrapped with torch._dynamo.disable so
  torch.compile takes a (cheap) graph break instead of tracing eigvalsh /
  cdist / data-dependent control flow.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch._dynamo as dynamo
except Exception:  # pragma: no cover - torch always ships _dynamo, but stay safe
    dynamo = None


def validate_geometry_router_config(config) -> None:
    """Reject geometry-router settings the first version does not implement.

    No-op when the router is disabled, so baseline configs never trip this.
    """
    if not getattr(config, "geometry_router_enabled", False):
        return
    mode = getattr(config, "geometry_router_mode", "soft")
    if mode != "soft":
        raise NotImplementedError(
            f"geometry_router_mode='{mode}' is reserved; only 'soft' is implemented")
    if not getattr(config, "geometry_router_on_attention", True):
        raise NotImplementedError(
            "geometry_router_on_attention=false is not implemented: v1 routes "
            "the attention operator only, so it must stay true")
    if getattr(config, "geometry_router_on_mlp", False):
        raise NotImplementedError(
            "geometry_router_on_mlp=true is reserved; v1 does not route the MLP")
    score = getattr(config, "geometry_hyperbolic_score", "busemann_proxy")
    if score not in ("busemann_proxy", "poincare_distance"):
        raise ValueError(f"Unknown geometry_hyperbolic_score: {score}")
    score = getattr(config, "geometry_sphere_score", "cosine")
    if score not in ("cosine", "negative_angular"):
        raise ValueError(f"Unknown geometry_sphere_score: {score}")


def geometry_model_kwargs(config) -> Dict[str, object]:
    """Geometry-router kwargs for ELF_models[...](...), read off a Config.

    Uses getattr with the disabled defaults so older configs without the
    geometry fields keep working unchanged. Validates that no reserved
    (unimplemented) option is switched on.
    """
    validate_geometry_router_config(config)
    return {
        "geometry_router_enabled": bool(getattr(config, "geometry_router_enabled", False)),
        "geometry_router_layers": getattr(config, "geometry_router_layers", "all"),
        "geometry_router_denoiser_only": bool(getattr(config, "geometry_router_denoiser_only", False)),
        "geometry_router_sample_size": getattr(config, "geometry_router_sample_size", 32),
        "geometry_router_quad_samples": getattr(config, "geometry_router_quad_samples", 512),
        "geometry_router_tau_h": getattr(config, "geometry_router_tau_h", 4.0),
        "geometry_router_tau_s": getattr(config, "geometry_router_tau_s", 4.0),
        "geometry_router_bias_e": getattr(config, "geometry_router_bias_e", 2.0),
        "geometry_router_bias_h": getattr(config, "geometry_router_bias_h", -2.0),
        "geometry_router_bias_s": getattr(config, "geometry_router_bias_s", -2.0),
        "geometry_router_time_e_bias": getattr(config, "geometry_router_time_e_bias", 1.0),
        "geometry_router_time_h_bias": getattr(config, "geometry_router_time_h_bias", 0.0),
        "geometry_router_time_s_bias": getattr(config, "geometry_router_time_s_bias", 0.0),
        "geometry_router_sphere_k": getattr(config, "geometry_router_sphere_k", "0.25,0.5,1.0,2.0,4.0"),
        "geometry_router_eps": getattr(config, "geometry_router_eps", 1e-6),
        "geometry_router_detach_scores": getattr(config, "geometry_router_detach_scores", True),
        "geometry_hyperbolic_curvature": getattr(config, "geometry_hyperbolic_curvature", 1.0),
        "geometry_hyperbolic_score": getattr(config, "geometry_hyperbolic_score", "busemann_proxy"),
        "geometry_sphere_score": getattr(config, "geometry_sphere_score", "cosine"),
    }


def parse_float_list(s: str) -> List[float]:
    """Parse a comma-separated float list, e.g. "0.25,0.5,1.0"."""
    return [float(tok) for tok in s.split(",") if tok.strip()]


def parse_layer_spec(spec: str, depth: int) -> Set[int]:
    """Parse a layer spec into a set of block indices.

    Supports "all", "0,1,2", and range mixes like "0-3,6,8-11".
    """
    if spec is None or spec.strip().lower() == "all":
        return set(range(depth))
    layers: Set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"Invalid layer range '{tok}' in spec '{spec}'")
            layers.update(range(lo, hi + 1))
        else:
            layers.add(int(tok))
    bad = [i for i in layers if i < 0 or i >= depth]
    if bad:
        raise ValueError(f"Layer spec '{spec}' has indices {bad} outside [0, {depth})")
    return layers


def masked_token_subsample(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    sample_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministically subsample up to `sample_size` valid tokens per example.

    x: (B, N, C); mask: optional (B, N) with 1=valid. Returns
    (sampled (B, M, C), sampled_mask (B, M)). No RNG: rows with >= M valid
    tokens take evenly spaced valid tokens; rows with fewer repeat the last
    valid token as padding, flagged invalid in sampled_mask so statistics
    ignore the duplicates.
    """
    B, N, _ = x.shape
    M = min(sample_size, N)
    device = x.device
    if mask is None:
        mask = torch.ones((B, N), dtype=torch.float32, device=device)
    mask = mask.to(torch.float32)

    counts = mask.sum(dim=1).long()  # (B,)
    # Stable argsort puts valid token positions first, in original order.
    valid_first = torch.argsort(mask, dim=1, descending=True, stable=True)  # (B, N)

    pos = torch.arange(M, device=device)  # (M,)
    counts_safe = counts.clamp(min=1).unsqueeze(1)  # (B, 1)
    # >= M valid: evenly spaced over the valid tokens; < M valid: take each
    # valid token once then repeat the last one.
    even = (pos.unsqueeze(0) * counts_safe) // M
    padded = torch.minimum(pos.unsqueeze(0), counts_safe - 1)
    take = torch.where(counts.unsqueeze(1) >= M, even, padded).clamp(min=0)  # (B, M)

    idx = torch.gather(valid_first, 1, take)  # (B, M) positions into N
    sampled = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    sampled_mask = (pos.unsqueeze(0) < counts.unsqueeze(1)).to(torch.float32)  # (B, M)
    return sampled, sampled_mask


def pairwise_dist(x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None,
                  eps: float = 1e-6) -> torch.Tensor:
    """Diameter-normalized pairwise Euclidean distances, fp32.

    x: (B, M, C) -> (B, M, M). Distances are divided by each example's max
    valid pairwise distance so downstream curvature candidates are scale-free.
    """
    xf = x.float()
    D = torch.cdist(xf, xf, p=2)  # (B, M, M)
    if valid_mask is not None:
        pair_valid = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)  # (B, M, M)
        diam = (D * pair_valid).amax(dim=(1, 2))
    else:
        diam = D.amax(dim=(1, 2))
    diam = diam.clamp(min=eps)
    return D / (diam + eps).view(-1, 1, 1)


def _quadruple_indices(M: int, quad_samples: int, device: torch.device,
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic quadruples (a, b, c, d) plus a distinctness mask."""
    q = torch.arange(quad_samples, device=device)
    a = (q * 1 + 0) % M
    b = (q * 3 + 1) % M
    c = (q * 5 + 2) % M
    d = (q * 7 + 3) % M
    distinct = (
        (a != b) & (a != c) & (a != d) & (b != c) & (b != d) & (c != d)
    )
    return a, b, c, d, distinct


def estimate_delta_rel(D: torch.Tensor, valid_mask: torch.Tensor,
                       quad_samples: int, eps: float = 1e-6) -> torch.Tensor:
    """Relative four-point delta-hyperbolicity estimate, e_H in ~[0, 1].

    D: (B, M, M) diameter-normalized; valid_mask: (B, M). Low e_H means the
    point set is tree-like / hyperbolic-like. Uses deterministic quadruples
    (no torch.randint) so training is reproducible. Returns the neutral score
    1.0 for examples without enough valid quadruples.
    """
    B, M, _ = D.shape
    device = D.device
    if M < 4:
        return torch.ones((B,), dtype=torch.float32, device=device)

    a, b, c, d, distinct = _quadruple_indices(M, quad_samples, device)

    vm = valid_mask.to(torch.float32)
    quad_valid = (
        vm[:, a] * vm[:, b] * vm[:, c] * vm[:, d]
    ) * distinct.to(torch.float32).unsqueeze(0)  # (B, Q)

    s1 = D[:, a, b] + D[:, c, d]
    s2 = D[:, a, c] + D[:, b, d]
    s3 = D[:, a, d] + D[:, b, c]
    sums = torch.stack([s1, s2, s3], dim=-1)  # (B, Q, 3)
    sums_sorted, _ = torch.sort(sums, dim=-1)  # ascending: [S, Mid, L]
    delta_quad = (sums_sorted[..., 2] - sums_sorted[..., 1]) / 2.0  # (B, Q)

    delta = (delta_quad * quad_valid).amax(dim=1)  # (B,) invalid quads -> 0
    delta_rel = (2.0 * delta).clamp(0.0, 1.0)

    has_quads = quad_valid.sum(dim=1) > 0
    return torch.where(has_quads, delta_rel, torch.ones_like(delta_rel))


def estimate_spherical_fit(D: torch.Tensor, valid_mask: torch.Tensor,
                           k_candidates: List[float], rank_dim: int,
                           eps: float = 1e-6) -> torch.Tensor:
    """Constant-positive-curvature Gram-fit residual, e_S in ~[0, 1].

    For each candidate curvature K, builds G = (1/K) * cos(sqrt(K) * D) — the
    Gram matrix points on a radius-1/sqrt(K) sphere would produce — and scores
    how far G is from PSD-with-bounded-rank. e_S = min over feasible K. This
    is a routing signal, not a proof the hidden states lie on a sphere.
    Note: this is intentionally NOT delta-hyperbolicity; delta_rel only judges
    tree-likeness and says nothing about spherical fit.
    """
    B, M, _ = D.shape
    device = D.device
    vm = valid_mask.to(torch.float32)
    n_valid = vm.sum(dim=1)  # (B,)
    if M < 3:
        return torch.ones((B,), dtype=torch.float32, device=device)

    pair_valid = vm.unsqueeze(2) * vm.unsqueeze(1)  # (B, M, M)
    max_d = (D * pair_valid).amax(dim=(1, 2))  # (B,)
    rank_cap = min(M, rank_dim + 1)

    best = torch.full((B,), float("inf"), dtype=torch.float32, device=device)
    for K in k_candidates:
        if K <= 0:
            continue
        sqrt_k = math.sqrt(K)
        feasible = sqrt_k * max_d < (math.pi - eps)  # (B,)
        if not bool(feasible.any()):
            continue
        G = (1.0 / K) * torch.cos(sqrt_k * D)
        # Zeroing invalid rows AND columns makes G block-diag(G_valid, 0), so
        # the batched eigvalsh below yields eig(G_valid) plus exact zeros.
        G = G * pair_valid
        G = 0.5 * (G + G.transpose(1, 2))  # enforce symmetry for eigvalsh
        eigvals = torch.linalg.eigvalsh(G.float())  # (B, M), ascending

        negative_penalty = eigvals.clamp(max=0.0).abs().sum(dim=1)
        pos_desc, _ = torch.sort(eigvals.clamp(min=0.0), dim=1, descending=True)
        rank_penalty = pos_desc[:, rank_cap:].sum(dim=1) if rank_cap < M else torch.zeros_like(negative_penalty)
        total = eigvals.abs().sum(dim=1)
        e_s_k = (negative_penalty + rank_penalty) / (total + eps)

        e_s_k = torch.where(feasible, e_s_k, torch.full_like(e_s_k, float("inf")))
        best = torch.minimum(best, e_s_k)

    enough_tokens = n_valid >= 3
    ok = enough_tokens & torch.isfinite(best)
    return torch.where(ok, best.clamp(0.0, 1.0), torch.ones_like(best))


def _compute_geometry_scores(hidden: torch.Tensor,
                             geometry_mask: Optional[torch.Tensor],
                             sample_size: int, quad_samples: int,
                             k_candidates: List[float],
                             eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 e_H / e_S for a batch of hidden states. (B, N, C) -> (B,), (B,)."""
    sampled, sampled_mask = masked_token_subsample(hidden.float(), geometry_mask, sample_size)
    D = pairwise_dist(sampled, sampled_mask, eps=eps)
    e_h = estimate_delta_rel(D, sampled_mask, quad_samples, eps=eps)
    e_s = estimate_spherical_fit(D, sampled_mask, k_candidates,
                                 rank_dim=hidden.shape[-1], eps=eps)
    return e_h, e_s


if dynamo is not None:
    # Graph-break torch.compile around eigvalsh/cdist/data-dependent code.
    _compute_geometry_scores = dynamo.disable(_compute_geometry_scores)


class GeometryRouter(nn.Module):
    """Soft router over [Euclidean, Hyperbolic, Sphere] attention operators.

    Holds no learnable parameters or persistent buffers, so enabling it never
    changes the model's state_dict. Gate logits combine fixed priors
    (bias_*), the geometry scores (scaled by tau_*), and a time-dependent
    term that, with defaults, biases routing toward Euclidean at small t.
    """

    def __init__(
        self,
        sample_size: int = 32,
        quad_samples: int = 512,
        sphere_k_candidates: str = "0.25,0.5,1.0,2.0,4.0",
        tau_h: float = 4.0,
        tau_s: float = 4.0,
        bias_e: float = 2.0,
        bias_h: float = -2.0,
        bias_s: float = -2.0,
        time_e_bias: float = 1.0,
        time_h_bias: float = 0.0,
        time_s_bias: float = 0.0,
        eps: float = 1e-6,
        detach_scores: bool = True,
        log_metrics: bool = False,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.quad_samples = quad_samples
        if isinstance(sphere_k_candidates, str):
            self.sphere_k_candidates = parse_float_list(sphere_k_candidates)
        else:
            self.sphere_k_candidates = [float(k) for k in sphere_k_candidates]
        self.tau_h = tau_h
        self.tau_s = tau_s
        self.bias_e = bias_e
        self.bias_h = bias_h
        self.bias_s = bias_s
        self.time_e_bias = time_e_bias
        self.time_h_bias = time_h_bias
        self.time_s_bias = time_s_bias
        self.eps = eps
        self.detach_scores = detach_scores
        self.log_metrics = log_metrics
        # Last detached gate means, for offline inspection only (not a buffer:
        # never checkpointed, never synced).
        self.latest_gate_mean: Optional[torch.Tensor] = None

    def forward(
        self,
        hidden: torch.Tensor,
        t: torch.Tensor,
        geometry_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """hidden: (B, N, C); t: (B,); geometry_mask: optional (B, N), 1=valid.

        Returns (gates (B, 3) in hidden.dtype, scores dict of detached fp32
        tensors). Gate order is [E, H, S].
        """
        if t is None:
            raise ValueError("GeometryRouter.forward requires the diffusion time t (B,)")
        if self.detach_scores:
            with torch.no_grad():
                e_h, e_s = _compute_geometry_scores(
                    hidden.detach(), geometry_mask, self.sample_size,
                    self.quad_samples, self.sphere_k_candidates, self.eps,
                )
        else:
            e_h, e_s = _compute_geometry_scores(
                hidden, geometry_mask, self.sample_size,
                self.quad_samples, self.sphere_k_candidates, self.eps,
            )

        t_f = t.float().reshape(-1)
        l_e = self.bias_e + self.time_e_bias * (1.0 - t_f)
        l_h = self.bias_h - self.tau_h * e_h + self.time_h_bias * t_f
        l_s = self.bias_s - self.tau_s * e_s + self.time_s_bias * t_f
        logits = torch.stack([l_e, l_h, l_s], dim=-1)  # (B, 3)
        gates = F.softmax(logits, dim=-1)

        scores: Dict[str, torch.Tensor] = {
            "e_H": e_h.detach(),
            "e_S": e_s.detach(),
            "logits": logits.detach(),
            "gates": gates.detach(),
        }
        if self.log_metrics:
            self.latest_gate_mean = gates.detach().mean(dim=0)
        return gates.to(hidden.dtype), scores
