# model.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Should be consistent with dataloader.py
DEFAULT_NUM_DEGRADATION_TYPES = 6


def build_dct_matrix(size: int) -> torch.Tensor:
    """
    Build orthonormal DCT-II transform matrix.

    For an input patch X, 2D DCT is:
        D = C @ X @ C.T
    """
    mat = torch.zeros(size, size, dtype=torch.float32)

    for k in range(size):
        for n in range(size):
            if k == 0:
                alpha = math.sqrt(1.0 / size)
            else:
                alpha = math.sqrt(2.0 / size)
            mat[k, n] = alpha * math.cos(math.pi * (n + 0.5) * k / size)

    return mat


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def build_initial_frequency_masks(
    num_bands: int,
    patch_size: int,
    low_radius: float = 0.18,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build soft band-pass initialization masks.

    Returns:
        init_masks: [B, P, P], values in (0, 1)
        low_region: [P, P], binary low-frequency region indicator
    """
    coords = torch.arange(patch_size, dtype=torch.float32)
    u, v = torch.meshgrid(coords, coords, indexing="ij")

    # Normalize frequency radius to [0, 1].
    radius = torch.sqrt(u**2 + v**2)
    radius = radius / (math.sqrt(2.0) * (patch_size - 1) + 1e-6)

    low_region = (radius <= low_radius).float()

    if num_bands == 1:
        centers = [0.65]
        widths = [0.35]
    elif num_bands == 2:
        centers = [0.35, 0.70]
        widths = [0.22, 0.25]
    elif num_bands == 3:
        centers = [0.28, 0.52, 0.78]
        widths = [0.18, 0.20, 0.22]
    else:
        # For B >= 4, use several coarse frequency bands plus adaptive high bands.
        centers = torch.linspace(0.25, 0.85, steps=num_bands).tolist()
        widths = torch.linspace(0.18, 0.28, steps=num_bands).tolist()

    masks = []
    for c, w in zip(centers, widths):
        mask = torch.exp(-0.5 * ((radius - c) / w) ** 2)

        # Suppress the DC / very-low-frequency region at initialization.
        mask = mask * (1.0 - 0.75 * low_region)
        mask = mask.clamp(0.02, 0.98)
        masks.append(mask)

    init_masks = torch.stack(masks, dim=0)  # [B, P, P]
    return init_masks, low_region


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ArtifactQueryBlock(nn.Module):
    """
    A cross-attention block that lets learnable artifact queries retrieve
    suspicious patch-frequency candidates.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries:    [B, Nq, D]
            candidates: [B, K, D]
        """
        q = self.q_norm(queries)
        kv = self.kv_norm(candidates)

        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            need_weights=False,
        )

        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_norm(queries))

        return queries


class PropagationAwareFaVA(nn.Module):
    """
    Propagation-Aware Frequency Artifact Tokenizer.

    Input:
        frames: [B, V, T, 3, H, W] or [B, T, 3, H, W]

    Output:
        {
            "tokens":               [B, V, N, D] or [B, N, D],
            "type_logits":          [B, V, C] or [B, C],
            "severity_score":       [B, V] or [B],
            "candidate_score_logits":[B, V, T, L, num_bands] or ...,
            "selected_indices":     [B, V, top_k] or ...,
            "low_frequency_penalty": scalar,
            "mask_diversity_loss":  scalar,
        }
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        in_channels: int = 3,
        num_bands: int = 4,
        token_dim: int = 256,
        num_artifact_tokens: int = 128,
        num_query_blocks: int = 2,
        num_heads: int = 8,
        top_k: int = 1024,
        candidate_hidden_dim: int = 256,
        head_hidden_dim: int = 256,
        num_degradation_types: int = DEFAULT_NUM_DEGRADATION_TYPES,
        dropout: float = 0.1,
        use_ycbcr: bool = True,
        peak_selection: str = "topk",
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size, "
                f"got image_size={image_size}, patch_size={patch_size}."
            )

        if peak_selection not in {"topk", "soft"}:
            raise ValueError("peak_selection must be either 'topk' or 'soft'.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.num_bands = num_bands
        self.token_dim = token_dim
        self.num_artifact_tokens = num_artifact_tokens
        self.top_k = top_k
        self.use_ycbcr = use_ycbcr
        self.peak_selection = peak_selection

        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size
        self.num_patches = self.grid_h * self.grid_w

        # DCT matrix.
        dct = build_dct_matrix(patch_size)
        self.register_buffer("dct_matrix", dct, persistent=False)

        # Learnable frequency masks.
        init_masks, low_region = build_initial_frequency_masks(
            num_bands=num_bands,
            patch_size=patch_size,
        )
        self.raw_frequency_masks = nn.Parameter(inverse_sigmoid(init_masks))
        self.register_buffer(
            "low_region",
            low_region.reshape(1, -1),
            persistent=False,
        )

        # Candidate feature statistics:
        # signed mean, absolute mean, log energy, std-like variance.
        self.num_stats = 4
        candidate_in_dim = in_channels * self.num_stats

        self.candidate_encoder = MLP(
            in_dim=candidate_in_dim,
            hidden_dim=candidate_hidden_dim,
            out_dim=token_dim,
            dropout=dropout,
        )

        self.temporal_embed = nn.Embedding(num_frames, token_dim)
        self.spatial_embed = nn.Embedding(self.num_patches, token_dim)
        self.band_embed = nn.Embedding(num_bands, token_dim)

        self.artifactness_scorer = MLP(
            in_dim=token_dim,
            hidden_dim=head_hidden_dim,
            out_dim=1,
            dropout=dropout,
        )

        self.artifact_queries = nn.Parameter(
            torch.randn(num_artifact_tokens, token_dim) * 0.02
        )

        self.query_blocks = nn.ModuleList(
            [
                ArtifactQueryBlock(
                    dim=token_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                )
                for _ in range(num_query_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(token_dim)

        # Pretraining heads.
        self.type_head = MLP(
            in_dim=token_dim,
            hidden_dim=head_hidden_dim,
            out_dim=num_degradation_types,
            dropout=dropout,
        )

        self.severity_head = MLP(
            in_dim=token_dim,
            hidden_dim=head_hidden_dim,
            out_dim=1,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames:
                [B, V, T, 3, H, W] from dataloader.py
                or [B, T, 3, H, W]

        Returns:
            A dict of model outputs.
        """
        original_dim = frames.dim()

        if original_dim == 6:
            bsz, num_variants, t, c, h, w = frames.shape
            flat_frames = frames.reshape(bsz * num_variants, t, c, h, w)
            restore_shape = (bsz, num_variants)
        elif original_dim == 5:
            bsz, t, c, h, w = frames.shape
            flat_frames = frames
            restore_shape = None
        else:
            raise ValueError(
                "frames must have shape [B, V, T, 3, H, W] "
                "or [B, T, 3, H, W]."
            )

        if t != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, but got {t}."
            )

        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, but got {c}."
            )

        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}x{self.image_size}, "
                f"but got {h}x{w}."
            )

        candidate_tokens, candidate_score_logits = self._encode_candidates(
            flat_frames
        )

        selected_candidates, selected_indices = self._select_candidates(
            candidate_tokens=candidate_tokens,
            candidate_score_logits=candidate_score_logits,
        )

        tokens = self._query_tokenize(selected_candidates)

        pooled = tokens.mean(dim=1)

        type_logits = self.type_head(pooled)
        severity_score = self.severity_head(pooled).squeeze(-1)
        outputs: Dict[str, torch.Tensor] = {
            "tokens": tokens,
            "type_logits": type_logits,
            "severity_score": severity_score,
            "candidate_score_logits": candidate_score_logits,
            "selected_indices": selected_indices,
            "low_frequency_penalty": self.low_frequency_penalty(),
            "mask_diversity_loss": self.frequency_mask_diversity_loss(),
        }

        if restore_shape is not None:
            bsz, num_variants = restore_shape
            outputs = self._restore_variant_shape(outputs, bsz, num_variants)

        return outputs

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------

    def _encode_candidates(
        self,
        frames: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert video frames into patch-frequency artifact candidates.

        Args:
            frames: [B, T, 3, H, W]

        Returns:
            candidate_tokens:       [B, T*L*num_bands, D]
            candidate_score_logits: [B, T*L*num_bands]
        """
        bsz, t, c, h, w = frames.shape

        if self.use_ycbcr:
            frames = self._rgb_to_ycbcr(frames)

        patches = self._extract_patches(frames)  # [B*T, L, C, P, P]
        dct_patches = self._apply_dct(patches)   # [B*T, L, C, P, P]

        l_count = dct_patches.shape[1]
        p = self.patch_size

        dct_flat = dct_patches.reshape(
            bsz,
            t,
            l_count,
            c,
            p * p,
        )  # [B, T, L, C, F]

        spectral_features = self._build_spectral_features(
            dct_flat
        )  # [B, T*L*num_bands, C*num_stats]

        candidate_tokens = self.candidate_encoder(spectral_features)
        candidate_tokens = candidate_tokens + self._build_position_embeddings(
            device=frames.device,
            dtype=candidate_tokens.dtype,
        ).unsqueeze(0)

        candidate_score_logits = self.artifactness_scorer(
            candidate_tokens
        ).squeeze(-1)

        return candidate_tokens, candidate_score_logits

    def _extract_patches(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, C, H, W]

        Returns:
            patches: [B*T, L, C, P, P]
        """
        bsz, t, c, h, w = frames.shape

        x = frames.reshape(bsz * t, c, h, w)

        patches = F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )  # [B*T, C*P*P, L]

        patches = patches.transpose(1, 2).contiguous()
        patches = patches.reshape(
            bsz * t,
            self.num_patches,
            c,
            self.patch_size,
            self.patch_size,
        )

        return patches

    def _apply_dct(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: [B*T, L, C, P, P]

        Returns:
            dct_patches: [B*T, L, C, P, P]
        """
        cmat = self.dct_matrix.to(dtype=patches.dtype, device=patches.device)

        # DCT = C @ X @ C.T
        dct = torch.einsum(
            "im,blcmn,jn->blcij",
            cmat,
            patches,
            cmat,
        )

        return dct

    def _build_spectral_features(self, dct_flat: torch.Tensor) -> torch.Tensor:
        """
        Compute compact spectral statistics for each patch-frequency band.

        Args:
            dct_flat: [B, T, L, C, F]

        Returns:
            features: [B, T*L*num_bands, C*num_stats]
        """
        bsz, t, l_count, c, f_dim = dct_flat.shape

        masks = torch.sigmoid(self.raw_frequency_masks)
        masks = masks.reshape(self.num_bands, f_dim)
        masks = masks.to(dtype=dct_flat.dtype, device=dct_flat.device)

        denom = masks.sum(dim=-1).clamp_min(1e-6)  # [num_bands]

        abs_dct = dct_flat.abs()
        sq_dct = dct_flat.pow(2)

        # [B, T, L, C, num_bands]
        signed_mean = torch.einsum(
            "ntpcf,bf->ntpcb",
            dct_flat,
            masks,
        ) / denom.view(1, 1, 1, 1, -1)

        abs_mean = torch.einsum(
            "ntpcf,bf->ntpcb",
            abs_dct,
            masks,
        ) / denom.view(1, 1, 1, 1, -1)

        energy = torch.einsum(
            "ntpcf,bf->ntpcb",
            sq_dct,
            masks,
        ) / denom.view(1, 1, 1, 1, -1)

        variance = (energy - signed_mean.pow(2)).clamp_min(0.0)

        log_energy = torch.log1p(energy)
        std_like = torch.sqrt(variance + 1e-6)

        # [B, T, L, C, num_bands, num_stats]
        features = torch.stack(
            [signed_mean, abs_mean, log_energy, std_like],
            dim=-1,
        )

        # [B, T, L, num_bands, C, num_stats]
        features = features.permute(0, 1, 2, 4, 3, 5).contiguous()

        # [B, T*L*num_bands, C*num_stats]
        features = features.reshape(
            bsz,
            t * l_count * self.num_bands,
            c * self.num_stats,
        )

        return features

    def _build_position_embeddings(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns:
            pos: [T*L*num_bands, D]
        """
        t_ids = torch.arange(self.num_frames, device=device)
        p_ids = torch.arange(self.num_patches, device=device)
        b_ids = torch.arange(self.num_bands, device=device)

        t_emb = self.temporal_embed(t_ids).to(dtype=dtype)
        p_emb = self.spatial_embed(p_ids).to(dtype=dtype)
        b_emb = self.band_embed(b_ids).to(dtype=dtype)

        pos = (
            t_emb[:, None, None, :]
            + p_emb[None, :, None, :]
            + b_emb[None, None, :, :]
        )  # [T, L, B, D]

        pos = pos.reshape(
            self.num_frames * self.num_patches * self.num_bands,
            self.token_dim,
        )

        return pos

    # ------------------------------------------------------------------
    # Peak-aware selection and query tokenizer
    # ------------------------------------------------------------------

    def _select_candidates(
        self,
        candidate_tokens: torch.Tensor,
        candidate_score_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            candidate_tokens:       [B, M, D]
            candidate_score_logits: [B, M]

        Returns:
            selected_candidates: [B, K, D]
            selected_indices:    [B, K]
        """
        bsz, m_count, dim = candidate_tokens.shape

        if self.peak_selection == "soft":
            weights = torch.sigmoid(candidate_score_logits).unsqueeze(-1)
            selected_candidates = candidate_tokens * weights
            selected_indices = torch.arange(
                m_count,
                device=candidate_tokens.device,
            ).unsqueeze(0).repeat(bsz, 1)
            return selected_candidates, selected_indices

        k = min(self.top_k, m_count)

        selected_scores, selected_indices = torch.topk(
            candidate_score_logits,
            k=k,
            dim=1,
            largest=True,
            sorted=False,
        )

        gather_idx = selected_indices.unsqueeze(-1).expand(-1, -1, dim)
        selected_candidates = torch.gather(
            candidate_tokens,
            dim=1,
            index=gather_idx,
        )

        # Keep gradients to the artifactness scorer for selected candidates.
        selected_weights = torch.sigmoid(selected_scores).unsqueeze(-1)
        selected_candidates = selected_candidates * selected_weights

        return selected_candidates, selected_indices

    def _query_tokenize(self, selected_candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            selected_candidates: [B, K, D]

        Returns:
            artifact_tokens: [B, N, D]
        """
        bsz = selected_candidates.shape[0]

        queries = self.artifact_queries.unsqueeze(0).expand(bsz, -1, -1)

        for block in self.query_blocks:
            queries = block(queries, selected_candidates)

        tokens = self.final_norm(queries)
        return tokens

    # ------------------------------------------------------------------
    # Utility losses for masks / token diversity
    # ------------------------------------------------------------------

    def low_frequency_penalty(self) -> torch.Tensor:
        """
        Penalize excessive attention to very low-frequency coefficients.
        """
        masks = torch.sigmoid(self.raw_frequency_masks)
        masks = masks.reshape(self.num_bands, -1)

        low_region = self.low_region.to(device=masks.device, dtype=masks.dtype)

        penalty = (masks * low_region).sum(dim=-1) / low_region.sum().clamp_min(1.0)
        return penalty.mean()

    def frequency_mask_diversity_loss(self) -> torch.Tensor:
        """
        Encourage different frequency masks to cover complementary regions.
        """
        masks = torch.sigmoid(self.raw_frequency_masks)
        masks = masks.reshape(self.num_bands, -1)

        masks = F.normalize(masks, dim=-1)
        sim = masks @ masks.t()

        eye = torch.eye(
            self.num_bands,
            device=masks.device,
            dtype=masks.dtype,
        )

        off_diag = sim - eye
        return off_diag.pow(2).sum() / max(1, self.num_bands * (self.num_bands - 1))

    @staticmethod
    def token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
        """
        Encourage artifact query tokens to be non-collapsed.

        Args:
            tokens: [B, V, N, D] or [B, N, D]
        """
        if tokens.dim() == 4:
            bsz, num_variants, n_tokens, dim = tokens.shape
            tokens = tokens.reshape(bsz * num_variants, n_tokens, dim)
        elif tokens.dim() == 3:
            pass
        else:
            raise ValueError("tokens must be [B, V, N, D] or [B, N, D].")

        tokens = F.normalize(tokens, dim=-1)
        sim = tokens @ tokens.transpose(1, 2)

        n_tokens = sim.shape[-1]
        eye = torch.eye(
            n_tokens,
            device=tokens.device,
            dtype=tokens.dtype,
        ).unsqueeze(0)

        loss = (sim - eye).pow(2).mean()
        return loss

    # ------------------------------------------------------------------
    # Shape and color helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rgb_to_ycbcr(frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, 3, H, W], RGB in [0, 1]

        Returns:
            ycbcr: [B, T, 3, H, W]
        """
        r = frames[:, :, 0:1]
        g = frames[:, :, 1:2]
        b = frames[:, :, 2:3]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

        return torch.cat([y, cb, cr], dim=2)

    @staticmethod
    def _restore_variant_shape(
        outputs: Dict[str, torch.Tensor],
        bsz: int,
        num_variants: int,
    ) -> Dict[str, torch.Tensor]:
        restored: Dict[str, torch.Tensor] = {}

        for key, value in outputs.items():
            if not torch.is_tensor(value):
                restored[key] = value
                continue

            # Scalars such as low_frequency_penalty.
            if value.dim() == 0:
                restored[key] = value
                continue

            if value.shape[0] == bsz * num_variants:
                restored[key] = value.reshape(bsz, num_variants, *value.shape[1:])
            else:
                restored[key] = value

        return restored


def compute_fava_type_rank_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    type_weight: float = 1.0,
    rank_weight: float = 0.3,
    low_weight: float = 0.05,
    mask_div_weight: float = 0.0,
    token_div_weight: float = 0.0,
    rank_margin: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Compute type-rank losses for Propagation-Aware FaVA.

    Expected batch fields from dataloader.py:
        type_labels:     [B, V, C]
        severity:        [B, V]
    This public training path uses only degradation-type classification,
    severity ranking, and frequency/token regularization.
    """
    device = outputs["type_logits"].device

    type_labels = batch["type_labels"].to(device=device, dtype=torch.float32)
    severity_labels = batch["severity"].to(device=device, dtype=torch.float32)

    type_logits = outputs["type_logits"]
    severity_score = outputs["severity_score"]

    type_loss = F.binary_cross_entropy_with_logits(
        type_logits,
        type_labels,
    )

    rank_loss = pairwise_ranking_loss(
        scores=severity_score,
        labels=severity_labels,
        margin=rank_margin,
    )

    low_loss = outputs.get(
        "low_frequency_penalty",
        torch.zeros((), device=device),
    )

    mask_div_loss = outputs.get(
        "mask_diversity_loss",
        torch.zeros((), device=device),
    )

    token_div_loss = torch.zeros((), device=device)
    if token_div_weight > 0.0:
        token_div_loss = PropagationAwareFaVA.token_diversity_loss(
            outputs["tokens"]
        )

    total = (
        type_weight * type_loss
        + rank_weight * rank_loss
        + low_weight * low_loss
        + mask_div_weight * mask_div_loss
        + token_div_weight * token_div_loss
    )

    return {
        "loss": total,
        "type_loss": type_loss.detach(),
        "rank_loss": rank_loss.detach(),
        "low_loss": low_loss.detach(),
        "mask_div_loss": mask_div_loss.detach(),
        "token_div_loss": token_div_loss.detach(),
    }


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Pairwise ranking loss.

    Args:
        scores: [B, V]
        labels: [B, V], larger means heavier degradation.

    For each pair i, j:
        if label_j > label_i, require score_j > score_i + margin.
    """
    if scores.dim() != 2 or labels.dim() != 2:
        raise ValueError("scores and labels should both be [B, V].")

    device = scores.device

    labels_i = labels.unsqueeze(2)  # [B, V, 1]
    labels_j = labels.unsqueeze(1)  # [B, 1, V]

    scores_i = scores.unsqueeze(2)
    scores_j = scores.unsqueeze(1)

    pair_mask = labels_j > labels_i
    score_diff = scores_j - scores_i

    if pair_mask.sum() == 0:
        return torch.zeros((), device=device)

    losses = F.relu(margin - score_diff)
    return losses[pair_mask].mean()
