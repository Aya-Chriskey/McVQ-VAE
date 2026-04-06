import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import geoopt

def variance_scaling_(tensor: torch.Tensor,
                      scale: float = 1.0,
                      mode: str = "fan_avg",
                      distribution: str = "uniform") -> torch.Tensor:
    """Rough PyTorch equivalent of TensorFlow / Sonnet VarianceScaling."""
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    variance = scale / max(1.0, float(denom))

    with torch.no_grad():
        if distribution == "uniform":
            bound = math.sqrt(3.0 * variance)
            return tensor.uniform_(-bound, bound)
        elif distribution == "normal":
            std = math.sqrt(variance)
            return tensor.normal_(0.0, std)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        

class VectorQuantizer(nn.Module):
    """Standard VQ-VAE quantizer.

    PyTorch rewrite of the Sonnet VectorQuantizer in origin_VQ.py.
    Input shape: [B, C, H, W], where C == embedding_dim.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        variance_scaling_(self.embeddings, distribution="uniform")

    def quantize(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        # encoding_indices: [B, H, W]
        # return: [B, H, W, C]
        return F.embedding(encoding_indices, self.embeddings)

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        del is_training  # keep the same call signature as the TensorFlow version

        if inputs.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got shape {tuple(inputs.shape)}")
        if inputs.size(1) != self.embedding_dim:
            raise ValueError(
                f"Input channel dimension must equal embedding_dim={self.embedding_dim}, "
                f"but got C={inputs.size(1)}"
            )

        # NCHW -> NHWC, because the original TensorFlow code quantizes along the last dimension.
        inputs_nhwc = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = inputs_nhwc.reshape(-1, self.embedding_dim)  # [B*H*W, C]

        # Squared Euclidean distances to each codebook vector.
        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + self.embeddings.pow(2).sum(dim=1, keepdim=True).t()
        )  # [B*H*W, K]

        encoding_indices = torch.argmin(distances, dim=1)  # [B*H*W]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)

        b, h, w, _ = inputs_nhwc.shape
        encoding_indices = encoding_indices.view(b, h, w)
        quantized_nhwc = self.quantize(encoding_indices)

        e_latent_loss = F.mse_loss(quantized_nhwc.detach(), inputs_nhwc)
        q_latent_loss = F.mse_loss(quantized_nhwc, inputs_nhwc.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator.
        quantized_nhwc = inputs_nhwc + (quantized_nhwc - inputs_nhwc).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        quantized = quantized_nhwc.permute(0, 3, 1, 2).contiguous()  # back to NCHW

        return {
            "quantize": quantized,
            "loss": loss,
            "q_latent_loss": q_latent_loss,
            "e_latent_loss": e_latent_loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }


class VectorQuantizerEMA(nn.Module):
    """EMA version of the VQ-VAE quantizer.

    Logic preserved from the Sonnet VectorQuantizerEMA:
      - embeddings are buffers rather than trainable parameters,
      - codebook updates use exponential moving averages,
      - loss only contains the commitment term.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 decay: float,
                 epsilon: float = 1e-5):
        super().__init__()
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0, 1]")

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        embedding = torch.empty(num_embeddings, embedding_dim)
        variance_scaling_(embedding, distribution="uniform")

        self.register_buffer("embeddings", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_dw", embedding.clone())

    def quantize(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(encoding_indices, self.embeddings)

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        if inputs.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got shape {tuple(inputs.shape)}")
        if inputs.size(1) != self.embedding_dim:
            raise ValueError(
                f"Input channel dimension must equal embedding_dim={self.embedding_dim}, "
                f"but got C={inputs.size(1)}"
            )

        inputs_nhwc = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = inputs_nhwc.reshape(-1, self.embedding_dim)  # [N, C]

        distances = (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + self.embeddings.pow(2).sum(dim=1, keepdim=True).t()
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_inputs.dtype)

        b, h, w, _ = inputs_nhwc.shape
        encoding_indices = encoding_indices.view(b, h, w)
        quantized_nhwc = self.quantize(encoding_indices)

        e_latent_loss = F.mse_loss(quantized_nhwc.detach(), inputs_nhwc)
        loss = self.commitment_cost * e_latent_loss

        if is_training:
            with torch.no_grad():
                cluster_size = encodings.sum(dim=0)  # [K]
                dw = encodings.t() @ flat_inputs      # [K, C]

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
                self.ema_dw.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

                n = self.ema_cluster_size.sum()
                normalized_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon)
                    * n
                )

                updated_embeddings = self.ema_dw / normalized_cluster_size.unsqueeze(1)
                self.embeddings.copy_(updated_embeddings)

        quantized_nhwc = inputs_nhwc + (quantized_nhwc - inputs_nhwc).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        quantized = quantized_nhwc.permute(0, 3, 1, 2).contiguous()

        return {
            "quantize": quantized,
            "loss": loss,
            "q_latent_loss": torch.zeros((), device=inputs.device, dtype=inputs.dtype),
            "e_latent_loss": e_latent_loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }
    


class _ManifoldVectorQuantizerBase(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 manifold: geoopt.Stereographic,
                 init_scale: float = 1e-2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.manifold = manifold
        self.init_scale = init_scale

    def _validate_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got shape {tuple(inputs.shape)}")
        if inputs.size(1) != self.embedding_dim:
            raise ValueError(
                f"Input channel dimension must equal embedding_dim={self.embedding_dim}, "
                f"but got C={inputs.size(1)}"
            )
        inputs_nhwc = inputs.permute(0, 2, 3, 1).contiguous()
        flat_inputs = inputs_nhwc.reshape(-1, self.embedding_dim)
        flat_inputs = self.manifold.projx(flat_inputs)
        return flat_inputs

    def _reshape_quantized(self, quantized_flat: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        b, _, h, w = inputs.shape
        quantized_nhwc = quantized_flat.view(b, h, w, self.embedding_dim)
        quantized_nhwc = self.manifold.projx(quantized_nhwc)
        quantized_nhwc = inputs.permute(0, 2, 3, 1).contiguous() + (quantized_nhwc - inputs.permute(0, 2, 3, 1).contiguous()).detach()
        return quantized_nhwc.permute(0, 3, 1, 2).contiguous()

    def _pairwise_squared_distance(self, flat_inputs: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        # return (
        #     flat_inputs.pow(2).sum(dim=1, keepdim=True)
        #     - 2 * flat_inputs @ embeddings.t()
        #     + embeddings.pow(2).sum(dim=1, keepdim=True).t()
        # )
        x = flat_inputs.unsqueeze(1)
        y = embeddings.unsqueeze(0)
        return self.manifold.dist(x, y, dim=-1).pow(2)

    def _perplexity(self, encodings: torch.Tensor) -> torch.Tensor:
        avg_probs = encodings.mean(dim=0)
        return torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())


class McVectorQuantizer(_ManifoldVectorQuantizerBase):
    """Manifold-aware VQ using Stereographic distance and Riemannian codebook parameters."""

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 manifold: geoopt.Stereographic,
                 init_scale: float = 1e-2):
        super().__init__(embedding_dim, num_embeddings, commitment_cost, manifold, init_scale)
        k_val = manifold.k.detach() if torch.is_tensor(manifold.k) else torch.tensor(manifold.k)
        if torch.all(k_val > 0):
            init_scale = min(init_scale, float(1.0 / math.sqrt(max(embedding_dim * float(k_val.max().item()), 1e-6))))
        init_tangent = torch.randn(num_embeddings, embedding_dim) * init_scale
        init_points = manifold.projx(manifold.expmap0(init_tangent, dim=-1))
        self.embeddings = geoopt.ManifoldParameter(init_points, manifold=manifold)

    def quantize(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(encoding_indices, self.embeddings)

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        del is_training
        flat_inputs = self._validate_inputs(inputs)
        embeddings = self.manifold.projx(self.embeddings)
        distances = self._pairwise_squared_distance(flat_inputs, embeddings)

        encoding_indices_flat = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices_flat, self.num_embeddings).type(flat_inputs.dtype)
        quantized_flat = F.embedding(encoding_indices_flat, embeddings)

        q_dist2 = self.manifold.dist(quantized_flat, flat_inputs.detach(), dim=-1).pow(2)
        e_dist2 = self.manifold.dist(quantized_flat.detach(), flat_inputs, dim=-1).pow(2)
        q_latent_loss = q_dist2.mean()
        e_latent_loss = e_dist2.mean()
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        b, _, h, w = inputs.shape
        encoding_indices = encoding_indices_flat.view(b, h, w)
        quantized = self._reshape_quantized(quantized_flat, inputs)
        perplexity = self._perplexity(encodings)

        return {
            "quantize": quantized,
            "loss": loss,
            "q_latent_loss": q_latent_loss,
            "e_latent_loss": e_latent_loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }


class McVectorQuantizerEMA(_ManifoldVectorQuantizerBase):
    """EMA manifold VQ using Stereographic distances and geodesic EMA codebook updates."""

    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 manifold: geoopt.Stereographic,
                 decay: float,
                 epsilon: float = 1e-5,
                 init_scale: float = 1e-2):
        super().__init__(embedding_dim, num_embeddings, commitment_cost, manifold, init_scale)
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0, 1]")
        self.decay = decay
        self.epsilon = epsilon

        k_val = manifold.k.detach() if torch.is_tensor(manifold.k) else torch.tensor(manifold.k)
        if torch.all(k_val > 0):
            init_scale = min(init_scale, float(1.0 / math.sqrt(max(embedding_dim * float(k_val.max().item()), 1e-6))))
        init_tangent = torch.randn(num_embeddings, embedding_dim) * init_scale
        init_points = manifold.projx(manifold.expmap0(init_tangent, dim=-1))
        self.register_buffer("embeddings", init_points)
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))

    def quantize(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(encoding_indices, self.embeddings)

    def _ema_update(self, flat_inputs: torch.Tensor, encoding_indices_flat: torch.Tensor) -> None:
        with torch.no_grad():
            device = flat_inputs.device
            cluster_size = torch.bincount(encoding_indices_flat, minlength=self.num_embeddings).to(device=device, dtype=flat_inputs.dtype)
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
            updated = self.embeddings.clone()
            for idx in range(self.num_embeddings):
                mask = encoding_indices_flat == idx
                if not torch.any(mask):
                    continue
                assigned = flat_inputs[mask]
                current = self.embeddings[idx].unsqueeze(0).expand_as(assigned)
                tangent = self.manifold.logmap(current, assigned, dim=-1)
                tangent_mean = tangent.mean(dim=0, keepdim=True)
                step = (1.0 - self.decay) * tangent_mean
                updated_point = self.manifold.expmap(self.embeddings[idx].unsqueeze(0), step, dim=-1)
                updated[idx] = self.manifold.projx(updated_point.squeeze(0))
            self.embeddings.copy_(updated)

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        flat_inputs = self._validate_inputs(inputs)
        embeddings = self.manifold.projx(self.embeddings)
        distances = self._pairwise_squared_distance(flat_inputs, embeddings)

        encoding_indices_flat = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices_flat, self.num_embeddings).type(flat_inputs.dtype)
        quantized_flat = F.embedding(encoding_indices_flat, embeddings)

        e_dist2 = self.manifold.dist(quantized_flat.detach(), flat_inputs, dim=-1).pow(2)
        loss = self.commitment_cost * e_dist2.mean()

        if is_training:
            self._ema_update(flat_inputs, encoding_indices_flat)
            quantized_flat = F.embedding(encoding_indices_flat, self.embeddings)

        b, _, h, w = inputs.shape
        encoding_indices = encoding_indices_flat.view(b, h, w)
        quantized = self._reshape_quantized(quantized_flat, inputs)
        perplexity = self._perplexity(encodings)

        return {
            "quantize": quantized,
            "loss": loss,
            "q_latent_loss": torch.zeros((), device=inputs.device, dtype=inputs.dtype),
            "e_latent_loss": e_dist2.mean(),
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }
