import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Sequence, List, Union, Any


def _global_pooled_cross_decorrelation_loss(branch_features: Sequence[torch.Tensor],
                                            eps: float = 1e-5) -> torch.Tensor:
    if len(branch_features) <= 1:
        if len(branch_features) == 0:
            return torch.tensor(0.0)
        return branch_features[0].new_zeros(())

    pooled_features = []
    for feature in branch_features:
        if feature.dim() < 2:
            raise ValueError(f"Expected branch feature with at least 2 dims, got shape {tuple(feature.shape)}.")
        if feature.dim() == 2:
            pooled = feature
        else:
            reduce_dims = tuple(range(2, feature.dim()))
            pooled = feature.mean(dim=reduce_dims)
        pooled = pooled - pooled.mean(dim=0, keepdim=True)
        pooled = pooled / (pooled.std(dim=0, keepdim=True) + eps)
        pooled_features.append(pooled)

    batch_size = pooled_features[0].shape[0]
    loss = pooled_features[0].new_zeros(())
    count = 0
    for i in range(len(pooled_features)):
        for j in range(i + 1, len(pooled_features)):
            cross_cov = pooled_features[i].transpose(0, 1) @ pooled_features[j] / max(1, batch_size)
            loss = loss + cross_cov.pow(2).mean()
            count += 1
    if count > 0:
        loss = loss / count
    return loss
import geoopt

from .embedding import variance_scaling_, VectorQuantizerEMA, VectorQuantizer, McVectorQuantizerEMA, McVectorQuantizer


SpaceSpec = Union[str, Sequence[Dict[str, Any]], Sequence[Sequence[float]]]


def clamp_tangent_for_positive_curvature(u: torch.Tensor,
                                         k,
                                         dim: int = 1,
                                         margin: float = 0.3,
                                         eps: float = 1e-6) -> torch.Tensor:
    if torch.is_tensor(k):
        k_val = k.detach().to(device=u.device, dtype=u.dtype)
    else:
        k_val = torch.tensor(k, device=u.device, dtype=u.dtype)

    if torch.all(k_val <= 0):
        return u

    max_norm = margin * math.pi / (2.0 * torch.sqrt(k_val).clamp_min(eps))
    u_norm = torch.linalg.vector_norm(u, dim=dim, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / u_norm, max=1.0)
    return u * scale


class ResidualBlock(nn.Module):
    def __init__(self, num_hiddens: int, num_residual_hiddens: int):
        super().__init__()
        self.conv3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv1 = nn.Conv2d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        variance_scaling_(self.conv3.weight, distribution="uniform")
        variance_scaling_(self.conv1.weight, distribution="uniform")
        if self.conv3.bias is not None:
            nn.init.zeros_(self.conv3.bias)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv3(F.relu(x))
        out = self.conv1(F.relu(out))
        return x + out


class ResidualStack(nn.Module):
    def __init__(self,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = inputs
        for layer in self.layers:
            h = layer(h)
        return F.relu(h)


class Encoder(nn.Module):
    def __init__(self,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int):
        super().__init__()
        self.enc_1 = nn.Conv2d(3, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.enc_2 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1)
        self.enc_3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in [self.enc_1, self.enc_2, self.enc_3]:
            variance_scaling_(layer.weight, distribution="uniform")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc_1(x))
        h = F.relu(self.enc_2(h))
        h = F.relu(self.enc_3(h))
        return self.residual_stack(h)


class Decoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int):
        super().__init__()
        self.dec_1 = nn.Conv2d(input_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_hiddens, num_residual_layers, num_residual_hiddens)
        self.dec_2 = nn.ConvTranspose2d(
            num_hiddens, num_hiddens, kernel_size=4, stride=2, padding=1
        )
        self.dec_3 = nn.ConvTranspose2d(
            num_hiddens, 3, kernel_size=4, stride=2, padding=1
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in [self.dec_1, self.dec_2, self.dec_3]:
            variance_scaling_(layer.weight, distribution="uniform")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dec_1(x)
        h = self.residual_stack(h)
        h = F.relu(self.dec_2(h))
        x_recon = self.dec_3(h)
        return x_recon


class VQVAE(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 pre_vq_conv1: nn.Conv2d,
                 data_variance: float,
                 num_channel: int = 3,
                 ema: bool = True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if ema:
            embedding = VectorQuantizerEMA(
                embedding_dim=decoder.dec_1.in_channels,
                num_embeddings=512,
                commitment_cost=0.25,
                decay=0.99,
            )
        else:
            embedding = VectorQuantizer(
                embedding_dim=decoder.dec_1.in_channels,
                num_embeddings=512,
                commitment_cost=0.25,
            )
        self.embedding = embedding
        self.pre_vq_conv1 = pre_vq_conv1
        self.data_variance = data_variance

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        z = self.pre_vq_conv1(self.encoder(inputs))
        vq_output = self.embedding(z, is_training=is_training)
        x_recon = self.decoder(vq_output["quantize"])
        recon_error = F.mse_loss(x_recon, inputs) / self.data_variance
        loss = recon_error + vq_output["loss"]
        return {
            "z": z,
            "x_recon": x_recon,
            "loss": loss,
            "recon_error": recon_error,
            "vq_output": vq_output,
        }


class VQCVAE(VQVAE):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 pre_vq_conv1: nn.Conv2d,
                 data_variance: float,
                 num_channel: int = 3,
                 ema: bool = True,
                 learnable: bool = False,
                 curvature: float = -1.0):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            pre_vq_conv1=pre_vq_conv1,
            data_variance=data_variance,
            num_channel=num_channel,
            ema=ema,
        )
        self.manifold = geoopt.Stereographic(k=curvature, learnable=learnable)

        if ema:
            embedding = McVectorQuantizerEMA(
                embedding_dim=decoder.dec_1.in_channels,
                num_embeddings=512,
                commitment_cost=0.25,
                manifold=self.manifold,
                decay=0.99,
            )
        else:
            embedding = McVectorQuantizer(
                embedding_dim=decoder.dec_1.in_channels,
                num_embeddings=512,
                commitment_cost=0.25,
                manifold=self.manifold,
            )
        self.embedding = embedding

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        z_tan = self.pre_vq_conv1(self.encoder(inputs))

        k_val = self.manifold.k.detach() if torch.is_tensor(self.manifold.k) else torch.tensor(self.manifold.k, device=z_tan.device, dtype=z_tan.dtype)
        if torch.all(k_val > 0):
            safe_scale = torch.sqrt(
                torch.clamp(
                    torch.as_tensor(self.embedding.embedding_dim, device=z_tan.device, dtype=z_tan.dtype) * k_val.to(dtype=z_tan.dtype),
                    min=1e-6,
                )
            )
            z_tan = z_tan / safe_scale
            z_tan = clamp_tangent_for_positive_curvature(z_tan, self.manifold.k, dim=1, margin=0.3)
        print("curvature", self.manifold.k)
        z_man = self.manifold.expmap0(z_tan, dim=1)
        vq_output = self.embedding(z_man, is_training=is_training)
        z_man_q = vq_output["quantize"]
        z_tan_q = self.manifold.logmap0(z_man_q, dim=1)
        x_recon = self.decoder(z_tan_q)
        recon_error = F.mse_loss(x_recon, inputs) / self.data_variance
        loss = recon_error + vq_output["loss"]
        return {
            "z": z_man,
            "z_tangent": z_tan,
            "x_recon": x_recon,
            "loss": loss,
            "recon_error": recon_error,
            "vq_output": vq_output,
        }



def parse_space_spec(spaces: SpaceSpec) -> List[Dict[str, float]]:
    if spaces is None:
        raise ValueError("spaces must be provided for mixed-curvature VQCVAE")

    parsed: List[Dict[str, float]] = []
    if isinstance(spaces, str):
        items = spaces.replace(";", " ").split()
        for item in items:
            parts = [part.strip() for part in item.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError(
                    "Each mixed-space spec must have the form 'curvature,dimension'. "
                    f"Got {item!r}."
                )
            curvature = float(parts[0])
            dimension = int(parts[1])
            if dimension <= 0:
                raise ValueError(f"Space dimension must be positive, got {dimension} in {item!r}.")
            parsed.append({"curvature": curvature, "dimension": dimension})
    else:
        for item in spaces:
            if isinstance(item, dict):
                curvature = float(item["curvature"])
                dimension = int(item["dimension"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                curvature = float(item[0])
                dimension = int(item[1])
            else:
                raise ValueError(
                    "spaces must be a string like '1,32 -1,32' or a sequence of "
                    "{'curvature': ..., 'dimension': ...} items."
                )
            if dimension <= 0:
                raise ValueError(f"Space dimension must be positive, got {dimension}.")
            parsed.append({"curvature": curvature, "dimension": dimension})

    if not parsed:
        raise ValueError("At least one mixed space must be specified.")
    return parsed


class MixedCurvatureComponent(nn.Module):
    def __init__(self,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 decay: float,
                 use_ema: bool,
                 curvature: float,
                 learnable_curvature: bool,
                 alpha: float = 1.0,
                 learnable_alpha: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = Encoder(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        self.pre_vq_conv1 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        variance_scaling_(self.pre_vq_conv1.weight, distribution="uniform")
        if self.pre_vq_conv1.bias is not None:
            nn.init.zeros_(self.pre_vq_conv1.bias)

        self.manifold = geoopt.Stereographic(k=curvature, learnable=learnable_curvature)
        if use_ema:
            self.embedding = McVectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                manifold=self.manifold,
                decay=decay,
            )
        else:
            self.embedding = McVectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                manifold=self.manifold,
            )

        self.alpha = nn.Parameter(
            torch.tensor(float(alpha), dtype=torch.float32),
            requires_grad=learnable_alpha,
        )

    def _prepare_tangent(self, z_tan: torch.Tensor) -> torch.Tensor:
        k_val = self.manifold.k.detach() if torch.is_tensor(self.manifold.k) else torch.tensor(self.manifold.k, device=z_tan.device, dtype=z_tan.dtype)
        k_val = k_val.to(device=z_tan.device, dtype=z_tan.dtype)
        if torch.all(k_val > 0):
            safe_scale = torch.sqrt(
                torch.clamp(
                    torch.as_tensor(self.embedding_dim, device=z_tan.device, dtype=z_tan.dtype) * k_val,
                    min=1e-6,
                )
            )
            z_tan = z_tan / safe_scale
            z_tan = clamp_tangent_for_positive_curvature(z_tan, self.manifold.k, dim=1, margin=0.3)
        return z_tan

    def encode_tangent(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pre_vq_conv1(self.encoder(inputs))

    def encode_manifold(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_tan = self._prepare_tangent(self.encode_tangent(inputs))
        z_man = self.manifold.expmap0(z_tan, dim=1)
        return {"z_tangent": z_tan, "z": z_man}

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        encoded = self.encode_manifold(inputs)
        z_tan = encoded["z_tangent"]
        z_man = encoded["z"]
        vq_output = self.embedding(z_man, is_training=is_training)
        z_man_q = vq_output["quantize"]
        z_tan_q = self.manifold.logmap0(z_man_q, dim=1)
        z_tan_q = self.alpha * z_tan_q
        return {
            "z": z_man,
            "z_tangent": z_tan,
            "z_tangent_quantized": z_tan_q,
            "vq_output": vq_output,
        }

    def encode_index(self, inputs: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        encoded = self.encode_manifold(inputs)
        vq_output = self.embedding(encoded["z"], is_training=is_training)
        return vq_output["encoding_indices"]

    def decode_indices(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        z_man_q = self.embedding.quantize(encoding_indices)
        z_man_q = self.manifold.projx(z_man_q)
        z_man_q = z_man_q.permute(0, 3, 1, 2).contiguous()
        z_tan_q = self.manifold.logmap0(z_man_q, dim=1)
        return self.alpha * z_tan_q


class MixedCurvatureVQCVAE(nn.Module):
    def __init__(self,
                 components: Sequence[MixedCurvatureComponent],
                 decoder: Decoder,
                 data_variance: float):
        super().__init__()
        self.components = nn.ModuleList(list(components))
        if len(self.components) == 0:
            raise ValueError("MixedCurvatureVQCVAE requires at least one component.")
        self.decoder = decoder
        self.data_variance = data_variance
        self.is_mixed = True
        self.total_embedding_dim = sum(component.embedding_dim for component in self.components)

    @property
    def curvature(self) -> List[float]:
        return [float(component.manifold.k.detach().reshape(()).cpu().item()) for component in self.components]

    @property
    def alpha(self) -> List[float]:
        return [float(component.alpha.detach().reshape(()).cpu().item()) for component in self.components]

    def decode(self, z_tangent: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_tangent)

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        component_outputs = []
        quantized_tangent_spaces = []
        manifold_spaces = []
        tangent_spaces = []
        vq_losses = []
        q_latent_losses = []
        e_latent_losses = []
        perplexities = []
        encoding_indices = []
        distances = []

        for component in self.components:
            component_output = component(inputs, is_training=is_training)
            component_outputs.append(component_output)
            quantized_tangent_spaces.append(component_output["z_tangent_quantized"])
            tangent_spaces.append(component_output["z_tangent"])
            manifold_spaces.append(component_output["z"])

            vq_output = component_output["vq_output"]
            vq_losses.append(vq_output["loss"])
            q_latent_losses.append(vq_output.get("q_latent_loss", torch.zeros_like(vq_output["loss"])))
            e_latent_losses.append(vq_output.get("e_latent_loss", torch.zeros_like(vq_output["loss"])))
            perplexities.append(vq_output["perplexity"])
            encoding_indices.append(vq_output["encoding_indices"])
            distances.append(vq_output["distances"])

        latent = torch.cat(quantized_tangent_spaces, dim=1)
        decorrelation_loss = _global_pooled_cross_decorrelation_loss(quantized_tangent_spaces)
        x_recon = self.decoder(latent)
        recon_error = F.mse_loss(x_recon, inputs) / self.data_variance
        total_vq_loss = torch.stack(vq_losses).sum()
        total_q_latent_loss = torch.stack(q_latent_losses).sum()
        total_e_latent_loss = torch.stack(e_latent_losses).sum()
        mean_perplexity = torch.stack(perplexities).mean()
        stacked_indices = torch.stack(encoding_indices, dim=1)

        return {
            "z": manifold_spaces,
            "z_tangent": tangent_spaces,
            "latent": latent,
            "x_recon": x_recon,
            "loss": recon_error + total_vq_loss,
            "recon_error": recon_error,
            "decorrelation_loss": decorrelation_loss,
            "curvature": torch.tensor(self.curvature, device=inputs.device, dtype=inputs.dtype),
            "alpha": torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype),
            "component_outputs": component_outputs,
            "vq_output": {
                "quantize": latent,
                "loss": total_vq_loss,
                "q_latent_loss": total_q_latent_loss,
                "e_latent_loss": total_e_latent_loss,
                "perplexity": mean_perplexity,
                "perplexity_per_component": torch.stack(perplexities),
                "encoding_indices": stacked_indices,
                "encoding_indices_per_component": encoding_indices,
                "distances": distances,
            },
        }

    def latent(self, inputs: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        latents = [component(inputs, is_training=is_training)["z_tangent_quantized"] for component in self.components]
        return torch.cat(latents, dim=1)

    def encode_index(self, inputs: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        indices = [component.encode_index(inputs, is_training=is_training) for component in self.components]
        return torch.stack(indices, dim=1)

    def decode_samples(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        if embedding_indices.dim() != 4:
            raise ValueError(
                "embedding_indices must have shape [B, num_components, H, W] for mixed-curvature decoding. "
                f"Got {tuple(embedding_indices.shape)}."
            )
        if embedding_indices.size(1) != len(self.components):
            raise ValueError(
                "The second dimension of embedding_indices must equal the number of mixed-curvature components. "
                f"Got {embedding_indices.size(1)} and {len(self.components)}."
            )
        spaces = [
            component.decode_indices(embedding_indices[:, idx, ...])
            for idx, component in enumerate(self.components)
        ]
        latent = torch.cat(spaces, dim=1)
        return self.decode(latent)
