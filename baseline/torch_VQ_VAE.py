import math
import os
import pickle
import tarfile
import urllib.request
from dataclasses import dataclass
from itertools import cycle
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =========================
# Utilities
# =========================

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


# =========================
# CIFAR-10 loading (no torchvision dependency)
# =========================

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"


class CIFAR10DictDataset(Dataset):
    """Return samples in the same style as the TensorFlow notebook: {'image': tensor}."""

    def __init__(self, images: np.ndarray):
        """
        images: [N, 3, 32, 32], dtype=uint8 or float32 in [0, 255].
        """
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[index]).float() / 255.0 - 0.5
        return {"image": image}


def download_and_extract_cifar10(data_root: str) -> str:
    os.makedirs(data_root, exist_ok=True)
    extracted_dir = os.path.join(data_root, CIFAR10_EXTRACTED_DIR)
    archive_path = os.path.join(data_root, CIFAR10_FILENAME)

    if not os.path.isdir(extracted_dir):
        if not os.path.isfile(archive_path):
            print(f"Downloading CIFAR-10 to {archive_path} ...")
            urllib.request.urlretrieve(CIFAR10_URL, archive_path)
        print(f"Extracting {archive_path} ...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_root)

    return extracted_dir


def load_cifar10_batch(batch_path: str) -> np.ndarray:
    with open(batch_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]  # [N, 3072]
    data = data.reshape(-1, 3, 32, 32)
    return data


def load_cifar10_numpy(data_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    extracted_dir = download_and_extract_cifar10(data_root)

    train_batches = []
    for i in range(1, 6):
        batch_path = os.path.join(extracted_dir, f"data_batch_{i}")
        train_batches.append(load_cifar10_batch(batch_path))
    train_images = np.concatenate(train_batches, axis=0)  # [50000, 3, 32, 32]

    test_path = os.path.join(extracted_dir, "test_batch")
    test_images = load_cifar10_batch(test_path)  # [10000, 3, 32, 32]

    # Same split logic as the original notebook:
    # first 40k = train, next 10k = validation, test batch = test
    train_split = train_images[:40_000]
    valid_split = train_images[40_000:50_000]
    test_split = test_images

    train_data_variance = float(np.var(train_split.astype(np.float32) / 255.0))
    return train_split, valid_split, test_split, train_data_variance


# =========================
# VQ layers
# =========================

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
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "distances": distances,
        }


# =========================
# Encoder / Decoder
# =========================

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
        self.enc_1 = nn.Conv2d(3, num_hiddens // 2, kernel_size=4, stride=2, padding=1)
        self.enc_2 = nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1)
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
            num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1
        )
        self.dec_3 = nn.ConvTranspose2d(
            num_hiddens // 2, 3, kernel_size=4, stride=2, padding=1
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


class VQVAEModel(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 vqvae: nn.Module,
                 pre_vq_conv1: nn.Conv2d,
                 data_variance: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vqvae = vqvae
        self.pre_vq_conv1 = pre_vq_conv1
        self.data_variance = data_variance

    def forward(self, inputs: torch.Tensor, is_training: bool = True) -> Dict[str, torch.Tensor]:
        z = self.pre_vq_conv1(self.encoder(inputs))
        vq_output = self.vqvae(z, is_training=is_training)
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


# =========================
# Training / evaluation
# =========================

@dataclass
class Config:
    batch_size: int = 32
    image_size: int = 32
    num_training_updates: int = 10_000

    num_hiddens: int = 128
    num_residual_hiddens: int = 32
    num_residual_layers: int = 2

    embedding_dim: int = 64
    num_embeddings: int = 512
    commitment_cost: float = 0.25

    vq_use_ema: bool = True
    decay: float = 0.99
    learning_rate: float = 3e-4

    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_root: str = "./data"
    print_every: int = 100


def build_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader, float]:
    train_images, valid_images, test_images, train_data_variance = load_cifar10_numpy(config.data_root)

    train_dataset = CIFAR10DictDataset(train_images)
    valid_dataset = CIFAR10DictDataset(valid_images)
    test_dataset = CIFAR10DictDataset(test_images)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader, test_loader, train_data_variance


def build_model(config: Config, data_variance: float) -> VQVAEModel:
    encoder = Encoder(
        num_hiddens=config.num_hiddens,
        num_residual_layers=config.num_residual_layers,
        num_residual_hiddens=config.num_residual_hiddens,
    )
    decoder = Decoder(
        input_channels=config.embedding_dim,
        num_hiddens=config.num_hiddens,
        num_residual_layers=config.num_residual_layers,
        num_residual_hiddens=config.num_residual_hiddens,
    )

    pre_vq_conv1 = nn.Conv2d(
        in_channels=config.num_hiddens,
        out_channels=config.embedding_dim,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    variance_scaling_(pre_vq_conv1.weight, distribution="uniform")
    if pre_vq_conv1.bias is not None:
        nn.init.zeros_(pre_vq_conv1.bias)

    if config.vq_use_ema:
        vq_vae = VectorQuantizerEMA(
            embedding_dim=config.embedding_dim,
            num_embeddings=config.num_embeddings,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
        )
    else:
        vq_vae = VectorQuantizer(
            embedding_dim=config.embedding_dim,
            num_embeddings=config.num_embeddings,
            commitment_cost=config.commitment_cost,
        )

    return VQVAEModel(
        encoder=encoder,
        decoder=decoder,
        vqvae=vq_vae,
        pre_vq_conv1=pre_vq_conv1,
        data_variance=data_variance,
    )


def train_step(model: VQVAEModel,
               batch: Dict[str, torch.Tensor],
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, torch.Tensor]:
    model.train()
    images = batch["image"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    output = model(images, is_training=True)
    output["loss"].backward()
    optimizer.step()

    return output


@torch.no_grad()
def evaluate(model: VQVAEModel,
             data_loader: DataLoader,
             device: torch.device,
             max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()

    losses = []
    recon_errors = []
    perplexities = []
    vq_losses = []

    for batch_idx, batch in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        output = model(images, is_training=False)

        losses.append(output["loss"].item())
        recon_errors.append(output["recon_error"].item())
        perplexities.append(output["vq_output"]["perplexity"].item())
        vq_losses.append(output["vq_output"]["loss"].item())

    return {
        "loss": float(np.mean(losses)),
        "recon_error": float(np.mean(recon_errors)),
        "perplexity": float(np.mean(perplexities)),
        "vq_loss": float(np.mean(vq_losses)),
    }


@torch.no_grad()
def reconstruct_batch(model: VQVAEModel,
                      batch: Dict[str, torch.Tensor],
                      device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    images = batch["image"].to(device, non_blocking=True)
    output = model(images, is_training=False)
    originals = images.cpu().numpy()
    reconstructions = output["x_recon"].cpu().numpy()
    return originals, reconstructions


def convert_batch_to_image_grid(image_batch: np.ndarray) -> np.ndarray:
    """image_batch: [B, C, H, W] with range [-0.5, 0.5]."""
    image_batch = np.transpose(image_batch, (0, 2, 3, 1))
    reshaped = (
        image_batch.reshape(4, 8, 32, 32, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(4 * 32, 8 * 32, 3)
    )
    return np.clip(reshaped + 0.5, 0.0, 1.0)


def plot_training_curves(train_recon_errors, train_perplexities) -> None:
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(train_recon_errors)
    ax.set_yscale("log")
    ax.set_title("NMSE")

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(train_perplexities)
    ax.set_title("Average codebook usage (perplexity)")

    plt.tight_layout()
    plt.show()


def show_reconstructions(model: VQVAEModel,
                         train_loader: DataLoader,
                         valid_loader: DataLoader,
                         device: torch.device) -> None:
    train_batch = next(iter(train_loader))
    valid_batch = next(iter(valid_loader))

    train_originals, train_reconstructions = reconstruct_batch(model, train_batch, device)
    valid_originals, valid_reconstructions = reconstruct_batch(model, valid_batch, device)

    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(convert_batch_to_image_grid(train_originals[:32]))
    ax.set_title("training data originals")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(convert_batch_to_image_grid(train_reconstructions[:32]))
    ax.set_title("training data reconstructions")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(convert_batch_to_image_grid(valid_originals[:32]))
    ax.set_title("validation data originals")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(convert_batch_to_image_grid(valid_reconstructions[:32]))
    ax.set_title("validation data reconstructions")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    config = Config()
    device = torch.device(config.device)
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, train_data_variance = build_dataloaders(config)
    print(f"train data variance: {train_data_variance:.8f}")

    model = build_model(config, train_data_variance).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []

    infinite_train_loader = cycle(train_loader)

    for step_index in range(config.num_training_updates):
        batch = next(infinite_train_loader)
        train_results = train_step(model, batch, optimizer, device)

        train_losses.append(train_results["loss"].item())
        train_recon_errors.append(train_results["recon_error"].item())
        train_perplexities.append(train_results["vq_output"]["perplexity"].item())
        train_vqvae_loss.append(train_results["vq_output"]["loss"].item())

        if (step_index + 1) % config.print_every == 0:
            print(
                f"{step_index + 1} train loss: {np.mean(train_losses[-config.print_every:]):.6f} "
                f"recon_error: {np.mean(train_recon_errors[-config.print_every:]):.3f} "
                f"perplexity: {np.mean(train_perplexities[-config.print_every:]):.3f} "
                f"vqvae loss: {np.mean(train_vqvae_loss[-config.print_every:]):.3f}"
            )

    valid_metrics = evaluate(model, valid_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    print("\nValidation:")
    print(valid_metrics)
    print("\nTest:")
    print(test_metrics)

    plot_training_curves(train_recon_errors, train_perplexities)
    show_reconstructions(model, train_loader, valid_loader, device)


if __name__ == "__main__":
    main()
