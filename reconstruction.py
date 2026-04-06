import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}


# -----------------------------
# Lightweight image transforms.
# -----------------------------
class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


class ResizeShorterSide:
    def __init__(self, size: int, resample=Image.BILINEAR):
        self.size = int(size)
        self.resample = resample

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")
        shorter = min(width, height)
        if shorter == self.size:
            return image
        if width < height:
            new_width = self.size
            new_height = int(round(height * (self.size / width)))
        else:
            new_height = self.size
            new_width = int(round(width * (self.size / height)))
        return image.resize((new_width, new_height), self.resample)


class CenterCrop:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        crop_w = min(self.size, width)
        crop_h = min(self.size, height)
        left = max((width - crop_w) // 2, 0)
        top = max((height - crop_h) // 2, 0)
        right = left + crop_w
        bottom = top + crop_h
        image = image.crop((left, top, right, bottom))
        if image.size != (self.size, self.size):
            image = image.resize((self.size, self.size), Image.BILINEAR)
        return image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., None]
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
        return tensor


class ReplicateChannels:
    def __init__(self, num_output_channels: int = 3):
        self.num_output_channels = int(num_output_channels)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(tensor.shape)}")
        if tensor.size(0) == self.num_output_channels:
            return tensor
        if tensor.size(0) == 1:
            return tensor.repeat(self.num_output_channels, 1, 1)
        raise ValueError(
            f"Cannot replicate tensor with shape {tuple(tensor.shape)} to {self.num_output_channels} channels."
        )


class InputCenterNormalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor - 0.5


# -----------------------------
# Dataset helpers.
# -----------------------------
class FolderDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        self.samples = [
            path for path in sorted(root_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not self.samples:
            raise RuntimeError(f"No image files found under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, 0


@dataclass
class InferenceContext:
    saved_args: Dict[str, Any]
    device: torch.device
    checkpoint_path: str
    model: nn.Module
    data_variance: float


DEFAULT_HYPERPARAMS = {
    "custom": {"hidden": 128, "k": 512},
    "imagenet": {"hidden": 256, "k": 1024},
    "ffhq": {"hidden": 256, "k": 1024},
    "cifar10": {"hidden": 128, "k": 512},
    "mnist": {"hidden": 64, "k": 128},
    "svhn": {"hidden": 96, "k": 256},
}


STANDARD_DATASET_NAMES = {"cifar10", "mnist", "svhn"}
FOLDER_DATASET_NAMES = {"custom", "imagenet", "ffhq"}


def _saved_arg(args_dict: Dict[str, Any], name: str, default: Any = None) -> Any:
    return args_dict.get(name, default)



def infer_image_size(dataset_name: str, image_size: Optional[int]) -> int:
    if image_size is not None:
        return int(image_size)
    defaults = {
        "mnist": 28,
        "cifar10": 32,
        "svhn": 32,
        "imagenet": 84,
        "custom": 256,
        "ffhq": 256,
    }
    if dataset_name not in defaults:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return defaults[dataset_name]



def build_transform(dataset_name: str, image_size: int):
    tail = [ToTensor(), ReplicateChannels(3), InputCenterNormalize()]
    if dataset_name in {"custom", "imagenet", "ffhq"}:
        return Compose([
            ResizeShorterSide(image_size),
            CenterCrop(image_size),
            *tail,
        ])
    if dataset_name in {"cifar10", "mnist", "svhn"}:
        return Compose(tail)
    raise ValueError(f"Unsupported dataset: {dataset_name}")



def _join_dataset_root(data_dir: str, dataset_name: str, dataset_dir_name: Optional[str]) -> str:
    if dataset_name == "custom":
        if not dataset_dir_name:
            raise ValueError("dataset_dir_name must be present for dataset='custom'.")
        return os.path.join(data_dir, dataset_dir_name)
    return os.path.join(data_dir, dataset_name)



def _resolve_folder_split(root: str, split: str, dataset_name: str) -> str:
    if dataset_name in {"custom", "imagenet"}:
        candidates = [split]
        if split == "test":
            candidates.extend(["val", "validation"])
        elif split == "train":
            candidates.extend(["training"])
        for candidate in candidates:
            candidate_path = os.path.join(root, candidate)
            if os.path.isdir(candidate_path):
                return candidate_path
        # Fall back to root itself for robustness.
        return root
    return root



def build_dataset(saved_args: Dict[str, Any], split: str, data_dir: Optional[str] = None,
                  dataset_dir_name: Optional[str] = None, image_size: Optional[int] = None):
    dataset_name = _saved_arg(saved_args, "dataset")
    if dataset_name is None:
        raise ValueError("args.json is missing required field 'dataset'.")

    data_dir = data_dir if data_dir is not None else _saved_arg(saved_args, "data_dir", "./data")
    dataset_dir_name = dataset_dir_name if dataset_dir_name is not None else _saved_arg(saved_args, "dataset_dir_name")
    image_size = infer_image_size(dataset_name, image_size if image_size is not None else _saved_arg(saved_args, "image_size"))
    transform = build_transform(dataset_name, image_size)

    root = _join_dataset_root(data_dir, dataset_name, dataset_dir_name)

    if dataset_name in FOLDER_DATASET_NAMES:
        split_root = _resolve_folder_split(root, split, dataset_name)
        return FolderDataset(split_root, transform=transform)

    try:
        from torchvision import datasets as tv_datasets
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "torchvision is required for cifar10/mnist/svhn inference datasets, but it could not be imported. "
            f"Original error: {exc}"
        ) from exc

    if dataset_name == "cifar10":
        return tv_datasets.CIFAR10(root=root, train=(split == "train"), transform=transform, download=True)
    if dataset_name == "mnist":
        return tv_datasets.MNIST(root=root, train=(split == "train"), transform=transform, download=True)
    if dataset_name == "svhn":
        return tv_datasets.SVHN(root=root, split="train" if split == "train" else "test", transform=transform, download=True)
    raise ValueError(f"Unsupported dataset: {dataset_name}")



def build_dataloader(saved_args: Dict[str, Any], split: str, data_dir: Optional[str] = None,
                     dataset_dir_name: Optional[str] = None, image_size: Optional[int] = None,
                     batch_size: Optional[int] = None, num_workers: Optional[int] = None,
                     shuffle: Optional[bool] = None) -> DataLoader:
    dataset = build_dataset(
        saved_args=saved_args,
        split=split,
        data_dir=data_dir,
        dataset_dir_name=dataset_dir_name,
        image_size=image_size,
    )
    batch_size = int(batch_size if batch_size is not None else _saved_arg(saved_args, "batch_size", 128))
    num_workers = int(num_workers if num_workers is not None else _saved_arg(saved_args, "num_workers", 4))
    pin_memory = torch.cuda.is_available()
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )



def estimate_data_variance(loader: DataLoader, device: torch.device, max_batches: int = 32) -> float:
    sq_sum = 0.0
    total_sum = 0.0
    count = 0
    for batch_index, (data, _) in enumerate(loader):
        if batch_index >= max_batches:
            break
        data = data.to(device, non_blocking=True)
        sq_sum += torch.sum(data * data).item()
        total_sum += torch.sum(data).item()
        count += data.numel()
    mean = total_sum / max(1, count)
    variance = sq_sum / max(1, count) - mean * mean
    return max(float(variance), 1e-6)


# -----------------------------
# Checkpoint / model loading.
# -----------------------------
def read_saved_args(model_dir: str) -> Dict[str, Any]:
    args_path = os.path.join(model_dir, "args.json")
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f"Could not find args.json under model directory: {args_path}")
    with open(args_path, "r", encoding="utf-8") as handle:
        saved_args = json.load(handle)
    return saved_args



def resolve_device(device: Optional[str]) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def find_checkpoint(model_dir: str, checkpoint: Optional[str] = None) -> str:
    candidates: List[str] = []
    if checkpoint:
        checkpoint_path = checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(model_dir, checkpoint_path)
        candidates.append(checkpoint_path)
    else:
        candidates.append(os.path.join(model_dir, "best_val.pth"))
        ckpt_dir = os.path.join(model_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            names = [name for name in os.listdir(ckpt_dir) if name.endswith(".pth") or name.endswith(".pt") or name.endswith(".pth.tar")]
            names.sort()
            candidates.extend(os.path.join(ckpt_dir, name) for name in reversed(names))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a checkpoint under {model_dir}. Checked: {candidates}"
    )



def _extract_state_dict(raw_checkpoint: Any) -> Dict[str, Any]:
    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                raw_checkpoint = value
                break
    if not isinstance(raw_checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(raw_checkpoint)!r}")
    state_dict = dict(raw_checkpoint)
    if state_dict and all(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict



def _parse_float_list(text_or_values: Optional[Any], expected_len: int, name: str) -> Optional[List[float]]:
    if text_or_values is None:
        return None
    if isinstance(text_or_values, str):
        tokens = text_or_values.replace(",", " ").split()
    else:
        tokens = list(text_or_values)
    values = [float(token) for token in tokens]
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain exactly {expected_len} values, got {len(values)}.")
    return values



def _extract_component_count(state_dict: Dict[str, Any]) -> int:
    pattern = re.compile(r"^components\.(\d+)\.")
    indices = set()
    for key in state_dict.keys():
        match = pattern.match(str(key))
        if match:
            indices.add(int(match.group(1)))
    return 0 if not indices else max(indices) + 1



def _extract_single_curvature(saved_args: Dict[str, Any], state_dict: Dict[str, Any]) -> float:
    for key in ("manifold.k", "curvature"):
        if key in state_dict:
            value = state_dict[key]
            return float(torch.as_tensor(value).reshape(()).cpu().item())
    for key in ("c", "curvature"):
        value = saved_args.get(key)
        if value is not None:
            return float(value)
    return -1.0



def _extract_mixed_curvatures(saved_args: Dict[str, Any], state_dict: Dict[str, Any], count: int) -> Optional[List[float]]:
    values: List[float] = []
    for idx in range(count):
        key = f"components.{idx}.manifold.k"
        if key not in state_dict:
            values = []
            break
        values.append(float(torch.as_tensor(state_dict[key]).reshape(()).cpu().item()))
    if values:
        return values
    return _parse_float_list(saved_args.get("curvatures"), count, "curvatures")



def _extract_mixed_alphas(saved_args: Dict[str, Any], state_dict: Dict[str, Any], count: int) -> Optional[List[float]]:
    values: List[float] = []
    for idx in range(count):
        key = f"components.{idx}.alpha"
        if key not in state_dict:
            values = []
            break
        values.append(float(torch.as_tensor(state_dict[key]).reshape(()).cpu().item()))
    if values:
        return values
    return _parse_float_list(saved_args.get("alphas"), count, "alphas")



def _build_single_encoder_decoder(hidden: int, num_residual_layers: int, num_residual_hiddens: int,
                                  embedding_dim: int):
    from model.auto_encoder import Encoder, Decoder
    from model.embedding import variance_scaling_

    encoder = Encoder(
        num_hiddens=hidden,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
    )
    pre_vq_conv1 = nn.Conv2d(
        in_channels=hidden,
        out_channels=embedding_dim,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    variance_scaling_(pre_vq_conv1.weight, distribution="uniform")
    if pre_vq_conv1.bias is not None:
        nn.init.zeros_(pre_vq_conv1.bias)

    decoder = Decoder(
        input_channels=embedding_dim,
        num_hiddens=hidden,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
    )
    return encoder, decoder, pre_vq_conv1



def build_model_from_saved_args(saved_args: Dict[str, Any], state_dict: Dict[str, Any], data_variance: float) -> nn.Module:
    from model.auto_encoder import VQVAE, VQCVAE, MixedCurvatureComponent, MixedCurvatureVQCVAE, parse_space_spec
    from model.embedding import VectorQuantizer, VectorQuantizerEMA, McVectorQuantizer, McVectorQuantizerEMA

    dataset_name = _saved_arg(saved_args, "dataset")
    model_name = _saved_arg(saved_args, "model", "vqcvae")
    mixed = bool(_saved_arg(saved_args, "mixed", False))
    hidden = int(_saved_arg(saved_args, "hidden", DEFAULT_HYPERPARAMS[dataset_name]["hidden"]))
    num_embeddings = int(_saved_arg(saved_args, "k", DEFAULT_HYPERPARAMS[dataset_name]["k"]))
    embedding_dim = int(_saved_arg(saved_args, "embedding_dim", 64))
    num_residual_layers = int(_saved_arg(saved_args, "num_residual_layers", 2))
    num_residual_hiddens = int(_saved_arg(saved_args, "num_residual_hiddens", 32))
    commitment_cost = float(_saved_arg(saved_args, "commit_coef", 0.25))
    decay = float(_saved_arg(saved_args, "decay", 0.99))
    use_ema = bool(_saved_arg(saved_args, "ema", True))
    fix_curvature = bool(_saved_arg(saved_args, "fix_curvature", False))
    fix_alpha = bool(_saved_arg(saved_args, "fix_alpha", False))

    if model_name == "vqvae":
        encoder, decoder, pre_vq_conv1 = _build_single_encoder_decoder(
            hidden=hidden,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            embedding_dim=embedding_dim,
        )
        model = VQVAE(
            encoder=encoder,
            decoder=decoder,
            pre_vq_conv1=pre_vq_conv1,
            data_variance=data_variance,
            num_channel=3,
            ema=use_ema,
        )
        if use_ema:
            model.embedding = VectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                decay=decay,
            )
        else:
            model.embedding = VectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
            )
        return model

    if model_name != "vqcvae":
        raise ValueError(f"Unsupported model type in args.json: {model_name!r}")

    if not mixed:
        curvature = _extract_single_curvature(saved_args, state_dict)
        encoder, decoder, pre_vq_conv1 = _build_single_encoder_decoder(
            hidden=embedding_dim,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            embedding_dim=embedding_dim,
        )
        model = VQCVAE(
            encoder=encoder,
            decoder=decoder,
            pre_vq_conv1=pre_vq_conv1,
            data_variance=data_variance,
            num_channel=3,
            ema=use_ema,
            learnable=not fix_curvature,
            curvature=curvature,
        )
        if use_ema:
            model.embedding = McVectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                manifold=model.manifold,
                decay=decay,
            )
        else:
            model.embedding = McVectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                manifold=model.manifold,
            )
        return model

    spaces = parse_space_spec(_saved_arg(saved_args, "spaces"))
    component_count = len(spaces)
    if component_count == 0:
        component_count = _extract_component_count(state_dict)
    curvatures = _extract_mixed_curvatures(saved_args, state_dict, component_count)
    alphas = _extract_mixed_alphas(saved_args, state_dict, component_count)

    components = []
    total_embedding_dim = 0
    for index, space in enumerate(spaces):
        current_curvature = float(space["curvature"]) if curvatures is None else float(curvatures[index])
        current_alpha = 1.0 if alphas is None else float(alphas[index])
        current_embedding_dim = int(space["dimension"])
        total_embedding_dim += current_embedding_dim
        components.append(
            MixedCurvatureComponent(
                num_hiddens=current_embedding_dim,
                num_residual_layers=num_residual_layers,
                num_residual_hiddens=num_residual_hiddens,
                embedding_dim=current_embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                decay=decay,
                use_ema=use_ema,
                curvature=current_curvature,
                learnable_curvature=not fix_curvature,
                alpha=current_alpha,
                learnable_alpha=not fix_alpha,
            )
        )

    from model.auto_encoder import Decoder
    decoder = Decoder(
        input_channels=total_embedding_dim,
        num_hiddens=hidden,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
    )
    model = MixedCurvatureVQCVAE(
        components=components,
        decoder=decoder,
        data_variance=data_variance,
    )
    model.spaces = spaces
    return model



def load_inference_context(model_dir: str,
                           checkpoint: Optional[str] = None,
                           device: Optional[str] = None,
                           data_dir: Optional[str] = None,
                           dataset_dir_name: Optional[str] = None,
                           image_size: Optional[int] = None,
                           batch_size: Optional[int] = None,
                           num_workers: Optional[int] = None) -> InferenceContext:
    saved_args = read_saved_args(model_dir)
    resolved_device = resolve_device(device)
    checkpoint_path = find_checkpoint(model_dir, checkpoint)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(raw_checkpoint)

    # Re-estimate data variance using the training split when possible so that recon_error
    # remains comparable with the value logged during training.
    try:
        variance_loader = build_dataloader(
            saved_args=saved_args,
            split="train",
            data_dir=data_dir,
            dataset_dir_name=dataset_dir_name,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    except Exception:
        variance_loader = build_dataloader(
            saved_args=saved_args,
            split="test",
            data_dir=data_dir,
            dataset_dir_name=dataset_dir_name,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
    data_variance = estimate_data_variance(variance_loader, resolved_device)

    model = build_model_from_saved_args(saved_args, state_dict, data_variance)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")
    model.to(resolved_device)
    model.eval()
    return InferenceContext(
        saved_args=saved_args,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        model=model,
        data_variance=data_variance,
    )


# -----------------------------
# Reconstruction helpers.
# -----------------------------
def _unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor.detach().cpu().float() + 0.5, 0.0, 1.0)



def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = _unnormalize(tensor)
    array = tensor.permute(1, 2, 0).numpy()
    array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array)



def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor_to_pil(tensor).save(path)



def make_comparison_grid(original: torch.Tensor, reconstruction: torch.Tensor, ncols: int = 8) -> Image.Image:
    original = _unnormalize(original)
    reconstruction = _unnormalize(reconstruction)
    n = min(original.size(0), reconstruction.size(0), ncols)
    if n <= 0:
        raise ValueError("Need at least one sample to build a comparison grid.")

    samples = []
    for batch in (original[:n], reconstruction[:n]):
        row_arrays = []
        for tensor in batch:
            array = tensor.permute(1, 2, 0).numpy()
            array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
            row_arrays.append(array)
        samples.append(np.concatenate(row_arrays, axis=1))
    canvas = np.concatenate(samples, axis=0)
    return Image.fromarray(canvas)



def prepare_output_dirs(output_dir: str, gt_dir_name: str, rec_dir_name: str,
                        comparison_dir_name: str, overwrite: bool) -> Tuple[str, str, str]:
    gt_dir = os.path.join(output_dir, gt_dir_name)
    rec_dir = os.path.join(output_dir, rec_dir_name)
    comparison_dir = os.path.join(output_dir, comparison_dir_name)
    if overwrite:
        for path in (gt_dir, rec_dir, comparison_dir):
            if os.path.isdir(path):
                shutil.rmtree(path)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    return gt_dir, rec_dir, comparison_dir


@torch.inference_mode()
def run_reconstruction(model_dir: str,
                       checkpoint: Optional[str] = None,
                       split: str = "test",
                       device: Optional[str] = None,
                       data_dir: Optional[str] = None,
                       dataset_dir_name: Optional[str] = None,
                       image_size: Optional[int] = None,
                       batch_size: Optional[int] = None,
                       num_workers: Optional[int] = None,
                       max_samples: Optional[int] = None,
                       output_dir: Optional[str] = None,
                       gt_dir_name: str = "train",
                       rec_dir_name: str = "rec",
                       comparison_dir_name: str = "comparison",
                       overwrite: bool = False,
                       save_comparison_grids: bool = True) -> Dict[str, Any]:
    context = load_inference_context(
        model_dir=model_dir,
        checkpoint=checkpoint,
        device=device,
        data_dir=data_dir,
        dataset_dir_name=dataset_dir_name,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    loader = build_dataloader(
        saved_args=context.saved_args,
        split=split,
        data_dir=data_dir,
        dataset_dir_name=dataset_dir_name,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    output_dir = output_dir or model_dir
    gt_dir, rec_dir, comparison_dir = prepare_output_dirs(
        output_dir=output_dir,
        gt_dir_name=gt_dir_name,
        rec_dir_name=rec_dir_name,
        comparison_dir_name=comparison_dir_name,
        overwrite=overwrite,
    )

    sample_count = 0
    grid_count = 0
    for batch_index, (inputs, _) in enumerate(loader):
        if max_samples is not None and sample_count >= max_samples:
            break
        if max_samples is not None:
            remaining = max_samples - sample_count
            if remaining <= 0:
                break
            if inputs.size(0) > remaining:
                inputs = inputs[:remaining]

        inputs = inputs.to(context.device, non_blocking=True)
        output = context.model(inputs, is_training=False)
        recon = output["x_recon"]

        batch_size_now = inputs.size(0)
        for local_index in range(batch_size_now):
            filename = f"{sample_count + local_index:06d}.png"
            save_tensor_image(inputs[local_index], os.path.join(gt_dir, filename))
            save_tensor_image(recon[local_index], os.path.join(rec_dir, filename))

        if save_comparison_grids:
            grid_image = make_comparison_grid(inputs, recon)
            grid_image.save(os.path.join(comparison_dir, f"batch_{batch_index:04d}.png"))
            grid_count += 1

        sample_count += batch_size_now
        print(f"[reconstruction] processed {sample_count} samples")

    manifest = {
        "model_dir": os.path.abspath(model_dir),
        "checkpoint_path": os.path.abspath(context.checkpoint_path),
        "output_dir": os.path.abspath(output_dir),
        "split": split,
        "num_samples": sample_count,
        "data_variance": context.data_variance,
        "saved_args": context.saved_args,
        "gt_dir": os.path.abspath(gt_dir),
        "rec_dir": os.path.abspath(rec_dir),
        "comparison_dir": os.path.abspath(comparison_dir),
        "comparison_grid_count": grid_count,
    }
    manifest_path = os.path.join(output_dir, "reconstruction_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[reconstruction] saved manifest to {manifest_path}")
    return manifest



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct images with a trained VQCVAE or MixedCurvatureVQCVAE checkpoint."
    )
    parser.add_argument("--model-dir", required=True, help="Training result directory containing args.json and checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or path relative to --model-dir. Defaults to best_val.pth.")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split used for reconstruction.")
    parser.add_argument("--device", default="auto", help="Inference device, e.g. cpu, cuda, cuda:0. Defaults to auto.")
    parser.add_argument("--data-dir", default=None, help="Override dataset root directory stored in args.json.")
    parser.add_argument("--dataset-dir-name", default=None, help="Override dataset_dir_name stored in args.json.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size used by the folder-dataset transform pipeline.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size used during reconstruction.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of reconstructed samples.")
    parser.add_argument("--output-dir", default=None, help="Directory used to store reconstructed results. Defaults to --model-dir.")
    parser.add_argument("--gt-dir-name", default="train", help="Subdirectory name for original images. Kept compatible with evaluation0.py.")
    parser.add_argument("--rec-dir-name", default="rec", help="Subdirectory name for reconstructed images. Kept compatible with evaluation0.py.")
    parser.add_argument("--comparison-dir-name", default="comparison", help="Subdirectory name for side-by-side comparison grids.")
    parser.add_argument("--overwrite", action="store_true", help="Delete previous output subdirectories before writing new reconstructions.")
    parser.add_argument("--no-comparison-grids", action="store_true", help="Do not save side-by-side comparison grids.")
    return parser



def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    manifest = run_reconstruction(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        split=args.split,
        device=args.device,
        data_dir=args.data_dir,
        dataset_dir_name=args.dataset_dir_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        gt_dir_name=args.gt_dir_name,
        rec_dir_name=args.rec_dir_name,
        comparison_dir_name=args.comparison_dir_name,
        overwrite=args.overwrite,
        save_comparison_grids=not args.no_comparison_grids,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# python reconstruction.py --model-dir=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test_geometric_perplexity_ema/2026-04-06_19-48-09_cifar10_vqvae_h128_k128_emb64_ema --device=cuda:3 --overwrite