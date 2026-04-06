import argparse
import json
import math
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.auto_encoder import Decoder, Encoder, MixedCurvatureComponent, MixedCurvatureVQCVAE, VQCVAE, VQVAE, parse_space_spec
from model.embedding import McVectorQuantizer, McVectorQuantizerEMA, VectorQuantizer, VectorQuantizerEMA, variance_scaling_


def _saved_arg(args_dict: Dict[str, Any], name: str, default: Any = None) -> Any:
    return args_dict.get(name, default)


def infer_image_size(dataset_name: str, image_size: Optional[int]) -> int:
    if image_size is not None:
        return int(image_size)
    defaults = {
        'mnist': 28,
        'cifar10': 32,
        'svhn': 32,
        'imagenet': 84,
        'custom': 256,
        'ffhq': 256,
    }
    if dataset_name not in defaults:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    return defaults[dataset_name]


class ReplicateChannels:
    def __init__(self, num_output_channels: int = 3):
        self.num_output_channels = num_output_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            return x
        if x.size(0) == self.num_output_channels:
            return x
        if x.size(0) == 1:
            return x.repeat(self.num_output_channels, 1, 1)
        raise ValueError(f'Cannot replicate tensor with shape {tuple(x.shape)} to {self.num_output_channels} channels.')


class InputCenterNormalize:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x - 0.5


def build_transform(dataset_name: str, image_size: int):
    from torchvision import transforms
    common_tail = [transforms.ToTensor(), ReplicateChannels(3), InputCenterNormalize()]
    if dataset_name in {'custom', 'imagenet', 'ffhq'}:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            *common_tail,
        ])
    if dataset_name in {'cifar10', 'mnist', 'svhn'}:
        return transforms.Compose(common_tail)
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def build_dataloader(saved_args: Dict[str, Any],
                     split: str,
                     data_dir: Optional[str] = None,
                     dataset_dir_name: Optional[str] = None,
                     image_size: Optional[int] = None,
                     batch_size: Optional[int] = None,
                     num_workers: Optional[int] = None,
                     shuffle: Optional[bool] = None) -> DataLoader:
    from torchvision import datasets

    dataset_name = _saved_arg(saved_args, 'dataset')
    if dataset_name is None:
        raise ValueError("args.json is missing required field 'dataset'.")

    data_dir = data_dir if data_dir is not None else _saved_arg(saved_args, 'data_dir', './data')
    dataset_dir_name = dataset_dir_name if dataset_dir_name is not None else _saved_arg(saved_args, 'dataset_dir_name')
    image_size = infer_image_size(dataset_name, image_size if image_size is not None else _saved_arg(saved_args, 'image_size'))
    transform = build_transform(dataset_name, image_size)

    root_name = dataset_name if dataset_name != 'custom' else dataset_dir_name
    if root_name is None:
        raise ValueError('dataset_dir_name must be provided for custom datasets.')
    dataset_root = os.path.join(data_dir, root_name)
    if dataset_name in {'imagenet', 'custom', 'ffhq'}:
        if split == 'train':
            dataset_root = os.path.join(dataset_root, 'train')
        else:
            dataset_root = os.path.join(dataset_root, 'val')

    datasets_classes = {
        'custom': datasets.ImageFolder,
        'imagenet': datasets.ImageFolder,
        'ffhq': datasets.ImageFolder,
        'cifar10': datasets.CIFAR10,
        'mnist': datasets.MNIST,
        'svhn': datasets.SVHN,
    }
    dataset_train_args = {
        'custom': {},
        'imagenet': {},
        'ffhq': {},
        'cifar10': {'train': True, 'download': True},
        'mnist': {'train': True, 'download': True},
        'svhn': {'split': 'train', 'download': True},
    }
    dataset_test_args = {
        'custom': {},
        'imagenet': {},
        'ffhq': {},
        'cifar10': {'train': False, 'download': True},
        'mnist': {'train': False, 'download': True},
        'svhn': {'split': 'test', 'download': True},
    }

    dataset = datasets_classes[dataset_name](
        dataset_root,
        transform=transform,
        **(dataset_train_args[dataset_name] if split == 'train' else dataset_test_args[dataset_name]),
    )
    batch_size = int(batch_size if batch_size is not None else _saved_arg(saved_args, 'batch_size', 128))
    num_workers = int(num_workers if num_workers is not None else _saved_arg(saved_args, 'num_workers', 4))
    pin_memory = torch.cuda.is_available()
    if shuffle is None:
        shuffle = split == 'train'
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


def read_saved_args(model_dir: str) -> Dict[str, Any]:
    args_path = os.path.join(model_dir, 'args.json')
    if not os.path.isfile(args_path):
        raise FileNotFoundError(f'Could not find args.json under model directory: {args_path}')
    with open(args_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def resolve_device(device: Optional[str]) -> torch.device:
    if device and device != 'auto':
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_checkpoint(model_dir: str, checkpoint: Optional[str] = None) -> str:
    candidates = []
    if checkpoint:
        checkpoint_path = checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(model_dir, checkpoint_path)
        candidates.append(checkpoint_path)
    else:
        candidates.append(os.path.join(model_dir, 'best_val.pth'))
        ckpt_dir = os.path.join(model_dir, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            names = [name for name in os.listdir(ckpt_dir) if name.endswith(('.pth', '.pt', '.pth.tar'))]
            names.sort()
            candidates.extend(os.path.join(ckpt_dir, name) for name in reversed(names))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f'Could not find a checkpoint under {model_dir}. Checked: {candidates}')


def _extract_state_dict(raw_checkpoint: Any) -> Dict[str, Any]:
    if isinstance(raw_checkpoint, dict):
        for key in ('state_dict', 'model_state_dict', 'model', 'net'):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                raw_checkpoint = value
                break
    if not isinstance(raw_checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint format: {type(raw_checkpoint)!r}')
    state_dict = dict(raw_checkpoint)
    if state_dict and all(str(key).startswith('module.') for key in state_dict.keys()):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def _parse_float_list(text_or_values: Optional[Any], expected_len: int, name: str) -> Optional[List[float]]:
    if text_or_values is None:
        return None
    if isinstance(text_or_values, str):
        tokens = text_or_values.replace(',', ' ').split()
    else:
        tokens = list(text_or_values)
    values = [float(token) for token in tokens]
    if len(values) != expected_len:
        raise ValueError(f'{name} must contain exactly {expected_len} values, got {len(values)}.')
    return values


def _extract_component_count(state_dict: Dict[str, Any]) -> int:
    pattern = re.compile(r'^components\.(\d+)\.')
    indices = set()
    for key in state_dict.keys():
        match = pattern.match(str(key))
        if match:
            indices.add(int(match.group(1)))
    return 0 if not indices else max(indices) + 1


def _extract_single_curvature(saved_args: Dict[str, Any], state_dict: Dict[str, Any]) -> float:
    for key in ('manifold.k', 'curvature'):
        if key in state_dict:
            value = state_dict[key]
            return float(torch.as_tensor(value).reshape(()).cpu().item())
    for key in ('c', 'curvature'):
        value = saved_args.get(key)
        if value is not None:
            return float(value)
    return -1.0


def _extract_mixed_curvatures(saved_args: Dict[str, Any], state_dict: Dict[str, Any], count: int) -> Optional[List[float]]:
    values = []
    for idx in range(count):
        key = f'components.{idx}.manifold.k'
        if key not in state_dict:
            values = []
            break
        values.append(float(torch.as_tensor(state_dict[key]).reshape(()).cpu().item()))
    if values:
        return values
    return _parse_float_list(saved_args.get('curvatures'), count, 'curvatures')


def _extract_mixed_alphas(saved_args: Dict[str, Any], state_dict: Dict[str, Any], count: int) -> Optional[List[float]]:
    values = []
    for idx in range(count):
        key = f'components.{idx}.alpha'
        if key not in state_dict:
            values = []
            break
        values.append(float(torch.as_tensor(state_dict[key]).reshape(()).cpu().item()))
    if values:
        return values
    return _parse_float_list(saved_args.get('alphas'), count, 'alphas')


def _build_single_encoder_decoder(hidden: int, num_residual_layers: int, num_residual_hiddens: int, embedding_dim: int):
    encoder = Encoder(
        num_hiddens=hidden,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
    )
    pre_vq_conv1 = nn.Conv2d(hidden, embedding_dim, kernel_size=1, stride=1, padding=0)
    variance_scaling_(pre_vq_conv1.weight, distribution='uniform')
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
    dataset_name = _saved_arg(saved_args, 'dataset')
    default_hyperparams = {
        'custom': {'hidden': 128, 'k': 512},
        'imagenet': {'hidden': 256, 'k': 1024},
        'ffhq': {'hidden': 256, 'k': 1024},
        'cifar10': {'hidden': 128, 'k': 512},
        'mnist': {'hidden': 64, 'k': 128},
        'svhn': {'hidden': 96, 'k': 256},
    }
    model_name = _saved_arg(saved_args, 'model', 'vqcvae')
    mixed = bool(_saved_arg(saved_args, 'mixed', False))
    hidden = int(_saved_arg(saved_args, 'hidden', default_hyperparams[dataset_name]['hidden']))
    num_embeddings = int(_saved_arg(saved_args, 'k', default_hyperparams[dataset_name]['k']))
    embedding_dim = int(_saved_arg(saved_args, 'embedding_dim', 64))
    num_residual_layers = int(_saved_arg(saved_args, 'num_residual_layers', 2))
    num_residual_hiddens = int(_saved_arg(saved_args, 'num_residual_hiddens', 32))
    commitment_cost = float(_saved_arg(saved_args, 'commit_coef', 0.25))
    decay = float(_saved_arg(saved_args, 'decay', 0.99))
    use_ema = bool(_saved_arg(saved_args, 'ema', True))
    fix_curvature = bool(_saved_arg(saved_args, 'fix_curvature', False))
    fix_alpha = bool(_saved_arg(saved_args, 'fix_alpha', False))

    if model_name == 'vqvae':
        encoder, decoder, pre_vq_conv1 = _build_single_encoder_decoder(hidden, num_residual_layers, num_residual_hiddens, embedding_dim)
        model = VQVAE(
            encoder=encoder,
            decoder=decoder,
            pre_vq_conv1=pre_vq_conv1,
            data_variance=data_variance,
            num_channel=3,
            ema=use_ema,
        )
        model.embedding = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay) if use_ema else VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        return model

    if model_name != 'vqcvae':
        raise ValueError(f'Unsupported model type in args.json: {model_name!r}')

    if not mixed:
        curvature = _extract_single_curvature(saved_args, state_dict)
        encoder, decoder, pre_vq_conv1 = _build_single_encoder_decoder(hidden, num_residual_layers, num_residual_hiddens, embedding_dim)
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
        model.embedding = McVectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, model.manifold, decay) if use_ema else McVectorQuantizer(embedding_dim, num_embeddings, commitment_cost, model.manifold)
        return model

    spaces = parse_space_spec(_saved_arg(saved_args, 'spaces'))
    component_count = len(spaces)
    if component_count == 0:
        component_count = _extract_component_count(state_dict)
    curvatures = _extract_mixed_curvatures(saved_args, state_dict, component_count)
    alphas = _extract_mixed_alphas(saved_args, state_dict, component_count)

    components = []
    total_embedding_dim = 0
    for index, space in enumerate(spaces):
        current_curvature = float(space['curvature']) if curvatures is None else float(curvatures[index])
        current_alpha = 1.0 if alphas is None else float(alphas[index])
        current_embedding_dim = int(space['dimension'])
        total_embedding_dim += current_embedding_dim
        components.append(
            MixedCurvatureComponent(
                num_hiddens=hidden,
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
    decoder = Decoder(
        input_channels=total_embedding_dim,
        num_hiddens=hidden,
        num_residual_layers=num_residual_layers,
        num_residual_hiddens=num_residual_hiddens,
    )
    model = MixedCurvatureVQCVAE(components=components, decoder=decoder, data_variance=data_variance)
    model.spaces = spaces
    return model


def load_mixed_vqvae(model_dir: str,
                     checkpoint: Optional[str] = None,
                     device: Optional[str] = None,
                     data_dir: Optional[str] = None,
                     dataset_dir_name: Optional[str] = None,
                     image_size: Optional[int] = None,
                     batch_size: Optional[int] = None,
                     num_workers: Optional[int] = None) -> Tuple[nn.Module, Dict[str, Any], torch.device, float, str]:
    saved_args = read_saved_args(model_dir)
    resolved_device = resolve_device(device)
    checkpoint_path = find_checkpoint(model_dir, checkpoint)
    raw_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _extract_state_dict(raw_checkpoint)
    variance_loader = build_dataloader(
        saved_args=saved_args,
        split='train',
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
        raise RuntimeError(f'Missing keys when loading checkpoint: {missing}')
    if unexpected:
        raise RuntimeError(f'Unexpected keys when loading checkpoint: {unexpected}')
    model.to(resolved_device)
    model.eval()
    if not getattr(model, 'is_mixed', False):
        raise ValueError('The loaded model is not a mixed-curvature VQ-VAE. Please use a checkpoint trained with --mixed.')
    return model, saved_args, resolved_device, data_variance, checkpoint_path


def namespace_from_dict(values: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**values)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
