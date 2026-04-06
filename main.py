import argparse
import csv
import logging
import os
import random
import sys
import time
from typing import Dict, List, Tuple
import geoopt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass
        def add_image(self, *args, **kwargs):
            pass
        def add_histogram(self, *args, **kwargs):
            pass
        def add_scalars(self, *args, **kwargs):
            pass
        def close(self):
            pass


from model.auto_encoder import (
    Decoder,
    Encoder,
    VQVAE,
    VQCVAE,
    MixedCurvatureComponent,
    MixedCurvatureVQCVAE,
    parse_space_spec,
)
from model.embedding import (
    VectorQuantizer,
    VectorQuantizerEMA,
    McVectorQuantizer,
    McVectorQuantizerEMA,
    variance_scaling_,
)
from model.util import setup_logging_from_args



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

dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'ffhq': 3,
    'cifar10': 3,
    'mnist': 1,
    'svhn': 3,
}

default_hyperparams = {
    'custom': {'lr': 2e-4, 'hidden': 128, 'k': 512},
    'imagenet': {'lr': 2e-4, 'hidden': 256, 'k': 1024},
    'ffhq': {'lr': 2e-4, 'hidden': 256, 'k': 1024},
    'cifar10': {'lr': 2e-4, 'hidden': 128, 'k': 512},
    'mnist': {'lr': 1e-4, 'hidden': 64, 'k': 128},
    'svhn': {'lr': 2e-4, 'hidden': 96, 'k': 256},
}


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in {'yes', 'true', 't', '1', 'y'}:
        return True
    if v.lower() in {'no', 'false', 'f', '0', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')


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



def build_transforms(dataset_name: str, image_size: int):
    from torchvision import transforms
    common_tail = [transforms.ToTensor(), ReplicateChannels(3), InputCenterNormalize()]
    if dataset_name == 'custom':
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            *common_tail,
        ])
    if dataset_name == 'imagenet':
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            *common_tail,
        ])
    if dataset_name == 'ffhq':
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            *common_tail,
        ])
    if dataset_name in {'cifar10', 'mnist', 'svhn'}:
        return transforms.Compose(common_tail)
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def _parse_optional_float_list(values, expected_len: int, name: str):
    if values is None:
        return None
    if isinstance(values, str):
        tokens = values.replace(',', ' ').split()
    else:
        tokens = list(values)
    parsed = [float(token) for token in tokens]
    if len(parsed) != expected_len:
        raise ValueError(
            f'{name} must contain exactly {expected_len} values to match --spaces, got {len(parsed)}.'
        )
    return parsed


def get_model_curvature_values(model: nn.Module) -> List[float]:
    if hasattr(model, 'components'):
        values = []
        for component in model.components:
            values.append(float(component.manifold.k.detach().reshape(()).cpu().item()))
        return values
    if hasattr(model, 'manifold'):
        return [float(model.manifold.k.detach().reshape(()).cpu().item())]
    return []


def build_curvature_column_names(model: nn.Module) -> List[str]:
    curvature_values = get_model_curvature_values(model)
    if len(curvature_values) <= 1:
        return ['curvature']
    return [f'curvature_component_{idx}' for idx in range(len(curvature_values))]


class ConfigurableVQVAE(VQVAE):
    def __init__(self,
                 num_channels: int,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 decay: float,
                 use_ema: bool,
                 data_variance: float):
        encoder = Encoder(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        if num_channels != 3:
            raise ValueError(
                'Current auto_encoder.Encoder expects 3-channel inputs. '
                'main.py handles grayscale datasets by converting them to 3 channels in the transform pipeline.'
            )
        pre_vq_conv1 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        variance_scaling_(pre_vq_conv1.weight, distribution='uniform')
        if pre_vq_conv1.bias is not None:
            nn.init.zeros_(pre_vq_conv1.bias)

        decoder = Decoder(
            input_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            pre_vq_conv1=pre_vq_conv1,
            data_variance=data_variance,
            num_channel=num_channels,
            ema=use_ema,
        )
        self.embedding = self.embedding
        self._override_quantizer(num_embeddings, commitment_cost, decay, use_ema)

    def _override_quantizer(self,
                            num_embeddings: int,
                            commitment_cost: float,
                            decay: float,
                            use_ema: bool) -> None:
        embedding_dim = self.pre_vq_conv1.out_channels
        if use_ema:
            self.embedding = VectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                decay=decay,
            )
        else:
            self.embedding = VectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
            )


class ConfigurableVQCVAE(VQCVAE):
    def __init__(self,
                 num_channels: int,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 decay: float,
                 use_ema: bool,
                 data_variance: float,
                 curvature: float,
                 learnable_curvature: bool):
        encoder = Encoder(
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        if num_channels != 3:
            raise ValueError(
                'Current auto_encoder.Encoder expects 3-channel inputs. '
                'main.py handles grayscale datasets by converting them to 3 channels in the transform pipeline.'
            )
        pre_vq_conv1 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        variance_scaling_(pre_vq_conv1.weight, distribution='uniform')
        if pre_vq_conv1.bias is not None:
            nn.init.zeros_(pre_vq_conv1.bias)

        decoder = Decoder(
            input_channels=embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            pre_vq_conv1=pre_vq_conv1,
            data_variance=data_variance,
            num_channel=num_channels,
            ema=use_ema,
            learnable=learnable_curvature,
            curvature=curvature,
        )
        self._override_quantizer(num_embeddings, commitment_cost, decay, use_ema)

    def _override_quantizer(self,
                            num_embeddings: int,
                            commitment_cost: float,
                            decay: float,
                            use_ema: bool) -> None:
        embedding_dim = self.pre_vq_conv1.out_channels
        if use_ema:
            self.embedding = McVectorQuantizerEMA(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                decay=decay,
                manifold=self.manifold,
            )
        else:
            self.embedding = McVectorQuantizer(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost,
                manifold=self.manifold,
            )


class ConfigurableMixedCurvatureVQCVAE(MixedCurvatureVQCVAE):
    def __init__(self,
                 num_channels: int,
                 num_hiddens: int,
                 num_residual_layers: int,
                 num_residual_hiddens: int,
                 num_embeddings: int,
                 commitment_cost: float,
                 decay: float,
                 use_ema: bool,
                 data_variance: float,
                 spaces,
                 learnable_curvature: bool,
                 fix_alpha: bool,
                 alphas=None,
                 curvatures=None):
        if num_channels != 3:
            raise ValueError(
                'Current auto_encoder.Encoder expects 3-channel inputs. '
                'main.py handles grayscale datasets by converting them to 3 channels in the transform pipeline.'
            )

        parsed_spaces = parse_space_spec(spaces)
        parsed_alphas = _parse_optional_float_list(alphas, len(parsed_spaces), 'alphas')
        parsed_curvatures = _parse_optional_float_list(curvatures, len(parsed_spaces), 'curvatures')

        components = []
        total_embedding_dim = 0
        for idx, space in enumerate(parsed_spaces):
            embedding_dim = int(space['dimension'])
            curvature = float(space['curvature'])
            if parsed_curvatures is not None:
                curvature = parsed_curvatures[idx]
            alpha = 1.0 if parsed_alphas is None else parsed_alphas[idx]

            total_embedding_dim += embedding_dim
            components.append(
                MixedCurvatureComponent(
                    num_hiddens=embedding_dim,
                    num_residual_layers=num_residual_layers,
                    num_residual_hiddens=num_residual_hiddens,
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    commitment_cost=commitment_cost,
                    decay=decay,
                    use_ema=use_ema,
                    curvature=curvature,
                    learnable_curvature=learnable_curvature,
                    alpha=alpha,
                    learnable_alpha=not fix_alpha,
                )
            )

        decoder = Decoder(
            input_channels=total_embedding_dim,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )
        super().__init__(components=components, decoder=decoder, data_variance=data_variance)
        self.spaces = parsed_spaces


class MultipleScheduler(object):
    def __init__(self, sch):
        self.schedulers = sch

    def step(self):
        for sch in self.schedulers:
            sch.step()

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op
        self.scheduler = []

    def get_learning_rates(self):
        return [op.param_groups[0]['lr'] for op in self.optimizers]

    def zero_grad(self, set_to_none: bool = False):
        for op in self.optimizers:
            op.zero_grad(set_to_none=set_to_none)

    def step(self):
        for op in self.optimizers:
            op.step()

    def optim_lr_schedulaer_StepLR(self, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        for op in self.optimizers:
            self.scheduler.append(optim.lr_scheduler.StepLR(op, step_size, gamma, last_epoch, verbose))

        return MultipleScheduler(self.scheduler)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int):
        self.sum += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)



def unique_trainable_parameters(params):
    unique = []
    seen = set()
    for param in params:
        if param is None or not getattr(param, 'requires_grad', False):
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        unique.append(param)
    return unique



def set_seed(seed: int, cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def infer_image_size(dataset_name: str, arg_image_size: int) -> int:
    if arg_image_size is not None:
        return arg_image_size
    defaults = {
        'mnist': 28,
        'cifar10': 32,
        'svhn': 32,
        'imagenet': 84,
        'custom': 256,
        'ffhq': 256,
    }
    return defaults[dataset_name]



def build_dataloaders(args) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {'num_workers': args.num_workers}
    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name
    dataset_train_dir = os.path.join(args.data_dir, dataset_dir_name)
    dataset_test_dir = os.path.join(args.data_dir, dataset_dir_name)
    if args.dataset in {'imagenet', 'custom','ffhq'}:
        dataset_train_dir = os.path.join(dataset_train_dir, 'train')
        dataset_test_dir = os.path.join(dataset_test_dir, 'val')

    transform = build_transforms(args.dataset, args.image_size)

    from torchvision import datasets
    datasets_classes = {
        "custom": datasets.ImageFolder,
        "imagenet": datasets.ImageFolder,
        "ffhq": datasets.ImageFolder,
        "cifar10": datasets.CIFAR10,
        "mnist": datasets.MNIST,
        "svhn": datasets.SVHN,
    }

    train_dataset = datasets_classes[args.dataset](
        dataset_train_dir,
        transform=transform,
        **dataset_train_args[args.dataset],
    )
    test_dataset = datasets_classes[args.dataset](
        dataset_test_dir,
        transform=transform,
        **dataset_test_args[args.dataset],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader


@torch.no_grad()
def estimate_data_variance(loader: torch.utils.data.DataLoader,
                           device: torch.device,
                           max_batches: int = 32) -> float:
    sq_sum = 0.0
    sum_ = 0.0
    count = 0
    for idx, (data, _) in enumerate(loader):
        if idx >= max_batches:
            break
        data = data.to(device, non_blocking=True)
        sq_sum += torch.sum(data * data).item()
        sum_ += torch.sum(data).item()
        count += data.numel()
    mean = sum_ / max(1, count)
    variance = sq_sum / max(1, count) - mean * mean
    return max(variance, 1e-6)



def compute_loss_breakdown(model: nn.Module,
                           inputs: torch.Tensor,
                           output: Dict[str, torch.Tensor],
                           args) -> Dict[str, torch.Tensor]:
    del inputs
    vq_output = output['vq_output']
    recon_error = output['recon_error']
    decorrelation = output.get('decorrelation_loss', torch.zeros((), device=recon_error.device, dtype=recon_error.dtype))
    if not (getattr(args, 'decorrelation', False) and getattr(model, 'is_mixed', False)):
        decorrelation = torch.zeros_like(decorrelation)
    else:
        decorrelation = 0.1 * decorrelation
    total = recon_error + vq_output['loss'] + decorrelation

    if getattr(model, 'is_mixed', False):
        vq_only = vq_output.get('q_latent_loss', torch.zeros((), device=recon_error.device, dtype=recon_error.dtype))
        commitment = vq_output.get('e_latent_loss', torch.zeros((), device=recon_error.device, dtype=recon_error.dtype))
    elif isinstance(model.embedding, (McVectorQuantizer, McVectorQuantizerEMA)):
        vq_only = vq_output.get('q_latent_loss', torch.zeros((), device=recon_error.device, dtype=recon_error.dtype))
        commitment = vq_output.get('e_latent_loss', torch.zeros((), device=recon_error.device, dtype=recon_error.dtype))
    else:
        z = output['z']
        quantized = vq_output['quantize'].detach()
        commitment = F.mse_loss(quantized, z)
        if isinstance(model.embedding, VectorQuantizerEMA):
            vq_only = torch.zeros_like(commitment)
        else:
            vq_only = F.mse_loss(quantized, z.detach())

    return {
        'loss': total,
        'recon_error': recon_error,
        'vq_loss': vq_only,
        'commitment_loss': commitment,
        'perplexity': vq_output['perplexity'],
        'decorrelation_loss': decorrelation,
    }



def move_to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    return batch.to(device, non_blocking=True)



def train_one_epoch(epoch: int,
                    model: nn.Module,
                    train_loader: torch.utils.data.DataLoader,
                    optimizer,
                    device: torch.device,
                    args,
                    writer: SummaryWriter,
                    save_path: str) -> Dict[str, float]:
    model.train()
    meter_keys = ['loss', 'recon_error', 'vq_loss', 'commitment_loss', 'perplexity']
    if getattr(args, 'decorrelation', False):
        meter_keys.append('decorrelation_loss')
    meters = {k: AverageMeter() for k in meter_keys}
    start_time = time.time()
    last_batch = None
    last_output = None

    for batch_idx, (data, _) in enumerate(train_loader):
        data = move_to_device(data, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data, is_training=True)
        metrics = compute_loss_breakdown(model, data, output, args)
        metrics['loss'].backward()
        optimizer.step()

        batch_size = data.size(0)
        for key, meter in meters.items():
            meter.update(metrics[key].item(), batch_size)

        if batch_idx % args.log_interval == 0:
            logging.info(
                'Train Epoch: %d [%5d/%5d (%2d%%)] time: %.2f loss: %.6f recon: %.6f vq: %.6f commit: %.6f perplexity: %.4f decor: %.6f',
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                int(100.0 * batch_idx / max(1, len(train_loader))),
                time.time() - start_time,
                meters['loss'].avg,
                meters['recon_error'].avg,
                meters['vq_loss'].avg,
                meters['commitment_loss'].avg,
                meters['perplexity'].avg,
                meters.get('decorrelation_loss', AverageMeter()).avg,
            )
            start_time = time.time()

        last_batch = data
        last_output = output
        if args.dataset in {'imagenet', 'custom'} and batch_idx * len(data) > args.max_epoch_samples:
            break

    if last_batch is not None and last_output is not None:
        save_reconstructed_images(last_batch, epoch, last_output['x_recon'], save_path, 'reconstruction_train')
        write_images(last_batch, last_output['x_recon'], writer, 'train')
        write_codebook_histogram(last_output, writer, epoch, 'train')

    epoch_metrics = {f'{k}_train': meter.avg for k, meter in meters.items()}
    logging.info(
        '====> Epoch: %d loss_train: %.6f recon_error_train: %.6f vq_loss_train: %.6f commitment_loss_train: %.6f perplexity_train: %.4f decorrelation_loss_train: %.6f',
        epoch,
        epoch_metrics['loss_train'],
        epoch_metrics['recon_error_train'],
        epoch_metrics['vq_loss_train'],
        epoch_metrics['commitment_loss_train'],
        epoch_metrics['perplexity_train'],
        epoch_metrics.get('decorrelation_loss_train', 0.0),
    )
    return epoch_metrics


@torch.no_grad()
def evaluate(epoch: int,
             model: nn.Module,
             test_loader: torch.utils.data.DataLoader,
             device: torch.device,
             args,
             writer: SummaryWriter,
             save_path: str) -> Dict[str, float]:
    model.eval()
    meter_keys = ['loss', 'recon_error', 'vq_loss', 'commitment_loss', 'perplexity']
    if getattr(args, 'decorrelation', False):
        meter_keys.append('decorrelation_loss')
    meters = {k: AverageMeter() for k in meter_keys}
    last_batch = None
    last_output = None

    for batch_idx, (data, _) in enumerate(test_loader):
        data = move_to_device(data, device)
        output = model(data, is_training=False)
        metrics = compute_loss_breakdown(model, data, output, args)
        batch_size = data.size(0)
        for key, meter in meters.items():
            meter.update(metrics[key].item(), batch_size)
        last_batch = data
        last_output = output
        if args.dataset in {'imagenet', 'custom'} and batch_idx * len(data) > 1000:
            break

    if last_batch is not None and last_output is not None:
        write_images(last_batch, last_output['x_recon'], writer, 'test')
        save_reconstructed_images(last_batch, epoch, last_output['x_recon'], save_path, 'reconstruction_test')
        write_codebook_histogram(last_output, writer, epoch, 'test')
        save_checkpoint(model, epoch, save_path)

    eval_metrics = {f'{k}_test': meter.avg for k, meter in meters.items()}
    logging.info(
        '====> Test set: Epoch: %d loss_test: %.6f recon_error_test: %.6f vq_loss_test: %.6f commitment_loss_test: %.6f perplexity_test: %.4f decorrelation_loss_test: %.6f',
        epoch,
        eval_metrics['loss_test'],
        eval_metrics['recon_error_test'],
        eval_metrics['vq_loss_test'],
        eval_metrics['commitment_loss_test'],
        eval_metrics['perplexity_test'],
        eval_metrics.get('decorrelation_loss_test', 0.0),
    )
    return eval_metrics



def write_codebook_histogram(output: Dict[str, torch.Tensor], writer: SummaryWriter, epoch: int, split: str) -> None:
    indices = output['vq_output']['encoding_indices'].reshape(-1).detach().cpu()
    writer.add_histogram(f'codebook_indices/{split}', indices, global_step=epoch)



def write_images(data: torch.Tensor, recon: torch.Tensor, writer: SummaryWriter, suffix: str) -> None:
    original = torch.clamp(data + 0.5, 0.0, 1.0)
    reconstructed = torch.clamp(recon + 0.5, 0.0, 1.0)
    from torchvision.utils import make_grid
    writer.add_image(f'original/{suffix}', make_grid(original[:6]))
    writer.add_image(f'reconstructed/{suffix}', make_grid(reconstructed[:6]))



def save_reconstructed_images(data: torch.Tensor,
                              epoch: int,
                              recon: torch.Tensor,
                              save_path: str,
                              name: str) -> None:
    n = min(data.size(0), 8)
    comparison = torch.cat([data[:n], recon[:n]])
    from torchvision.utils import save_image
    save_image(
        comparison.cpu(),
        os.path.join(save_path, f'{name}_{epoch}.png'),
        nrow=n,
        normalize=True,
        value_range=(-0.5, 0.5),
    )



def save_checkpoint(model: nn.Module, epoch: int, save_path: str) -> None:
    ckpt_dir = os.path.join(save_path, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'model_{epoch}.pth'))



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PyTorch VQ-VAE training entrypoint')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vqvae', choices=['vqvae', 'vqcvae'], help='model variant to train')
    model_parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training')
    model_parser.add_argument('--hidden', type=int, metavar='N', help='base hidden channels (maps to num_hiddens)')
    model_parser.add_argument('--mixed', action='store_true', default=False, help='enable mixed-curvature VQCVAE with one encoder/codebook per space and a shared decoder')
    model_parser.add_argument('-c', '--curvature', type=float, dest='c', metavar='C', help='curvature used by VQCVAE; internal stereographic curvature is c')
    model_parser.add_argument('--spaces', type=str, metavar='spaces and dimension', help='mixed-space specification for --mixed, e.g. "1,32 0,32 -1,32"')
    model_parser.add_argument('--fix_alpha', action='store_true', default=False, help='freeze per-space alpha scaling factors when using --mixed')
    model_parser.add_argument('--fix_curvature', '-fix', action='store_true', default=False, help='freeze curvature when using --model=vqcvae')
    model_parser.add_argument('--pre', action='store_true', default=False, help='kept for backward-compatible argparse; not used by current implementation')
    model_parser.add_argument('-k', '--dict-size', type=int, dest='k', metavar='K', help='number of codebook embeddings')
    model_parser.add_argument('--lr', type=float, default=None, help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=1.0, help='kept for backward-compatible argparse; current model already includes VQ loss internally')
    model_parser.add_argument('--commit_coef', type=float, default=0.25, help='commitment coefficient for the vector quantizer')
    model_parser.add_argument('--kl_coef', type=float, default=None, help='kept for backward-compatible argparse; not used by current implementation')
    model_parser.add_argument('--embedding-dim', type=int, default=64, help='embedding dimension for the codebook and pre_vq_conv1 output channels')
    model_parser.add_argument('--num-residual-layers', type=int, default=2, help='number of residual blocks in encoder/decoder')
    model_parser.add_argument('--num-residual-hiddens', type=int, default=32, help='hidden width inside each residual block')
    model_parser.add_argument('--ema', type=str2bool, default=True, help='use EMA vector quantizer')
    model_parser.add_argument('--decay', type=float, default=0.99, help='EMA decay used when --ema=true')
    model_parser.add_argument('--manifold-lr', type=float, default=None, help='optional learning rate for manifold/codebook parameters when --model=vqcvae')
    model_parser.add_argument('--alphas', type=str, default=None, help='optional per-space alpha override for --mixed, e.g. "1 0.5 1"')
    model_parser.add_argument('--curvatures', type=str, default=None, help='optional per-space curvature override for --mixed, e.g. "1 0 -1"')
    model_parser.add_argument('--decorrelation', action='store_true', default=False, help='enable branch decorrelation loss for --mixed models')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'imagenet', 'custom', 'svhn', 'ffhq'], help='dataset to use')
    training_parser.add_argument('--dataset_dir_name', default='/mnt/data/lkm/data', help='dir name inside --data-dir when dataset==custom')
    training_parser.add_argument('--data-dir', default='/mnt/data/lkm/data', help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train')
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000, help='max number of samples per epoch for large folder datasets')
    training_parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    training_parser.add_argument('--gpus', default='0', help='GPU ids, e.g. 0 or 0,1')
    training_parser.add_argument('--num-workers', type=int, default=4, help='number of dataloader workers')
    training_parser.add_argument('--image-size', type=int, default=None, help='override dataset image size for folder datasets')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    logging_parser.add_argument('--save-name', default='', help='save folder name')
    logging_parser.add_argument('--data-format', default='json', help='kept for backward-compatible argparse')
    return parser



def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.lr = args.lr or default_hyperparams[args.dataset]['lr']
    args.hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    args.k = args.k or default_hyperparams[args.dataset]['k']
    args.image_size = infer_image_size(args.dataset, args.image_size)

    if args.mixed and args.model != 'vqcvae':
        raise ValueError('--mixed is only supported when --model=vqcvae.')
    if args.mixed and not args.spaces:
        raise ValueError('--mixed requires --spaces, for example --spaces="1,32 0,32 -1,32".')

    save_path = setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    if args.cuda:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        device = torch.device(f'cuda:{args.gpus[0]}')
    else:
        device = torch.device('cpu')

    set_seed(args.seed, args.cuda)
    train_loader, test_loader = build_dataloaders(args)
    data_variance = estimate_data_variance(train_loader, device)
    logging.info('Estimated training data variance: %.8f', data_variance)

    if args.model == 'vqvae':
        model = ConfigurableVQVAE(
            num_channels=3,
            num_hiddens=args.hidden,
            num_residual_layers=args.num_residual_layers,
            num_residual_hiddens=args.num_residual_hiddens,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.k,
            commitment_cost=args.commit_coef,
            decay=args.decay,
            use_ema=args.ema,
            data_variance=data_variance,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.model == 'vqcvae':
        if args.mixed:
            model = ConfigurableMixedCurvatureVQCVAE(
                num_channels=3,
                num_hiddens=args.hidden,
                num_residual_layers=args.num_residual_layers,
                num_residual_hiddens=args.num_residual_hiddens,
                num_embeddings=args.k,
                commitment_cost=args.commit_coef,
                decay=args.decay,
                use_ema=args.ema,
                data_variance=data_variance,
                spaces=args.spaces,
                learnable_curvature=not args.fix_curvature,
                fix_alpha=args.fix_alpha,
                alphas=args.alphas,
                curvatures=args.curvatures,
            ).to(device)

            euclidean_params = list(model.decoder.parameters())
            manifold_params = []
            alpha_params = []
            for component in model.components:
                euclidean_params.extend(list(component.encoder.parameters()))
                euclidean_params.extend(list(component.pre_vq_conv1.parameters()))
                manifold_params.extend(list(component.manifold.parameters()))
                manifold_params.extend(list(component.embedding.parameters()))
                if component.alpha.requires_grad:
                    alpha_params.append(component.alpha)

            euclidean_params = unique_trainable_parameters(euclidean_params)
            manifold_params = unique_trainable_parameters(manifold_params)
            alpha_params = unique_trainable_parameters(alpha_params)

            optimizers = []
            if euclidean_params:
                optimizers.append(optim.Adam(euclidean_params, lr=args.lr))
            if manifold_params:
                has_positive_curvature = any(float(component.manifold.k.detach().reshape(()).item()) > 0 for component in model.components)
                manifold_lr = args.manifold_lr if args.manifold_lr is not None else min(args.lr, 5e-5 if has_positive_curvature else args.lr)
                optimizers.append(geoopt.optim.RiemannianAdam(manifold_params, lr=manifold_lr))
            if alpha_params:
                optimizers.append(optim.Adam(alpha_params, lr=10.0 * args.lr))
            optimizer = MultipleOptimizer(*optimizers)
        else:
            curvature = args.c if args.c is not None else -1.0
            model = ConfigurableVQCVAE(
                num_channels=3,
                num_hiddens=args.hidden,
                num_residual_layers=args.num_residual_layers,
                num_residual_hiddens=args.num_residual_hiddens,
                embedding_dim=args.embedding_dim,
                num_embeddings=args.k,
                commitment_cost=args.commit_coef,
                decay=args.decay,
                use_ema=args.ema,
                data_variance=data_variance,
                curvature=curvature,
                learnable_curvature=not args.fix_curvature,
            ).to(device)
            euclidean_params = []
            manifold_params = []
            manifold_param_ids = set()

            manifold_params.extend(list(model.manifold.parameters()))
            manifold_param_ids.update(id(p) for p in manifold_params)

            embedding_params = [p for p in model.embedding.parameters()]
            manifold_params.extend(embedding_params)
            manifold_param_ids.update(id(p) for p in embedding_params)

            euclidean_params = [p for p in model.parameters() if id(p) not in manifold_param_ids]

            optimizers = []
            if euclidean_params:
                optimizers.append(optim.Adam(euclidean_params, lr=args.lr))
            if manifold_params:
                manifold_lr = args.manifold_lr if args.manifold_lr is not None else min(args.lr, 5e-5 if curvature > 0 else args.lr)
                optimizers.append(geoopt.optim.RiemannianAdam(manifold_params, lr=manifold_lr))
            optimizer = MultipleOptimizer(*optimizers)
    else:
        raise ValueError(f'Unsupported model: {args.model}')
    if isinstance(optimizer, MultipleOptimizer):
        scheduler = optimizer.optim_lr_schedulaer_StepLR(10 if args.dataset == 'imagenet' else 5, 0.5)
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10 if args.dataset == 'imagenet' else 5,
            gamma=0.5,
        )


    # optimizer_dec = optim.Adam(model.decoder.parameters(), lr=args.lr)
    # optimizer_manifold = geoopt.optim.RiemannianAdam(model.manifold.parameters(), lr=args.lr)
    # optimizer_enc = optim.Adam(model.encoder.parameters(), lr=args.lr)
    # optimizer = MultipleOptimizer(optimizer_dec, optimizer_manifold, optimizer_enc)

    # scheduler = optimizer.optim_lr_schedulaer_StepLR(10 if args.dataset == 'imagenet' else 5, 0.5, )




    

    csv_path = os.path.join(save_path, 'train.csv')
    curvature_column_names = build_curvature_column_names(model)
    metric_column_names = [
        'epoch', 'loss_test', 'recon_error_test', 'vq_loss_test', 'commitment_loss_test', 'perplexity_test'
    ]
    with open(csv_path, mode='w', newline='') as file:
        csv.writer(file).writerow(metric_column_names + curvature_column_names)

    best_val_loss = float('inf')
    best_val_epoch = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, device, args, writer, save_path)
        test_metrics = evaluate(epoch, model, test_loader, device, args, writer, save_path)

        for key in ['loss', 'recon_error', 'vq_loss', 'commitment_loss', 'perplexity']:
            writer.add_scalars(key, {
                'train': train_metrics[f'{key}_train'],
                'test': test_metrics[f'{key}_test'],
            }, global_step=epoch)

        current_val_loss = test_metrics['recon_error_test']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_val_epoch = epoch
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_model_state, os.path.join(save_path, 'best_val.pth'))

        current_curvature_values = get_model_curvature_values(model)
        with open(csv_path, mode='a', newline='') as file:
            csv.writer(file).writerow([
                epoch,
                test_metrics['loss_test'],
                test_metrics['recon_error_test'],
                test_metrics['vq_loss_test'],
                test_metrics['commitment_loss_test'],
                test_metrics['perplexity_test'],
                *current_curvature_values,
            ])
        scheduler.step()

    final_curvature_values = get_model_curvature_values(model)
    with open(csv_path, mode='a', newline='') as file:
        writerow = csv.writer(file)
        writerow.writerow([])
        writerow.writerow(['best_val_epoch', 'best_val_recon_error'])
        writerow.writerow([best_val_epoch, best_val_loss])
        writerow.writerow([])
        writerow.writerow(['final_curvatures'] + curvature_column_names)
        writerow.writerow(['final'] + final_curvature_values)

    if best_model_state is not None:
        logging.info('Best validation reconstruction error %.6f at epoch %d', best_val_loss, best_val_epoch)
    writer.close()


if __name__ == '__main__':
    main(sys.argv[1:])



# python main.py --dataset=cifar10 --model=vqcvae --data-dir=/mnt/data/lkm/data/datasets --results-dir=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test_geometric_perplexity_ema --epochs=50 --batch-size=128 --hidden=128 -k=128 --embedding-dim=64 --num-residual-layers=2 --num-residual-hiddens=32 --commit_coef=0.25 --ema=true --decay=0.99 --gpus=6 -fix --curvature=-1

# python main.py --dataset=cifar10 --model=vqcvae --data-dir=/mnt/data/lkm/data/datasets --epochs=50  --results-dir=/mnt/data/lkm/data/reMcVQVAE_results/learnable/ --batch-size=128 --hidden=128 -k=128 --mixed --spaces="1,32 -1,32" --num-residual-layers=2 --num-residual-hiddens=32 --commit_coef=0.25 --ema=false --decay=0.99 --gpus=6 --fix_alpha -fix --decorrelation