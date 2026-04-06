import argparse
import os
import time
from typing import Dict, List, Tuple

from torchvision.utils import save_image

import torch
from torch import optim

from pixelcnn_mixed.modules import ConditionalMixedPixelCNNPrior
from pixelcnn_mixed.utils import build_dataloader, load_mixed_vqvae, save_json


@torch.no_grad()
def infer_latent_shape(model, loader, device: torch.device) -> Tuple[int, int]:
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        indices = model.encode_index(images)
        return int(indices.size(-2)), int(indices.size(-1))
    raise RuntimeError('Could not infer latent shape because the dataloader is empty.')


def infer_num_classes(loader) -> int:
    dataset = loader.dataset
    if hasattr(dataset, 'classes'):
        return len(dataset.classes)
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if torch.is_tensor(targets):
            return int(torch.unique(targets).numel())
        return len(set(int(x) for x in targets))
    if hasattr(dataset, 'labels'):
        labels = dataset.labels
        if torch.is_tensor(labels):
            return int(torch.unique(labels).numel())
        return len(set(int(x) for x in labels))
    raise ValueError('Could not infer number of classes from dataset.')


def infer_codebook_sizes(model, saved_args: Dict) -> List[int]:
    sizes = []
    if hasattr(model, 'components'):
        for component in model.components:
            codebook = component.codebook
            value = getattr(codebook, 'num_embeddings', None)
            if value is None and hasattr(codebook, 'embeddings'):
                value = int(codebook.embeddings.shape[0])
            if value is None:
                value = int(saved_args.get('k', 0))
            sizes.append(int(value))
    if not sizes:
        k = int(saved_args.get('k', 0))
        if k <= 0:
            raise ValueError("args.json is missing a valid codebook size 'k'.")
        sizes = [k]
    return sizes


def train_epoch(loader, model, prior, optimizer, device: torch.device) -> Dict[str, float]:
    prior.train()
    loss_sum = 0.0
    branch_loss_sum = None
    batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        with torch.no_grad():
            indices = model.encode_index(images, is_training=False).detach()
        optimizer.zero_grad(set_to_none=True)
        loss, _, per_branch_losses = prior.loss(indices, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if branch_loss_sum is None:
            branch_loss_sum = [0.0 for _ in per_branch_losses]
        for idx, branch_loss in enumerate(per_branch_losses):
            branch_loss_sum[idx] += float(branch_loss.detach().cpu().item())
        batches += 1
    metrics = {'loss': loss_sum / max(1, batches)}
    if branch_loss_sum is not None:
        for idx, value in enumerate(branch_loss_sum):
            metrics[f'branch_{idx}_loss'] = value / max(1, batches)
    return metrics


@torch.no_grad()
def evaluate(loader, model, prior, device: torch.device) -> Dict[str, float]:
    prior.eval()
    loss_sum = 0.0
    branch_loss_sum = None
    batches = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        indices = model.encode_index(images, is_training=False).detach()
        loss, _, per_branch_losses = prior.loss(indices, labels)
        loss_sum += loss.item()
        if branch_loss_sum is None:
            branch_loss_sum = [0.0 for _ in per_branch_losses]
        for idx, branch_loss in enumerate(per_branch_losses):
            branch_loss_sum[idx] += float(branch_loss.detach().cpu().item())
        batches += 1
    metrics = {'loss': loss_sum / max(1, batches)}
    if branch_loss_sum is not None:
        for idx, value in enumerate(branch_loss_sum):
            metrics[f'branch_{idx}_loss'] = value / max(1, batches)
    return metrics


@torch.no_grad()
def sample_and_save(prior, model, labels: torch.Tensor, device: torch.device, latent_shape, output_path: str) -> None:
    prior.eval()
    labels = labels.to(device, non_blocking=True).long()
    samples = prior.generate(labels=labels, shape=latent_shape, device=device)
    recon = model.decode_samples(samples)
    save_image(torch.clamp(recon + 0.5, 0.0, 1.0).cpu(), output_path, nrow=max(1, int(labels.size(0) ** 0.5)))


@torch.no_grad()
def export_real_images(loader, output_dir: str, max_count: int) -> Tuple[int, List[int]]:
    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    labels_out: List[int] = []
    for images, labels in loader:
        images = torch.clamp(images + 0.5, 0.0, 1.0).cpu()
        labels = labels.cpu().tolist()
        for image, label in zip(images, labels):
            save_image(image, os.path.join(output_dir, f"{saved:06d}.png"))
            labels_out.append(int(label))
            saved += 1
            if saved >= max_count:
                return saved, labels_out
    return saved, labels_out


@torch.no_grad()
def export_generated_images(prior, model, device: torch.device, latent_shape, output_dir: str, label_sequence: List[int], batch_size: int) -> int:
    os.makedirs(output_dir, exist_ok=True)
    prior.eval()
    saved = 0
    total_count = len(label_sequence)
    sample_batch = max(1, batch_size)
    while saved < total_count:
        current_batch = min(sample_batch, total_count - saved)
        labels = torch.tensor(label_sequence[saved:saved + current_batch], dtype=torch.long, device=device)
        samples = prior.generate(labels=labels, shape=latent_shape, device=device)
        recon = torch.clamp(model.decode_samples(samples) + 0.5, 0.0, 1.0).cpu()
        for image in recon:
            save_image(image, os.path.join(output_dir, f"{saved:06d}.png"))
            saved += 1
    return saved


def build_eval_labels(loader, sample_count: int, num_classes: int) -> torch.Tensor:
    labels = []
    for _, batch_labels in loader:
        labels.extend(int(x) for x in batch_labels.tolist())
        if len(labels) >= sample_count:
            break
    if len(labels) < sample_count:
        labels.extend([idx % num_classes for idx in range(sample_count - len(labels))])
    return torch.tensor(labels[:sample_count], dtype=torch.long)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Conditional multi-branch class-conditional Gated PixelCNN prior for mixed-curvature VQ-VAE.')
    parser.add_argument('--model-dir', required=True, help='Directory containing args.json and the trained mixed-curvature checkpoint.')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path or path relative to --model-dir. Defaults to best_val.pth.')
    parser.add_argument('--data-dir', default=None, help='Override dataset root stored in args.json.')
    parser.add_argument('--dataset-dir-name', default=None, help='Override dataset_dir_name stored in args.json.')
    parser.add_argument('--image-size', type=int, default=None, help='Override image size stored in args.json.')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size for prior training/evaluation.')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of dataloader workers.')
    parser.add_argument('--device', default='auto', help='Computation device, e.g. cuda:0 or cpu.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the prior.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate for the prior.')
    parser.add_argument('--hidden-size-prior', type=int, default=64, help='Hidden size for PixelCNN.')
    parser.add_argument('--num-layers', type=int, default=15, help='Number of gated PixelCNN layers.')
    parser.add_argument('--output-dir', required=True, help='Where to save the prior checkpoints and samples.')
    parser.add_argument('--sample-count', type=int, default=16, help='How many samples to generate after each validation pass.')
    parser.add_argument('--generate-only', action='store_true', help='Skip training and only sample from an existing prior checkpoint.')
    parser.add_argument('--prior-checkpoint', default=None, help='Existing prior checkpoint for --generate-only or resume usage.')
    parser.add_argument('--fid-real-count', type=int, default=None, help='When --generate-only is used, export this many real training images into output-dir/train. Defaults to --sample-count.')
    parser.add_argument('--fid-generated-count', type=int, default=None, help='When --generate-only is used, export this many generated images into output-dir/generated. Defaults to --sample-count.')
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    model, saved_args, device, _, base_checkpoint = load_mixed_vqvae(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        data_dir=args.data_dir,
        dataset_dir_name=args.dataset_dir_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = build_dataloader(saved_args, 'train', args.data_dir, args.dataset_dir_name, args.image_size, args.batch_size, args.num_workers, shuffle=True)
    test_loader = build_dataloader(saved_args, 'test', args.data_dir, args.dataset_dir_name, args.image_size, args.batch_size, args.num_workers, shuffle=False)
    train_eval_loader = build_dataloader(saved_args, 'train', args.data_dir, args.dataset_dir_name, args.image_size, args.batch_size, args.num_workers, shuffle=False)

    latent_shape = infer_latent_shape(model, train_eval_loader, device)
    codebook_sizes = infer_codebook_sizes(model, saved_args)
    num_branches = len(codebook_sizes)
    num_classes = infer_num_classes(train_eval_loader)

    prior = ConditionalMixedPixelCNNPrior(codebook_sizes=codebook_sizes, hidden_size=args.hidden_size_prior, n_layers=args.num_layers, n_classes=num_classes).to(device)

    prior_checkpoint = args.prior_checkpoint or os.path.join(args.output_dir, 'conditional_prior_best.pt')
    if os.path.isfile(prior_checkpoint):
        state = torch.load(prior_checkpoint, map_location='cpu')
        prior.load_state_dict(state['state_dict'] if isinstance(state, dict) and 'state_dict' in state else state)

    metadata = {
        'prior_type': 'conditional',
        'conditioning': 'class-conditional + branch-conditional',
        'num_branches': num_branches,
        'codebook_sizes': codebook_sizes,
        'num_classes': num_classes,
        'latent_shape': list(latent_shape),
        'vqvae_model_dir': os.path.abspath(args.model_dir),
        'vqvae_checkpoint': os.path.abspath(base_checkpoint),
        'hidden_size_prior': args.hidden_size_prior,
        'num_layers': args.num_layers,
        'factorization': 'p(z1|y)p(z2|z1,y)...p(zM|z1,...,zM-1,y)',
    }
    save_json(metadata, os.path.join(args.output_dir, 'conditional_prior_meta.json'))

    if args.generate_only:
        real_count = int(args.fid_real_count) if args.fid_real_count is not None else int(args.sample_count)
        generated_count = int(args.fid_generated_count) if args.fid_generated_count is not None else int(args.sample_count)
        train_dir = os.path.join(args.output_dir, 'train')
        generated_dir = os.path.join(args.output_dir, 'generated')
        saved_real, label_sequence = export_real_images(train_eval_loader, train_dir, max_count=real_count)
        if len(label_sequence) < generated_count:
            label_sequence.extend([idx % num_classes for idx in range(generated_count - len(label_sequence))])
        saved_generated = export_generated_images(prior, model, device, latent_shape, generated_dir, label_sequence[:generated_count], args.batch_size or args.sample_count)
        preview_labels = torch.tensor(label_sequence[:min(args.sample_count, len(label_sequence))], dtype=torch.long)
        sample_and_save(prior, model, preview_labels, device, latent_shape, os.path.join(args.output_dir, 'conditional_prior_samples.png'))
        print(f'Exported {saved_real} real images to {train_dir} and {saved_generated} generated images to {generated_dir}.')
        return

    optimizer = optim.Adam(prior.parameters(), lr=args.lr)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_epoch(train_loader, model, prior, optimizer, device)
        valid_metrics = evaluate(test_loader, model, prior, device)
        print(f'[conditional prior] epoch={epoch:03d} train_loss={train_metrics["loss"]:.6f} valid_loss={valid_metrics["loss"]:.6f} time={time.time() - start:.1f}s')
        if valid_metrics['loss'] < best_loss:
            best_loss = valid_metrics['loss']
            torch.save({'state_dict': prior.state_dict(), 'metadata': metadata, 'epoch': epoch, 'valid_loss': best_loss}, prior_checkpoint)
        preview_labels = build_eval_labels(test_loader, args.sample_count, num_classes)
        sample_and_save(prior, model, preview_labels, device, latent_shape, os.path.join(args.output_dir, f'conditional_prior_samples_epoch_{epoch:03d}.png'))


if __name__ == '__main__':
    main()


# python pixelcnn_prior_conditional.py --model-dir=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test/2026-03-21_14-15-16_cifar10_vqcvae_h128_k128_emb64_noema --output-dir=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test/2026-03-21_14-15-16_cifar10_vqcvae_h128_k128_emb64_noema/prior_conditional --epochs=20 --device=cuda:1

# python pixelcnn_prior_conditional.py --model-dir=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test/2026-03-21_14-15-16_cifar10_vqcvae_h128_k128_emb64_noema --output-dir=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test/2026-03-21_14-15-16_cifar10_vqcvae_h128_k128_emb64_noema/prior_conditional --prior-checkpoint=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test/2026-03-21_14-15-16_cifar10_vqcvae_h128_k128_emb64_noema/prior_conditional/conditional_prior_best.pt --generate-only --sample-count=16 --fid-real-count=1000 --fid-generated-count=1000

#python -m pytorch_fid /mnt/data/lkm/data/reMcVQVAE_results/ablation/test/2026-03-21_14-15-16_cifar10_vqcvae_h128_k128_emb64_noema/prior_conditional/train /mnt/data/lkm/data/reMcVQVAE_results/ablation/test/2026-03-21_14-15-16_cifar10_vqcvae_h128_k128_emb64_noema/prior_conditional/generated