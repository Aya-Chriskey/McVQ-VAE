import argparse
import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from reconstruction import (
    InferenceContext,
    _unnormalize,
    build_dataloader,
    load_inference_context,
    make_comparison_grid,
    prepare_output_dirs,
    save_tensor_image,
)

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - environment-specific
    skimage_ssim = None



def _to_scalar(value: Any) -> float:
    return float(torch.as_tensor(value).detach().reshape(()).cpu().item())



def _batch_histograms(encoding_indices: torch.Tensor, codebook_size: int) -> List[torch.Tensor]:
    if encoding_indices.dim() == 3:
        return [torch.bincount(encoding_indices.reshape(-1).cpu(), minlength=codebook_size).to(torch.float64)]
    if encoding_indices.dim() == 4:
        histograms = []
        for component_index in range(encoding_indices.size(1)):
            component_indices = encoding_indices[:, component_index, ...]
            histograms.append(torch.bincount(component_indices.reshape(-1).cpu(), minlength=codebook_size).to(torch.float64))
        return histograms
    raise ValueError(
        "encoding_indices must have shape [B, H, W] or [B, num_components, H, W], "
        f"got {tuple(encoding_indices.shape)}"
    )



def _safe_perplexity(histogram: torch.Tensor) -> float:
    total = float(histogram.sum().item())
    if total <= 0:
        return 0.0
    probs = histogram / total
    probs = probs[probs > 0]
    return float(torch.exp(-(probs * torch.log(probs)).sum()).item())



def _usage_ratio(histogram: torch.Tensor) -> float:
    if histogram.numel() == 0:
        return 0.0
    return float((histogram > 0).sum().item() / histogram.numel())



def _ssim_single(gt: torch.Tensor, recon: torch.Tensor) -> float:
    if skimage_ssim is None:
        return float("nan")
    gt_np = gt.permute(1, 2, 0).cpu().numpy()
    recon_np = recon.permute(1, 2, 0).cpu().numpy()
    height, width = gt_np.shape[:2]
    min_side = min(height, width)
    if min_side < 3:
        return float("nan")
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)
    try:
        return float(skimage_ssim(gt_np, recon_np, data_range=1.0, channel_axis=-1, win_size=win_size))
    except TypeError:  # older scikit-image
        return float(skimage_ssim(gt_np, recon_np, data_range=1.0, multichannel=True, win_size=win_size))



def _run_fid(gt_dir: str, rec_dir: str) -> Tuple[Optional[float], Optional[str]]:
    cmd = [sys.executable, "-m", "pytorch_fid", gt_dir, rec_dir]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        combined_output = "\n".join(part for part in [stdout, stderr] if part)
        return None, f"pytorch_fid failed with exit code {exc.returncode}: {combined_output}"
    except Exception as exc:
        return None, str(exc)

    stdout = (completed.stdout or b"").decode("utf-8", errors="replace")
    stderr = (completed.stderr or b"").decode("utf-8", errors="replace")
    combined_output = "\n".join(part for part in [stdout, stderr] if part)
    match = re.search(r"FID:\s*([0-9eE+\-.]+)", combined_output)
    if match:
        return float(match.group(1)), None
    return None, f"Could not parse FID output from: {combined_output}"



def _collect_model_metadata(context: InferenceContext) -> Dict[str, Any]:
    model = context.model
    metadata: Dict[str, Any] = {
        "checkpoint_path": os.path.abspath(context.checkpoint_path),
        "data_variance": context.data_variance,
        "model": context.saved_args.get("model"),
        "mixed": bool(context.saved_args.get("mixed", False)),
    }
    if getattr(model, "is_mixed", False):
        metadata["curvatures"] = [
            float(component.manifold.k.detach().reshape(()).cpu().item())
            for component in model.components
        ]
        metadata["alphas"] = [
            float(component.alpha.detach().reshape(()).cpu().item())
            for component in model.components
        ]
    elif hasattr(model, "manifold"):
        metadata["curvature"] = float(model.manifold.k.detach().reshape(()).cpu().item())
    return metadata



def _save_histogram_plot(histogram: torch.Tensor, path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-specific
        print(f"[evaluation] skip plotting {path}: {exc}")
        return

    values = histogram.cpu().numpy()
    x_pos = np.arange(values.shape[0])
    plt.figure(figsize=(10, 4))
    plt.bar(x_pos, values)
    plt.title(title)
    plt.xlabel("Codebook index")
    plt.ylabel("Usage count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


@torch.inference_mode()
def run_evaluation(model_dir: str,
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
                   save_image_dirs: bool = False,
                   reconstruction_output_dir: Optional[str] = None,
                   gt_dir_name: str = "train",
                   rec_dir_name: str = "rec",
                   comparison_dir_name: str = "comparison",
                   overwrite: bool = False,
                   save_comparison_grids: bool = False,
                   compute_fid: bool = False) -> Dict[str, Any]:
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

    metrics_output_dir = output_dir or os.path.join(model_dir, "evaluation")
    os.makedirs(metrics_output_dir, exist_ok=True)

    save_image_dirs = save_image_dirs or compute_fid
    gt_dir = rec_dir = comparison_dir = None
    if save_image_dirs:
        reconstruction_output_dir = reconstruction_output_dir or model_dir
        gt_dir, rec_dir, comparison_dir = prepare_output_dirs(
            output_dir=reconstruction_output_dir,
            gt_dir_name=gt_dir_name,
            rec_dir_name=rec_dir_name,
            comparison_dir_name=comparison_dir_name,
            overwrite=overwrite,
        )

    codebook_size = int(context.saved_args.get("k", 0))
    if codebook_size <= 0:
        raise ValueError("args.json is missing a valid codebook size 'k'.")

    sums = {
        "loss": 0.0,
        "recon_error": 0.0,
        "vq_loss": 0.0,
        "q_latent_loss": 0.0,
        "e_latent_loss": 0.0,
        "forward_perplexity": 0.0,
        "l1": 0.0,
        "mse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    ssim_count = 0
    sample_count = 0
    comparison_grid_count = 0
    histograms: Optional[List[torch.Tensor]] = None

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
        vq_output = output["vq_output"]

        batch_size_now = inputs.size(0)
        sums["loss"] += _to_scalar(output["loss"]) * batch_size_now
        sums["recon_error"] += _to_scalar(output["recon_error"]) * batch_size_now
        sums["vq_loss"] += _to_scalar(vq_output.get("loss", 0.0)) * batch_size_now
        sums["q_latent_loss"] += _to_scalar(vq_output.get("q_latent_loss", 0.0)) * batch_size_now
        sums["e_latent_loss"] += _to_scalar(vq_output.get("e_latent_loss", 0.0)) * batch_size_now
        sums["forward_perplexity"] += _to_scalar(vq_output.get("perplexity", 0.0)) * batch_size_now

        inputs_vis = _unnormalize(inputs)
        recon_vis = _unnormalize(recon)
        l1 = torch.mean(torch.abs(inputs_vis - recon_vis), dim=(1, 2, 3))
        mse = torch.mean((inputs_vis - recon_vis) ** 2, dim=(1, 2, 3))
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        sums["l1"] += float(l1.sum().item())
        sums["mse"] += float(mse.sum().item())
        sums["psnr"] += float(psnr.sum().item())

        for local_index in range(batch_size_now):
            ssim_value = _ssim_single(inputs_vis[local_index], recon_vis[local_index])
            if not math.isnan(ssim_value):
                sums["ssim"] += ssim_value
                ssim_count += 1

            if save_image_dirs and gt_dir and rec_dir:
                filename = f"{sample_count + local_index:06d}.png"
                save_tensor_image(inputs[local_index], os.path.join(gt_dir, filename))
                save_tensor_image(recon[local_index], os.path.join(rec_dir, filename))

        if save_image_dirs and save_comparison_grids and comparison_dir is not None:
            grid = make_comparison_grid(inputs, recon)
            grid.save(os.path.join(comparison_dir, f"batch_{batch_index:04d}.png"))
            comparison_grid_count += 1

        batch_histograms = _batch_histograms(vq_output["encoding_indices"].detach().cpu(), codebook_size)
        if histograms is None:
            histograms = [hist.clone() for hist in batch_histograms]
        else:
            if len(histograms) != len(batch_histograms):
                raise RuntimeError(
                    f"Inconsistent number of codebook histograms: {len(histograms)} vs {len(batch_histograms)}"
                )
            for hist, batch_hist in zip(histograms, batch_histograms):
                hist += batch_hist

        sample_count += batch_size_now
        print(f"[evaluation] processed {sample_count} samples")

    if sample_count == 0:
        raise RuntimeError("No samples were evaluated. Please check dataset paths and --max-samples.")
    if histograms is None:
        raise RuntimeError("Failed to collect any codebook usage statistics.")

    mean_metrics = {key: value / sample_count for key, value in sums.items()}
    mean_metrics["ssim"] = float("nan") if ssim_count == 0 else sums["ssim"] / ssim_count

    component_perplexities = [_safe_perplexity(hist) for hist in histograms]
    component_usage_ratios = [_usage_ratio(hist) for hist in histograms]
    total_histogram = torch.stack(histograms, dim=0).sum(dim=0)
    global_perplexity = _safe_perplexity(total_histogram)
    global_usage_ratio = _usage_ratio(total_histogram)

    histogram_paths = []
    if len(histograms) == 1:
        path = os.path.join(metrics_output_dir, "codebook_usage.png")
        _save_histogram_plot(histograms[0], path, f"Codebook Usage (k={codebook_size})")
        histogram_paths.append(path)
    else:
        total_path = os.path.join(metrics_output_dir, "codebook_usage_total.png")
        _save_histogram_plot(total_histogram, total_path, f"Total Codebook Usage Across Components (k={codebook_size})")
        histogram_paths.append(total_path)
        for component_index, histogram in enumerate(histograms):
            path = os.path.join(metrics_output_dir, f"codebook_usage_component_{component_index}.png")
            _save_histogram_plot(histogram, path, f"Component {component_index} Codebook Usage (k={codebook_size})")
            histogram_paths.append(path)

    fid_value = None
    fid_error = None
    if compute_fid:
        if not (gt_dir and rec_dir):
            raise RuntimeError("FID requested but reconstruction directories were not prepared.")
        fid_value, fid_error = _run_fid(gt_dir, rec_dir)

    results: Dict[str, Any] = {
        "model_dir": os.path.abspath(model_dir),
        "split": split,
        "num_samples": sample_count,
        "metrics": {
            "loss": mean_metrics["loss"],
            "recon_error": mean_metrics["recon_error"],
            "vq_loss": mean_metrics["vq_loss"],
            "q_latent_loss": mean_metrics["q_latent_loss"],
            "e_latent_loss": mean_metrics["e_latent_loss"],
            "forward_perplexity_mean": mean_metrics["forward_perplexity"],
            "l1": mean_metrics["l1"],
            "mse": mean_metrics["mse"],
            "psnr": mean_metrics["psnr"],
            "ssim": mean_metrics["ssim"],
            "fid": fid_value,
        },
        "codebook": {
            "num_components": len(histograms),
            "global_perplexity": global_perplexity,
            "global_usage_ratio": global_usage_ratio,
            "perplexity_per_component": component_perplexities,
            "usage_ratio_per_component": component_usage_ratios,
            "histogram_paths": [os.path.abspath(path) for path in histogram_paths],
        },
        "saved_reconstructions": {
            "enabled": bool(save_image_dirs),
            "gt_dir": None if gt_dir is None else os.path.abspath(gt_dir),
            "rec_dir": None if rec_dir is None else os.path.abspath(rec_dir),
            "comparison_dir": None if comparison_dir is None else os.path.abspath(comparison_dir),
            "comparison_grid_count": comparison_grid_count,
        },
        "fid_error": fid_error,
        "model_metadata": _collect_model_metadata(context),
        "saved_args": context.saved_args,
    }

    json_path = os.path.join(metrics_output_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    text_path = os.path.join(metrics_output_dir, "numerical_results.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(f"samples: {sample_count}\n")
        for key, value in results["metrics"].items():
            handle.write(f"{key}: {value}\n")
        handle.write(f"global_perplexity: {global_perplexity}\n")
        handle.write(f"global_usage_ratio: {global_usage_ratio}\n")
        handle.write(f"perplexity_per_component: {component_perplexities}\n")
        handle.write(f"usage_ratio_per_component: {component_usage_ratios}\n")
        if fid_error:
            handle.write(f"fid_error: {fid_error}\n")

    print(f"[evaluation] saved metrics to {json_path}")
    return results



def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction quality of a trained VQCVAE or MixedCurvatureVQCVAE checkpoint."
    )
    parser.add_argument("--model-dir", required=True, help="Training result directory containing args.json and checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or path relative to --model-dir. Defaults to best_val.pth.")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split used for evaluation.")
    parser.add_argument("--device", default="auto", help="Inference device, e.g. cpu, cuda, cuda:0. Defaults to auto.")
    parser.add_argument("--data-dir", default=None, help="Override dataset root directory stored in args.json.")
    parser.add_argument("--dataset-dir-name", default=None, help="Override dataset_dir_name stored in args.json.")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size used by the folder-dataset transform pipeline.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size used during evaluation.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of evaluated samples.")
    parser.add_argument("--output-dir", default=None, help="Directory used to store evaluation results. Defaults to <model-dir>/evaluation.")
    parser.add_argument("--save-image-dirs", action="store_true", help="Additionally save original/reconstructed image pairs during evaluation.")
    parser.add_argument("--reconstruction-output-dir", default=None, help="Where to store saved image pairs. Defaults to --model-dir.")
    parser.add_argument("--gt-dir-name", default="train", help="Subdirectory name for originals when --save-image-dirs is enabled.")
    parser.add_argument("--rec-dir-name", default="rec", help="Subdirectory name for reconstructions when --save-image-dirs is enabled.")
    parser.add_argument("--comparison-dir-name", default="comparison", help="Subdirectory name for side-by-side grids when grids are enabled.")
    parser.add_argument("--overwrite", action="store_true", help="Delete previous saved image subdirectories before writing new ones.")
    parser.add_argument("--save-comparison-grids", action="store_true", help="Save side-by-side comparison grids when saving images.")
    parser.add_argument("--compute-fid", action="store_true", help="Compute FID with pytorch-fid. Implies --save-image-dirs.")
    return parser



def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    results = run_evaluation(
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
        save_image_dirs=args.save_image_dirs,
        reconstruction_output_dir=args.reconstruction_output_dir,
        gt_dir_name=args.gt_dir_name,
        rec_dir_name=args.rec_dir_name,
        comparison_dir_name=args.comparison_dir_name,
        overwrite=args.overwrite,
        save_comparison_grids=args.save_comparison_grids,
        compute_fid=args.compute_fid,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# python evaluation.py --model-dir=/mnt/data/lkm/data/reMcVQVAE_results/ablation/test_geometric_ema/2026-04-05_20-08-44_cifar10_vqcvae_h128_k128_emb64_ema --device=cuda:3 --compute-fid --overwrite