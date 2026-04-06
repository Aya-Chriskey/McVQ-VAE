import json
import logging
import os
import shutil
from datetime import datetime

import torch


IGNORED_ARG_FIELDS = {
    'curvature', 'c', 'spaces', 'fix_alpha', 'fix_curvature', 'mixed', 'kl_coef', 'vq_coef', 'pre', 'data_format'
}


def _safe_getattr(args, name, default=None):
    return getattr(args, name, default)



def _build_save_name(args) -> str:
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset = _safe_getattr(args, 'dataset', 'dataset')
    model = _safe_getattr(args, 'model', 'model')
    hidden = _safe_getattr(args, 'hidden', 'hidden')
    k = _safe_getattr(args, 'k', 'k')
    emb = _safe_getattr(args, 'embedding_dim', 'emb')
    ema = 'ema' if _safe_getattr(args, 'ema', False) else 'noema'
    return f'{timestamp}_{dataset}_{model}_h{hidden}_k{k}_emb{emb}_{ema}'



def setup_logging_from_args(args):
    resume = _safe_getattr(args, 'resume', False)
    save_name = _safe_getattr(args, 'save_name', '')
    results_dir = _safe_getattr(args, 'results_dir', './results')

    if not save_name:
        save_name = _build_save_name(args)

    save_path = os.path.join(results_dir, save_name)
    if os.path.exists(save_path) and not resume:
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    log_file = os.path.join(save_path, 'log.txt')
    setup_logging(log_file, resume)
    export_args(args, save_path)
    return save_path



def setup_logging(log_file='log.txt', resume=False):
    file_mode = 'a' if os.path.isfile(log_file) and resume else 'w'

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode=file_mode,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger('').addHandler(console)



def export_args(args, save_path):
    os.makedirs(save_path, exist_ok=True)
    json_file_name = os.path.join(save_path, 'args.json')
    serializable_args = {}
    for key, value in vars(args).items():
        if key in IGNORED_ARG_FIELDS:
            serializable_args[key] = value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serializable_args[key] = value
        else:
            serializable_args[key] = str(value)
    with open(json_file_name, 'w') as fp:
        json.dump(serializable_args, fp, sort_keys=True, indent=4)



def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(path, f"checkpoint_epoch_{state['epoch']}.pth.tar"))
