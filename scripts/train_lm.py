# -*- coding: utf-8 -*-
"""Run mini GPT language-model training from the command line."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bpe import BPETokenizer
from dataset import create_dataloader
from model import GPTModel
from train import (
    calc_accuracy_loader,
    calc_loss_loader,
    plot_training_curves,
    save_checkpoint,
    train_model,
)
from scripts.console import configure_utf8_stdio


@dataclass(frozen=True)
class TrainingPreset:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    batch_size: int
    learning_rate: float
    epochs: int
    train_chars: int | None
    val_chars: int | None
    eval_freq: int
    eval_iter: int
    ckpt_freq: int


PRESETS: dict[str, TrainingPreset] = {
    "smoke": TrainingPreset(
        vocab_size=300,
        context_length=32,
        emb_dim=32,
        n_heads=4,
        n_layers=1,
        drop_rate=0.0,
        batch_size=2,
        learning_rate=3e-4,
        epochs=1,
        train_chars=5_000,
        val_chars=1_000,
        eval_freq=10,
        eval_iter=2,
        ckpt_freq=0,
    ),
    "dev": TrainingPreset(
        vocab_size=1_000,
        context_length=64,
        emb_dim=64,
        n_heads=4,
        n_layers=2,
        drop_rate=0.1,
        batch_size=8,
        learning_rate=3e-4,
        epochs=2,
        train_chars=200_000,
        val_chars=20_000,
        eval_freq=100,
        eval_iter=10,
        ckpt_freq=250,
    ),
    "full": TrainingPreset(
        vocab_size=3_000,
        context_length=128,
        emb_dim=128,
        n_heads=4,
        n_layers=4,
        drop_rate=0.1,
        batch_size=16,
        learning_rate=3e-4,
        epochs=5,
        train_chars=None,
        val_chars=None,
        eval_freq=200,
        eval_iter=20,
        ckpt_freq=500,
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def build_model_config(preset: TrainingPreset) -> dict:
    return {
        "vocab_size": preset.vocab_size,
        "context_length": preset.context_length,
        "emb_dim": preset.emb_dim,
        "n_heads": preset.n_heads,
        "n_layers": preset.n_layers,
        "drop_rate": preset.drop_rate,
        "qkv_bias": False,
    }


def limit_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    return text[:max_chars]


def read_lm_texts(
    data_dir: Path,
    train_chars: int | None,
    val_chars: int | None,
) -> tuple[str, str]:
    train_path = data_dir / "nsmc_lm_train.txt"
    val_path = data_dir / "nsmc_lm_val.txt"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing training text: {train_path}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing validation text: {val_path}")
    train_text = limit_text(train_path.read_text(encoding="utf-8"), train_chars)
    val_text = limit_text(val_path.read_text(encoding="utf-8"), val_chars)
    return train_text, val_text


def apply_overrides(args: argparse.Namespace) -> TrainingPreset:
    preset = PRESETS[args.preset]
    updates = {}
    for field_name in asdict(preset):
        value = getattr(args, field_name, None)
        if value is not None:
            updates[field_name] = value
    updated = replace(preset, **updates)
    if updated.emb_dim % updated.n_heads != 0:
        raise ValueError("--emb-dim must be divisible by --n-heads.")
    if updated.vocab_size < 260:
        raise ValueError("--vocab-size must be at least 260 for byte-level BPE.")
    return updated


def make_run_dir(output_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_final_evaluation(metrics: dict) -> str:
    return (
        "\n[Final evaluation]\n"
        f"run_dir: {metrics['run_dir']}\n"
        f"device: {metrics['device']}\n"
        f"epochs: {metrics['epochs']}\n"
        f"global_step: {metrics['global_step']}\n"
        f"train_loss ({metrics['train_eval_batches']} batches): "
        f"{metrics['train_loss']:.4f}\n"
        f"val_loss ({metrics['val_eval_batches']} batches): "
        f"{metrics['val_loss']:.4f}\n"
        f"train_token_accuracy ({metrics['train_eval_batches']} batches): "
        f"{metrics['train_accuracy']:.4f}\n"
        f"val_token_accuracy ({metrics['val_eval_batches']} batches): "
        f"{metrics['val_accuracy']:.4f}\n"
        f"elapsed_sec: {metrics['elapsed_sec']:.2f}\n"
        f"checkpoint: {Path(metrics['run_dir']) / 'final_checkpoint.pt'}\n"
        f"metrics: {Path(metrics['run_dir']) / 'metrics.json'}\n"
        f"plot: {Path(metrics['run_dir']) / 'training_curves.png'}"
    )


def run_training(args: argparse.Namespace) -> dict:
    preset = apply_overrides(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = make_run_dir(args.output_dir, args.run_name)
    progress_freq = args.progress_freq if args.progress_freq is not None else preset.eval_freq

    train_text, val_text = read_lm_texts(
        args.data_dir,
        train_chars=preset.train_chars,
        val_chars=preset.val_chars,
    )

    tokenizer = BPETokenizer(vocab_size=preset.vocab_size)
    tokenizer.train(train_text)
    tokenizer.save(run_dir / "tokenizer.json")

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    if len(train_ids) <= preset.context_length:
        raise ValueError("Training text is too short for the selected context length.")
    if len(val_ids) <= preset.context_length:
        raise ValueError("Validation text is too short for the selected context length.")

    train_loader = create_dataloader(
        train_ids,
        context_length=preset.context_length,
        batch_size=preset.batch_size,
        drop_last=True,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=preset.context_length,
        batch_size=preset.batch_size,
        drop_last=False,
        shuffle=False,
    )

    model_config = build_model_config(preset)
    model = GPTModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=preset.learning_rate)

    config_payload = {
        "preset": args.preset,
        "model": model_config,
        "training": asdict(preset),
        "progress_freq": progress_freq,
        "seed": args.seed,
    }
    write_json(run_dir / "config.json", config_payload)

    start = time.perf_counter()
    old_cwd = Path.cwd()
    eval_history = []
    try:
        os.chdir(run_dir)
        epoch_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=preset.epochs,
            eval_freq=preset.eval_freq,
            eval_iter=preset.eval_iter,
            start_context=args.start_context,
            tokenizer=tokenizer,
            ckpt_freq=preset.ckpt_freq if preset.ckpt_freq > 0 else None,
            progress_freq=progress_freq,
            eval_history=eval_history,
        )
    finally:
        os.chdir(old_cwd)
    elapsed_sec = time.perf_counter() - start

    global_step = len(train_loader) * preset.epochs
    save_checkpoint(
        model,
        optimizer,
        epoch=preset.epochs,
        global_step=global_step,
        path=str(run_dir / "final_checkpoint.pt"),
    )

    train_loss = calc_loss_loader(train_loader, model, device, num_batches=preset.eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=preset.eval_iter)
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=preset.eval_iter
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device, num_batches=preset.eval_iter
    )
    final_eval_point = {
        "step": global_step,
        "epoch": preset.epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    }
    if not eval_history or eval_history[-1]["step"] != global_step:
        eval_history.append(final_eval_point)
    curves_path = run_dir / "training_curves.png"
    plot_training_curves(eval_history, curves_path)

    metrics = {
        "preset": args.preset,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "device": str(device),
        "epochs": preset.epochs,
        "global_step": global_step,
        "elapsed_sec": elapsed_sec,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "train_eval_batches": min(preset.eval_iter, len(train_loader)),
        "val_eval_batches": min(preset.eval_iter, len(val_loader)),
        "progress_freq": progress_freq,
        "epoch_losses": epoch_losses,
        "eval_history": eval_history,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "training_curves_path": str(curves_path),
    }
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-context", default="이 영화")

    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--emb-dim", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--drop-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-chars", type=int)
    parser.add_argument("--val-chars", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--eval-iter", type=int)
    parser.add_argument("--ckpt-freq", type=int)
    parser.add_argument(
        "--progress-freq",
        type=int,
        help="Print elapsed time and ETA every N training steps. Defaults to eval_freq.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    metrics = run_training(args)
    print(format_final_evaluation(metrics))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
