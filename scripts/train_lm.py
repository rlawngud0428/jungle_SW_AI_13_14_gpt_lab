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
import matplotlib.pyplot as plt
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
    calc_loss_batch,
    calc_loss_loader,
    generate_and_print_sample,
    save_checkpoint,
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


def format_duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "--:--:--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calc_accuracy_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    if len(data_loader) == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    batches_seen = 0
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            if batches_seen >= max_batches:
                break
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == target_batch).sum().item()
            total += target_batch.numel()
            batches_seen += 1

    if was_training:
        model.train()

    return correct / total if total > 0 else float("nan")


def plot_training_curves(eval_history: list[dict], path: Path) -> None:
    if not eval_history:
        return

    steps = [point["step"] for point in eval_history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(steps, [point["train_loss"] for point in eval_history], label="Train")
    axes[0].plot(steps, [point["val_loss"] for point in eval_history], label="Val")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(
        steps,
        [point["train_accuracy"] for point in eval_history],
        label="Train",
    )
    axes[1].plot(
        steps,
        [point["val_accuracy"] for point in eval_history],
        label="Val",
    )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Next-token Accuracy")
    axes[1].legend()

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def train_language_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    progress_freq: int | None = None,
    eval_history: list[dict] | None = None,
) -> list[float]:
    epoch_losses = []
    global_step = 0
    total_steps = len(train_loader) * num_epochs
    start_time = time.perf_counter()
    model.to(device)

    if eval_history is None:
        eval_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batches_seen = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches_seen += 1
            global_step += 1

            if progress_freq is not None and progress_freq > 0:
                if global_step % progress_freq == 0 or global_step == total_steps:
                    elapsed = time.perf_counter() - start_time
                    steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
                    remaining_steps = max(0, total_steps - global_step)
                    eta = (
                        remaining_steps / steps_per_sec
                        if steps_per_sec > 0
                        else float("inf")
                    )
                    percent = (global_step / total_steps * 100) if total_steps else 100.0
                    print(
                        f"[Progress] step {global_step}/{total_steps} "
                        f"({percent:.1f}%) | epoch {epoch + 1}/{num_epochs} | "
                        f"elapsed {format_duration(elapsed)} | "
                        f"eta {format_duration(eta)} | "
                        f"{steps_per_sec:.2f} steps/s"
                    )

            if eval_freq > 0 and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                train_accuracy = calc_accuracy_loader(train_loader, model, device, eval_iter)
                val_accuracy = calc_accuracy_loader(val_loader, model, device, eval_iter)
                eval_history.append(
                    {
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_accuracy,
                        "val_accuracy": val_accuracy,
                    }
                )
                print(
                    f"Ep {epoch + 1} (step {global_step}): "
                    f"train loss {train_loss:.3f}, val loss {val_loss:.3f}, "
                    f"train acc {train_accuracy:.3f}, val acc {val_accuracy:.3f}"
                )

            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt",
                )

        avg_epoch_loss = epoch_loss / batches_seen if batches_seen > 0 else float("nan")
        epoch_losses.append(avg_epoch_loss)
        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context=start_context,
            context_size=model.config.get("context_length", 256),
        )

    return epoch_losses


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
        epoch_losses = train_language_model(
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
