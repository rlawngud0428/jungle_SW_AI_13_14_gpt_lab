# -*- coding: utf-8 -*-
"""Benchmark mini GPT forward-pass throughput and validation loss."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

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
from train import calc_loss_batch, load_checkpoint

from scripts.train_lm import (
    PRESETS,
    apply_overrides,
    build_model_config,
    read_lm_texts,
    resolve_device,
    set_seed,
    write_json,
)
from scripts.console import configure_utf8_stdio


def _load_run(run_dir: Path, checkpoint: Path | None, device: torch.device):
    config_path = run_dir / "config.json"
    tokenizer_path = run_dir / "tokenizer.json"
    if checkpoint is None:
        checkpoint = run_dir / "final_checkpoint.pt"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = BPETokenizer(vocab_size=config_payload["model"]["vocab_size"])
    tokenizer.load(tokenizer_path)
    model = GPTModel(config_payload["model"])
    load_checkpoint(model, optimizer=None, path=str(checkpoint), device=device)
    return model, tokenizer, config_payload


def _build_fresh(args: argparse.Namespace, device: torch.device):
    preset = apply_overrides(args)
    train_text, _ = read_lm_texts(
        args.data_dir,
        train_chars=preset.train_chars,
        val_chars=preset.val_chars,
    )
    tokenizer = BPETokenizer(vocab_size=preset.vocab_size)
    tokenizer.train(train_text)
    model_config = build_model_config(preset)
    model = GPTModel(model_config)
    config_payload = {
        "preset": args.preset,
        "model": model_config,
        "training": {
            "batch_size": preset.batch_size,
            "context_length": preset.context_length,
            "train_chars": preset.train_chars,
            "val_chars": preset.val_chars,
        },
    }
    return model, tokenizer, config_payload


def benchmark_model(
    model: GPTModel,
    loader,
    device: torch.device,
    num_batches: int,
    warmup_batches: int,
) -> dict:
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_index, (input_batch, target_batch) in enumerate(loader):
            if batch_index >= warmup_batches:
                break
            _ = calc_loss_batch(input_batch, target_batch, model, device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        total_loss = 0.0
        total_tokens = 0
        total_batches = 0
        start = time.perf_counter()
        for batch_index, (input_batch, target_batch) in enumerate(loader):
            if batch_index >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_loss += loss.item()
            total_tokens += input_batch.numel()
            total_batches += 1
        elapsed = time.perf_counter() - start

    if total_batches == 0:
        raise ValueError("No benchmark batches were available.")

    return {
        "device": str(device),
        "batches": total_batches,
        "tokens": total_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / elapsed if elapsed > 0 else float("inf"),
        "loss": total_loss / total_batches,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=2)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.run_dir is not None:
        model, tokenizer, config_payload = _load_run(args.run_dir, args.checkpoint, device)
        training = config_payload["training"]
        batch_size = args.batch_size or training["batch_size"]
        context_length = args.context_length or config_payload["model"]["context_length"]
        _, val_text = read_lm_texts(
            args.data_dir,
            train_chars=training.get("train_chars"),
            val_chars=training.get("val_chars"),
        )
    else:
        model, tokenizer, config_payload = _build_fresh(args, device)
        training = config_payload["training"]
        batch_size = training["batch_size"]
        context_length = training["context_length"]
        _, val_text = read_lm_texts(
            args.data_dir,
            train_chars=training.get("train_chars"),
            val_chars=training.get("val_chars"),
        )

    val_ids = tokenizer.encode(val_text)
    if len(val_ids) <= context_length:
        raise ValueError("Validation text is too short for the selected context length.")

    loader = create_dataloader(
        val_ids,
        context_length=context_length,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )
    metrics = benchmark_model(
        model,
        loader,
        device=device,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
    )
    metrics["context_length"] = context_length
    metrics["batch_size"] = batch_size
    metrics["preset"] = config_payload.get("preset", args.preset)
    if args.run_dir is not None:
        metrics["run_dir"] = str(args.run_dir)

    output_path = args.output_json
    if output_path is None and args.run_dir is not None:
        output_path = args.run_dir / "benchmark.json"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(output_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
