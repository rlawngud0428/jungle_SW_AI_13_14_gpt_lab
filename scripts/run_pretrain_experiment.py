# -*- coding: utf-8 -*-
"""Run a GPT pretraining experiment and write presentation-ready outputs."""

import argparse
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpe import BPETokenizer
from dataset import create_dataloader
from experiment_utils import ExperimentLogger, build_comparison_report
from model import GPTModel
from train import calc_loss_batch, calc_loss_loader, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="pretrain_baseline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "pretrain")
    parser.add_argument("--train-text", type=Path, default=ROOT / "data" / "nsmc_lm_train.txt")
    parser.add_argument("--val-text", type=Path, default=ROOT / "data" / "nsmc_lm_val.txt")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--bpe-chars", type=int, default=300_000)
    parser.add_argument("--train-chars", type=int, default=300_000)
    parser.add_argument("--val-chars", type=int, default=60_000)

    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-iter", type=int, default=20)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    train_text = args.train_text.read_text(encoding="utf-8")
    val_text = args.val_text.read_text(encoding="utf-8")

    config = {
        "stage": "pretrain",
        "device": str(device),
        "seed": args.seed,
        "vocab_size": args.vocab_size,
        "bpe_chars": args.bpe_chars,
        "train_chars": args.train_chars,
        "val_chars": args.val_chars,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
        "batch_size": args.batch_size,
        "stride": args.stride or args.context_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "eval_iter": args.eval_iter,
        "train_text": str(args.train_text),
        "val_text": str(args.val_text),
    }
    logger = ExperimentLogger(args.output_dir, args.experiment_name, config)

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    bpe_start = time.perf_counter()
    tokenizer.train(train_text[: args.bpe_chars])
    bpe_seconds = time.perf_counter() - bpe_start
    tokenizer_path = logger.run_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    model_config = {
        "vocab_size": len(tokenizer.id_to_token),
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }
    (logger.run_dir / "model_config.json").write_text(
        __import__("json").dumps(model_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_ids = tokenizer.encode(train_text[: args.train_chars])
    val_ids = tokenizer.encode(val_text[: args.val_chars])
    train_loader = create_dataloader(
        train_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.stride,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ids,
        context_length=args.context_length,
        batch_size=args.batch_size,
        stride=args.stride,
        shuffle=False,
    )

    model = GPTModel(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1

        train_loss = total_loss / batches if batches else float("nan")
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=args.eval_iter)
        elapsed = time.perf_counter() - epoch_start
        logger.log_metric(
            stage="pretrain",
            epoch=epoch,
            metrics={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bpe_seconds": bpe_seconds if epoch == 1 else 0.0,
            },
            elapsed_seconds=elapsed,
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, elapsed={elapsed:.1f}s"
        )

    checkpoint_path = logger.run_dir / "pretrained_gpt.pt"
    save_checkpoint(model, optimizer, epoch=args.epochs, global_step=0, path=str(checkpoint_path))
    logger.finish(
        final_metrics={
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "bpe_seconds": bpe_seconds,
        }
    )
    build_comparison_report(args.output_dir)
    print(f"run_dir={logger.run_dir}")


if __name__ == "__main__":
    main()
