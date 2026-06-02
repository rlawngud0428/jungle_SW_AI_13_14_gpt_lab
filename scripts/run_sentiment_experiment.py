# -*- coding: utf-8 -*-
"""Run a sentiment finetuning experiment and write presentation-ready outputs."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpe import BPETokenizer
from experiment_utils import ExperimentLogger, build_comparison_report
from finetune import (
    GPTForSequenceClassification,
    ReviewSentimentDataset,
    evaluate_sentiment,
    make_sentiment_dataset,
    train_epoch_sentiment,
)
from model import GPTModel
from train import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="sentiment_baseline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "sentiment")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-data", type=Path, default=ROOT / "data" / "ratings_train.txt")
    parser.add_argument("--test-data", type=Path, default=ROOT / "data" / "ratings_test.txt")
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=5000)

    parser.add_argument("--pretrain-run-dir", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--lm-train-text", type=Path, default=ROOT / "data" / "nsmc_lm_train.txt")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--bpe-chars", type=int, default=300_000)

    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--qkv-bias", action="store_true")

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument("--classifier-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--pooling",
        choices=["last_token", "last_non_pad", "mean"],
        default="last_non_pad",
    )
    parser.add_argument(
        "--freeze-mode",
        choices=["none", "backbone", "embedding", "all_but_last_block"],
        default="none",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def limit_rows(rows: list[dict], limit: int) -> list[dict]:
    return rows[:limit] if limit and limit > 0 else rows


def class_balance(rows: list[dict]) -> dict[str, float | int]:
    total = len(rows)
    counts = {0: 0, 1: 0}
    for row in rows:
        label = int(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return {
        "count": total,
        "negative": counts.get(0, 0),
        "positive": counts.get(1, 0),
        "positive_ratio": counts.get(1, 0) / total if total else 0.0,
    }


def load_or_train_tokenizer(args: argparse.Namespace, run_dir: Path) -> BPETokenizer:
    tokenizer_path = args.tokenizer_path
    if args.pretrain_run_dir is not None and tokenizer_path is None:
        tokenizer_path = args.pretrain_run_dir / "tokenizer.json"

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    if tokenizer_path is not None and tokenizer_path.exists():
        return tokenizer.load(tokenizer_path)

    corpus = args.lm_train_text.read_text(encoding="utf-8")
    tokenizer.train(corpus[: args.bpe_chars])
    tokenizer.save(run_dir / "tokenizer.json")
    return tokenizer


def load_model_config(args: argparse.Namespace, vocab_size: int) -> dict:
    if args.pretrain_run_dir is not None:
        model_config_path = args.pretrain_run_dir / "model_config.json"
        if model_config_path.exists():
            return json.loads(model_config_path.read_text(encoding="utf-8"))

    return {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
    }


def apply_freeze_mode(model: GPTForSequenceClassification, freeze_mode: str) -> None:
    if freeze_mode == "none":
        return

    if freeze_mode == "backbone":
        for parameter in model.gpt.parameters():
            parameter.requires_grad = False
        return

    if freeze_mode == "embedding":
        for parameter in model.gpt.embedding.parameters():
            parameter.requires_grad = False
        return

    if freeze_mode == "all_but_last_block":
        for parameter in model.gpt.embedding.parameters():
            parameter.requires_grad = False
        for block in model.gpt.blocks[:-1]:
            for parameter in block.parameters():
                parameter.requires_grad = False


def build_optimizer(args: argparse.Namespace, model: GPTForSequenceClassification) -> torch.optim.Optimizer:
    classifier_lr = args.classifier_learning_rate or args.learning_rate
    backbone_lr = args.backbone_learning_rate or args.learning_rate

    backbone_params = [
        parameter
        for parameter in model.gpt.parameters()
        if parameter.requires_grad
    ]
    classifier_params = [
        parameter
        for parameter in list(model.dropout.parameters()) + list(model.classifier.parameters())
        if parameter.requires_grad
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": classifier_lr})

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    config = {
        "stage": "sentiment_finetune",
        "device": str(device),
        "seed": args.seed,
        "train_data": str(args.train_data),
        "test_data": str(args.test_data),
        "val_ratio": args.val_ratio,
        "train_limit": args.train_limit,
        "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "pretrain_run_dir": str(args.pretrain_run_dir) if args.pretrain_run_dir else None,
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "vocab_size": args.vocab_size,
        "bpe_chars": args.bpe_chars,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "drop_rate": args.drop_rate,
        "qkv_bias": args.qkv_bias,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate or args.learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate or args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "freeze_mode": "backbone" if args.freeze_backbone else args.freeze_mode,
        "pooling": args.pooling,
    }
    logger = ExperimentLogger(args.output_dir, args.experiment_name, config)

    tokenizer = load_or_train_tokenizer(args, logger.run_dir)
    model_config = load_model_config(args, vocab_size=len(tokenizer.id_to_token))
    config.update({f"model_{key}": value for key, value in model_config.items()})
    (logger.run_dir / "model_config.json").write_text(
        json.dumps(model_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_rows, val_rows, test_rows = make_sentiment_dataset(
        args.train_data,
        args.test_data,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_rows = limit_rows(train_rows, args.train_limit)
    val_rows = limit_rows(val_rows, args.val_limit)
    test_rows = limit_rows(test_rows, args.test_limit)
    config.update(
        {
            "train_balance": class_balance(train_rows),
            "val_balance": class_balance(val_rows),
            "test_balance": class_balance(test_rows),
        }
    )
    (logger.run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_loader = DataLoader(
        ReviewSentimentDataset(train_rows, tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ReviewSentimentDataset(val_rows, tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        ReviewSentimentDataset(test_rows, tokenizer, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
    )

    backbone = GPTModel(model_config)
    checkpoint_path = args.checkpoint_path
    if args.pretrain_run_dir is not None and checkpoint_path is None:
        checkpoint_path = args.pretrain_run_dir / "pretrained_gpt.pt"
    if checkpoint_path is not None and checkpoint_path.exists():
        load_checkpoint(backbone, optimizer=None, path=str(checkpoint_path), device=device)

    model = GPTForSequenceClassification(
        backbone,
        num_labels=2,
        drop_rate=args.drop_rate,
        pooling=args.pooling,
        pad_id=tokenizer.get_pad_id(),
    )
    freeze_mode = "backbone" if args.freeze_backbone else args.freeze_mode
    apply_freeze_mode(model, freeze_mode)

    optimizer = build_optimizer(args, model)
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = logger.run_dir / "best_sentiment.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_epoch_sentiment(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_sentiment(model, val_loader, device)
        elapsed = time.perf_counter() - epoch_start
        logger.log_metric(
            stage="finetune",
            epoch=epoch,
            metrics={
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            elapsed_seconds=elapsed,
        )
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, elapsed={elapsed:.1f}s"
        )

        is_better = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "config": config,
                },
                best_checkpoint_path,
            )

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc = evaluate_sentiment(model, test_loader, device)
    logger.finish(
        final_metrics={
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
        }
    )
    build_comparison_report(args.output_dir)
    print(f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
    print(f"run_dir={logger.run_dir}")


if __name__ == "__main__":
    main()
