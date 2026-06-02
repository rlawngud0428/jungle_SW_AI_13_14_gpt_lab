# -*- coding: utf-8 -*-
"""Run grouped sentiment finetuning experiments and build comparison reports."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = ROOT / "scripts" / "run_sentiment_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "sentiment_sweep")
    parser.add_argument("--pretrain-run-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["focused", "full"], default="focused")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=5000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")

    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--bpe-chars", type=int, default=300_000)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def build_sweep_specs(args: argparse.Namespace) -> list[dict]:
    base = {
        "vocab_size": args.vocab_size,
        "bpe_chars": args.bpe_chars,
        "context_length": args.context_length,
        "emb_dim": args.emb_dim,
        "n_layers": args.n_layers,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "drop_rate": args.drop_rate,
        "max_length": args.max_length,
        "pooling": "last_non_pad",
        "freeze_mode": "none",
        "backbone_learning_rate": args.learning_rate,
        "classifier_learning_rate": args.learning_rate,
    }

    specs = [
        {
            "category": "baseline",
            "name": "01_baseline",
            "description": "권장 범위 중심 기준 실험",
            "params": base,
        }
    ]

    focused_variants = [
        ("learning_rate", "02_lr_1e-4", "learning rate 낮춤", {"learning_rate": 1e-4}),
        ("learning_rate", "03_lr_5e-4", "learning rate 높임", {"learning_rate": 5e-4}),
        ("dropout", "04_dropout_0", "dropout 제거", {"drop_rate": 0.0}),
        ("dropout", "05_dropout_0_2", "dropout 강화", {"drop_rate": 0.2}),
        ("sequence_length", "06_maxlen_64", "짧은 리뷰 길이 제한", {"max_length": 64, "context_length": 64}),
        ("pooling", "07_pooling_mean", "평균 pooling", {"pooling": "mean"}),
        ("pooling", "08_pooling_last_token", "마지막 위치 pooling", {"pooling": "last_token"}),
        ("freeze", "09_freeze_backbone", "GPT backbone 고정", {"freeze_mode": "backbone"}),
        ("freeze", "10_freeze_embedding", "embedding만 고정", {"freeze_mode": "embedding"}),
        (
            "differential_lr",
            "11_diff_lr",
            "backbone은 낮은 LR, classifier는 높은 LR",
            {
                "backbone_learning_rate": 1e-4,
                "classifier_learning_rate": 5e-4,
                "learning_rate": 3e-4,
            },
        ),
    ]

    full_extra = [
        ("batch_size", "12_batch_8", "batch size 8", {"batch_size": 8}),
        ("batch_size", "13_batch_4", "batch size 4", {"batch_size": 4}),
        ("architecture", "14_layers_1", "1 layer 모델", {"n_layers": 1}),
        ("architecture", "15_layers_4", "4 layer 모델", {"n_layers": 4}),
        ("architecture", "16_emb_64", "embedding 64", {"emb_dim": 64}),
        ("architecture", "17_emb_192", "embedding 192", {"emb_dim": 192}),
        ("sequence_length", "18_context_64", "context length 64", {"context_length": 64, "max_length": 64}),
    ]

    variants = focused_variants + (full_extra if args.mode == "full" else [])
    for category, name, description, changes in variants:
        params = {**base, **changes}
        if "learning_rate" in changes and "backbone_learning_rate" not in changes:
            params["backbone_learning_rate"] = params["learning_rate"]
            params["classifier_learning_rate"] = params["learning_rate"]
        specs.append(
            {
                "category": category,
                "name": name,
                "description": description,
                "params": params,
            }
        )
    return specs


def write_plan(output_dir: Path, specs: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sentiment Experiment Sweep Plan",
        "",
        "| Order | Category | Experiment | Purpose | Key Params |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, spec in enumerate(specs, start=1):
        params = spec["params"]
        key_params = ", ".join(
            [
                f"lr={params['learning_rate']}",
                f"drop={params['drop_rate']}",
                f"batch={params['batch_size']}",
                f"ctx={params['context_length']}",
                f"emb={params['emb_dim']}",
                f"layers={params['n_layers']}",
                f"pool={params['pooling']}",
                f"freeze={params['freeze_mode']}",
            ]
        )
        lines.append(
            f"| {index} | {spec['category']} | `{spec['name']}` | {spec['description']} | {key_params} |"
        )
    (output_dir / "sweep_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_for_spec(args: argparse.Namespace, spec: dict) -> list[str]:
    params = spec["params"]
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--experiment-name",
        spec["name"],
        "--output-dir",
        str(args.output_dir),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--train-limit",
        str(args.train_limit),
        "--val-limit",
        str(args.val_limit),
        "--test-limit",
        str(args.test_limit),
        "--vocab-size",
        str(params["vocab_size"]),
        "--bpe-chars",
        str(params["bpe_chars"]),
        "--context-length",
        str(params["context_length"]),
        "--emb-dim",
        str(params["emb_dim"]),
        "--n-layers",
        str(params["n_layers"]),
        "--batch-size",
        str(params["batch_size"]),
        "--learning-rate",
        str(params["learning_rate"]),
        "--backbone-learning-rate",
        str(params["backbone_learning_rate"]),
        "--classifier-learning-rate",
        str(params["classifier_learning_rate"]),
        "--drop-rate",
        str(params["drop_rate"]),
        "--max-length",
        str(params["max_length"]),
        "--pooling",
        params["pooling"],
        "--freeze-mode",
        params["freeze_mode"],
    ]
    if args.pretrain_run_dir is not None:
        command.extend(["--pretrain-run-dir", str(args.pretrain_run_dir)])
    return command


def main() -> None:
    args = parse_args()
    specs = build_sweep_specs(args)
    write_plan(args.output_dir, specs)

    for index, spec in enumerate(specs, start=1):
        command = command_for_spec(args, spec)
        print(f"[{index}/{len(specs)}] {spec['name']} - {spec['description']}")
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=ROOT, check=True)

    if not args.dry_run:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "compare_experiments.py"), str(args.output_dir)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
