# -*- coding: utf-8 -*-
"""Experiment logging/reporting utility tests."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_experiment_logger_writes_presentation_outputs(tmp_path):
    from experiment_utils import ExperimentLogger

    logger = ExperimentLogger(
        output_dir=tmp_path,
        experiment_name="baseline",
        config={
            "context_length": 64,
            "emb_dim": 64,
            "learning_rate": 3e-4,
            "batch_size": 16,
        },
    )

    logger.log_metric(
        stage="finetune",
        epoch=1,
        metrics={
            "train_loss": 0.7,
            "train_acc": 0.55,
            "val_loss": 0.68,
            "val_acc": 0.58,
        },
        elapsed_seconds=12.3,
    )
    logger.log_metric(
        stage="finetune",
        epoch=2,
        metrics={
            "train_loss": 0.62,
            "train_acc": 0.64,
            "val_loss": 0.61,
            "val_acc": 0.66,
        },
        elapsed_seconds=25.4,
    )
    logger.finish(final_metrics={"test_loss": 0.6, "test_acc": 0.67})

    run_dir = logger.run_dir
    assert (run_dir / "config.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "plots" / "finetune_loss.png").exists()
    assert (run_dir / "plots" / "finetune_accuracy.png").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "baseline"
    assert summary["config"]["context_length"] == 64
    assert summary["final_metrics"]["test_acc"] == 0.67

    report = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "context_length" in report
    assert "val_acc" in report
    assert "test_acc" in report


def test_build_comparison_report_collects_runs(tmp_path):
    from experiment_utils import ExperimentLogger, build_comparison_report

    for name, val_acc in (("baseline", 0.6), ("lr_1e-4", 0.65)):
        logger = ExperimentLogger(
            output_dir=tmp_path,
            experiment_name=name,
            config={"learning_rate": 1e-4 if name == "lr_1e-4" else 3e-4},
        )
        logger.log_metric(
            stage="finetune",
            epoch=1,
            metrics={"train_loss": 0.7, "val_loss": 0.65, "val_acc": val_acc},
            elapsed_seconds=3.0,
        )
        logger.finish(final_metrics={"test_acc": val_acc - 0.01})

    comparison = build_comparison_report(tmp_path)

    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "comparison_val_acc.png").exists()
    assert len(comparison) == 2
    assert comparison[0]["best_val_acc"] >= comparison[1]["best_val_acc"]


def test_sentiment_sweep_specs_group_key_experiments(tmp_path):
    from argparse import Namespace
    from run_sentiment_sweep import build_sweep_specs, write_plan

    args = Namespace(
        mode="focused",
        vocab_size=1000,
        bpe_chars=300_000,
        context_length=128,
        emb_dim=128,
        n_layers=2,
        batch_size=16,
        learning_rate=3e-4,
        drop_rate=0.1,
        max_length=128,
    )

    specs = build_sweep_specs(args)
    names = {spec["name"] for spec in specs}
    categories = {spec["category"] for spec in specs}

    assert "01_baseline" in names
    assert "07_pooling_mean" in names
    assert "09_freeze_backbone" in names
    assert "11_diff_lr" in names
    assert {"learning_rate", "dropout", "pooling", "freeze", "differential_lr"} <= categories

    write_plan(tmp_path, specs)
    plan = (tmp_path / "sweep_plan.md").read_text(encoding="utf-8")
    assert "Sentiment Experiment Sweep Plan" in plan
    assert "01_baseline" in plan
