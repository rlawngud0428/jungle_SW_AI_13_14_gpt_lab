# -*- coding: utf-8 -*-
"""Utilities for recording experiment metrics and presentation-ready reports."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _json_default(value: Any) -> str:
    return str(value)


def _safe_name(name: str) -> str:
    allowed = []
    for char in name.strip().lower().replace(" ", "_"):
        allowed.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(allowed).strip("_") or "experiment"


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


class ExperimentLogger:
    """Write config, per-epoch metrics, plots, and markdown summaries for one run."""

    def __init__(
        self,
        output_dir: str | Path,
        experiment_name: str,
        config: dict[str, Any],
    ):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.config = config
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.output_dir / f"{timestamp}_{_safe_name(experiment_name)}"
        self.plots_dir = self.run_dir / "plots"
        self.metrics: list[dict[str, Any]] = []
        self.started_at = time.time()

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(self.run_dir / "config.json", self.config)

    def log_metric(
        self,
        stage: str,
        epoch: int,
        metrics: dict[str, float],
        elapsed_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Append one metric row and persist it to metrics.jsonl."""
        row = {
            "stage": stage,
            "epoch": epoch,
            "elapsed_seconds": elapsed_seconds,
            **metrics,
        }
        self.metrics.append(row)
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        return row

    def finish(self, final_metrics: dict[str, float] | None = None) -> dict[str, Any]:
        """Write summary files and plots for the completed run."""
        final_metrics = final_metrics or {}
        total_seconds = time.time() - self.started_at
        summary = {
            "experiment_name": self.experiment_name,
            "run_dir": str(self.run_dir),
            "config": self.config,
            "final_metrics": final_metrics,
            "best_metrics": self._best_metrics(),
            "total_seconds": total_seconds,
            "total_time": _format_seconds(total_seconds),
        }

        self._write_json(self.run_dir / "summary.json", summary)
        self._write_metrics_csv()
        self._write_markdown(summary)
        self._plot_stage("pretrain")
        self._plot_stage("finetune")
        return summary

    def _best_metrics(self) -> dict[str, float]:
        best: dict[str, float] = {}
        for key in ("val_acc", "test_acc"):
            values = [row[key] for row in self.metrics if key in row and row[key] is not None]
            if values:
                best[f"best_{key}"] = max(values)
        for key in ("val_loss", "train_loss"):
            values = [row[key] for row in self.metrics if key in row and row[key] is not None]
            if values:
                best[f"best_{key}"] = min(values)
        return best

    def _write_metrics_csv(self) -> None:
        if not self.metrics:
            return
        fieldnames: list[str] = []
        for row in self.metrics:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (self.run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics)

    def _write_markdown(self, summary: dict[str, Any]) -> None:
        lines = [
            f"# Experiment: {self.experiment_name}",
            "",
            "## Hyperparameters",
            "",
            "| Key | Value |",
            "| --- | --- |",
        ]
        for key, value in sorted(self.config.items()):
            lines.append(f"| {key} | `{value}` |")

        lines.extend(
            [
                "",
                "## Final Metrics",
                "",
                "| Metric | Value |",
                "| --- | --- |",
            ]
        )
        for key, value in summary["final_metrics"].items():
            lines.append(f"| {key} | {value:.4f} |")
        for key, value in summary["best_metrics"].items():
            lines.append(f"| {key} | {value:.4f} |")
        lines.append(f"| total_time | {summary['total_time']} |")

        if self.metrics:
            lines.extend(
                [
                    "",
                    "## Epoch Metrics",
                    "",
                    "| Stage | Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Elapsed |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for row in self.metrics:
                lines.append(
                    "| {stage} | {epoch} | {train_loss} | {train_acc} | {val_loss} | {val_acc} | {elapsed} |".format(
                        stage=row.get("stage", ""),
                        epoch=row.get("epoch", ""),
                        train_loss=_metric_text(row.get("train_loss")),
                        train_acc=_metric_text(row.get("train_acc")),
                        val_loss=_metric_text(row.get("val_loss")),
                        val_acc=_metric_text(row.get("val_acc")),
                        elapsed=_format_seconds(row["elapsed_seconds"])
                        if row.get("elapsed_seconds") is not None
                        else "",
                    )
                )

        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _plot_stage(self, stage: str) -> None:
        rows = [row for row in self.metrics if row.get("stage") == stage]
        if not rows:
            return

        epochs = [row["epoch"] for row in rows]
        loss_series = {
            "train_loss": [row.get("train_loss") for row in rows],
            "val_loss": [row.get("val_loss") for row in rows],
            "test_loss": [row.get("test_loss") for row in rows],
        }
        acc_series = {
            "train_acc": [row.get("train_acc") for row in rows],
            "val_acc": [row.get("val_acc") for row in rows],
            "test_acc": [row.get("test_acc") for row in rows],
        }
        self._plot_series(epochs, loss_series, self.plots_dir / f"{stage}_loss.png", "Loss")
        self._plot_series(epochs, acc_series, self.plots_dir / f"{stage}_accuracy.png", "Accuracy")

    def _plot_series(
        self,
        epochs: list[int],
        series: dict[str, list[float | None]],
        path: Path,
        ylabel: str,
    ) -> None:
        plotted = False
        plt.figure(figsize=(7, 4))
        for name, values in series.items():
            points = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
            if not points:
                continue
            xs, ys = zip(*points)
            plt.plot(xs, ys, marker="o", label=name)
            plotted = True
        if not plotted:
            plt.close()
            return
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{self.experiment_name} {ylabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _metric_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float):
        return f"{value:.4f}"
    return str(value)


def build_comparison_report(output_dir: str | Path) -> list[dict[str, Any]]:
    """Collect run summaries under output_dir and write comparison files."""
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(output_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary.get("config", {})
        final_metrics = summary.get("final_metrics", {})
        best_metrics = summary.get("best_metrics", {})
        row = {
            "experiment_name": summary.get("experiment_name"),
            "run_dir": summary.get("run_dir"),
            "total_seconds": summary.get("total_seconds"),
            "total_time": summary.get("total_time"),
            **{f"config_{key}": value for key, value in config.items()},
            **final_metrics,
            **best_metrics,
        }
        row.setdefault("best_val_acc", final_metrics.get("val_acc"))
        row.setdefault("best_val_loss", final_metrics.get("val_loss"))
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row.get("best_val_acc") is not None,
            row.get("best_val_acc") or float("-inf"),
        ),
        reverse=True,
    )
    _write_comparison_csv(output_dir / "comparison.csv", rows)
    _write_comparison_markdown(output_dir / "comparison.md", rows)
    _plot_comparison(output_dir / "comparison_val_acc.png", rows)
    return rows


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Comparison",
        "",
        "| Rank | Experiment | Best Val Acc | Best Val Loss | Test Acc | Time |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {name} | {val_acc} | {val_loss} | {test_acc} | {time} |".format(
                rank=rank,
                name=row.get("experiment_name", ""),
                val_acc=_metric_text(row.get("best_val_acc")),
                val_loss=_metric_text(row.get("best_val_loss")),
                test_acc=_metric_text(row.get("test_acc")),
                time=row.get("total_time", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    points = [
        (row.get("experiment_name", ""), row.get("best_val_acc"))
        for row in rows
        if row.get("best_val_acc") is not None
    ]
    if not points:
        return
    names, values = zip(*points)
    plt.figure(figsize=(max(7, len(names) * 1.4), 4))
    plt.bar(names, values)
    plt.ylabel("Best Validation Accuracy")
    plt.title("Experiment Comparison")
    plt.ylim(0, max(1.0, max(values) * 1.1))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
