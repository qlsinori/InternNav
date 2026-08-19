#!/usr/bin/env python3
"""Export completed System2 Trainer and GPU telemetry logs to TensorBoard."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smooth-window", type=int, default=20)
    return parser.parse_args()


def export_trainer_state(writer: SummaryWriter, state_path: Path, smooth_window: int) -> dict[str, float]:
    state = json.loads(state_path.read_text())
    history = sorted(
        (entry for entry in state["log_history"] if "loss" in entry and "step" in entry),
        key=lambda entry: entry["step"],
    )
    if not history:
        raise ValueError(f"No per-step loss entries in {state_path}")

    loss_window: deque[float] = deque(maxlen=smooth_window)
    original_losses: list[float] = []
    resumed_losses: list[float] = []
    all_losses: list[float] = []

    for entry in history:
        step = int(entry["step"])
        loss = float(entry["loss"])
        all_losses.append(loss)
        loss_window.append(loss)
        (original_losses if step <= 182 else resumed_losses).append(loss)

        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/loss_sma", sum(loss_window) / len(loss_window), step)
        if "learning_rate" in entry:
            writer.add_scalar("train/learning_rate", float(entry["learning_rate"]), step)
        if "grad_norm" in entry:
            writer.add_scalar("train/grad_norm_pre_clip", float(entry["grad_norm"]), step)
        if "epoch" in entry:
            writer.add_scalar("train/epoch", float(entry["epoch"]), step)

    final_step = int(history[-1]["step"])
    summary = {
        "final_step": final_step,
        "full_mean_loss": sum(all_losses) / len(all_losses),
        "original_mean_loss": sum(original_losses) / len(original_losses),
        "resume_mean_loss": sum(resumed_losses) / len(resumed_losses),
        "last_20_mean_loss": sum(all_losses[-20:]) / min(20, len(all_losses)),
        "final_epoch": float(state["epoch"]),
    }
    summary["epoch_coverage_pct"] = 100.0 * summary["final_epoch"]
    summary["full_train_perplexity"] = math.exp(summary["full_mean_loss"])
    summary["last_20_perplexity"] = math.exp(summary["last_20_mean_loss"])

    for name, value in summary.items():
        if name != "final_step":
            writer.add_scalar(f"summary/{name}", value, final_step)
    return summary


def export_gpu_telemetry(writer: SummaryWriter, telemetry_paths: list[Path]) -> int:
    sample_steps: defaultdict[int, int] = defaultdict(int)
    rows_written = 0

    for telemetry_path in telemetry_paths:
        if not telemetry_path.exists():
            continue
        with telemetry_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                gpu = int(row["index"].strip())
                step = sample_steps[gpu]
                sample_steps[gpu] += 1
                walltime = datetime.fromisoformat(row["timestamp_utc"].strip()).timestamp()
                prefix = f"gpu/gpu{gpu}"
                values = {
                    "memory_used_mib": float(row["memory_used_mib"].strip()),
                    "memory_free_mib": float(row["memory_free_mib"].strip()),
                    "utilization_pct": float(row["utilization_gpu_pct"].strip()),
                    "power_w": float(row["power_w"].strip()),
                    "temperature_c": float(row["temperature_c"].strip()),
                }
                for name, value in values.items():
                    writer.add_scalar(f"{prefix}/{name}", value, step, walltime=walltime)
                rows_written += 1
    return rows_written


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "trainer_state.json"
    final_step_hint = int(json.loads(state_path.read_text())["global_step"])
    output_dir = (
        args.output_dir or run_dir / "tensorboard" / f"history_1_to_{final_step_hint}"
    ).resolve()
    if any(output_dir.glob("events.out.tfevents.*")):
        raise FileExistsError(f"TensorBoard events already exist in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_dir), filename_suffix=".system2")
    try:
        summary = export_trainer_state(writer, state_path, args.smooth_window)
        telemetry_rows = export_gpu_telemetry(
            writer,
            [
                run_dir / "logs" / "gpu_telemetry.csv",
                run_dir / "logs" / "gpu_telemetry_resume_182_to_728.csv",
            ],
        )
        summary["gpu_telemetry_rows"] = telemetry_rows
        writer.add_text("run/export_summary", json.dumps(summary, indent=2, sort_keys=True), summary["final_step"])
        writer.flush()
    finally:
        writer.close()

    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
