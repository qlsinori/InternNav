#!/usr/bin/env python3
"""Continuously record health for the long System2 epoch-1 continuation.

This sidecar is intentionally read-only with respect to the trainer.  It turns
the trainer, TensorBoard, GPU, checkpoint, and marker state into one small JSON
record that can be audited without scraping a multi-gigabyte training log.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def tensorboard_state(log_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "step": None,
        "epoch": None,
        "loss": None,
        "grad_norm": None,
        "learning_rate": None,
        "last_event_age_seconds": None,
        "finite": False,
    }
    try:
        accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        tag_to_key = {
            "train/loss": "loss",
            "train/grad_norm": "grad_norm",
            "train/learning_rate": "learning_rate",
            "train/epoch": "epoch",
        }
        wall_times: list[float] = []
        steps: list[int] = []
        for tag, key in tag_to_key.items():
            events = accumulator.Scalars(tag)
            if not events:
                continue
            latest = events[-1]
            result[key] = latest.value
            wall_times.append(latest.wall_time)
            steps.append(latest.step)
        if steps:
            result["step"] = min(steps)
        if wall_times:
            result["last_event_age_seconds"] = max(0.0, time.time() - min(wall_times))
        scalar_values = [result[key] for key in tag_to_key.values()]
        result["finite"] = all(
            value is not None and math.isfinite(float(value)) for value in scalar_values
        )
    except (KeyError, OSError, RuntimeError) as error:
        result["read_error"] = f"{type(error).__name__}: {error}"
    return result


def gpu_processes() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        return {
            "compute_process_rows": len(rows),
            "unique_pids": sorted(
                {int(parts[1].strip()) for row in rows if len(parts := row.split(",")) >= 2}
            ),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {
            "compute_process_rows": None,
            "unique_pids": [],
            "read_error": f"{type(error).__name__}: {error}",
        }


def checkpoint_state(run_dir: Path) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    for path in run_dir.glob("checkpoint-*"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if not path.is_dir() or match is None:
            continue
        state = read_json(path / "trainer_state.json")
        step = int(match.group(1))
        try:
            age_seconds = max(0.0, time.time() - path.stat().st_mtime)
        except OSError:
            age_seconds = None
        checkpoints.append(
            {
                "name": path.name,
                "step": step,
                "complete": bool(
                    state
                    and state.get("global_step") == step
                    and (path / "model.safetensors.index.json").is_file()
                    and len(list(path.glob("model-*-of-*.safetensors"))) > 0
                ),
                "save_steps": state.get("save_steps") if state else None,
                "age_seconds": age_seconds,
            }
        )
    checkpoints.sort(key=lambda item: item["step"])
    return {
        "count": len(checkpoints),
        "items": checkpoints,
        "latest": checkpoints[-1] if checkpoints else None,
    }


def collect(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir
    control_dir = run_dir / "control"
    marker = next(
        (
            name
            for name in ("EPOCH1_COMPLETE", "EPOCH1_FAILED", "EPOCH1_RUNNING")
            if (control_dir / name).exists()
        ),
        "MISSING",
    )
    launcher_pid = read_pid(control_dir / "launcher_resume_to_epoch1.pid")
    tensorboard = tensorboard_state(run_dir / "tensorboard" / "live_2913_to_24488")
    checkpoints = checkpoint_state(run_dir)
    root_state = read_json(run_dir / "trainer_state.json") or {}
    final_checkpoint = run_dir / f"checkpoint-{args.target_step}"
    final_state = read_json(final_checkpoint / "trainer_state.json") or {}

    problems: list[str] = []
    if marker == "EPOCH1_FAILED":
        problems.append("trainer failure marker exists")
    if marker == "EPOCH1_RUNNING" and not pid_alive(launcher_pid):
        problems.append("launcher is not alive while run is marked running")
    if tensorboard["step"] is not None and not tensorboard["finite"]:
        problems.append("latest TensorBoard scalar set is non-finite or incomplete")
    event_age = tensorboard["last_event_age_seconds"]
    if marker == "EPOCH1_RUNNING" and event_age is not None and event_age > args.stale_seconds:
        problems.append(f"TensorBoard has not advanced for {event_age:.0f} seconds")
    recent_checkpoint_write = any(
        not item["complete"]
        and item["age_seconds"] is not None
        and item["age_seconds"] <= args.checkpoint_write_grace_seconds
        for item in checkpoints["items"]
    )
    if checkpoints["count"] > args.max_checkpoints and not recent_checkpoint_write:
        problems.append(
            f"checkpoint count {checkpoints['count']} exceeds limit {args.max_checkpoints}"
        )
    if any(
        not item["complete"]
        and (
            item["age_seconds"] is None
            or item["age_seconds"] > args.checkpoint_write_grace_seconds
        )
        for item in checkpoints["items"]
    ):
        problems.append("one or more checkpoint directories are incomplete")

    final_verified = bool(
        marker == "EPOCH1_COMPLETE"
        and root_state.get("global_step") == args.target_step
        and final_state.get("global_step") == args.target_step
        and checkpoints["count"] <= args.max_checkpoints
    )
    if final_verified:
        status = "complete_verified"
    elif problems:
        status = "unhealthy"
    elif recent_checkpoint_write:
        status = "checkpoint_saving"
    else:
        status = "healthy_running"

    return {
        "observed_at_utc": utc_now(),
        "status": status,
        "target_step": args.target_step,
        "marker": marker,
        "launcher_pid": launcher_pid,
        "launcher_alive": pid_alive(launcher_pid),
        "tensorboard": tensorboard,
        "gpu": gpu_processes(),
        "checkpoints": checkpoints,
        "root_global_step": root_state.get("global_step"),
        "final_verified": final_verified,
        "recent_checkpoint_write": recent_checkpoint_write,
        "problems": problems,
    }


def write_record(run_dir: Path, record: dict[str, Any]) -> None:
    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    output = control_dir / "epoch1_health.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    with (run_dir / "logs" / "epoch1_health.jsonl").open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=24488)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--checkpoint-write-grace-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pid_file = args.run_dir / "control" / "monitor_system2_epoch1.pid"
    pid_file.write_text(f"{os.getpid()}\n")
    try:
        while True:
            record = collect(args)
            write_record(args.run_dir, record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.once or record["status"] == "complete_verified":
                return 0
            if record["marker"] == "EPOCH1_FAILED":
                return 1
            time.sleep(args.interval)
    finally:
        if read_pid(pid_file) == os.getpid():
            pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
