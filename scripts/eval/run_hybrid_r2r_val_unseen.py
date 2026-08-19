"""Run distributed R2R val-unseen evaluation for a hybrid DualVLN checkpoint."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from internnav.evaluator import Evaluator


CONFIG_PATH = Path("scripts/eval/configs/habitat_dual_system_cfg.py")
MODEL_PATH = Path(os.environ["INTERNVLA_MODEL_PATH"])
OUTPUT_PATH = Path(os.environ["DUALVLN_OUTPUT_PATH"])
SAVE_VIDEO = os.environ.get("DUALVLN_SAVE_VIDEO", "1") == "1"
VIS_DEBUG = os.environ.get("DUALVLN_VIS_DEBUG", "0") == "1"

if not MODEL_PATH.joinpath("model.safetensors.index.json").is_file():
    raise FileNotFoundError(f"Incomplete hybrid checkpoint: {MODEL_PATH}")

spec = importlib.util.spec_from_file_location("hybrid_r2r_val_unseen_cfg", CONFIG_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load evaluation config: {CONFIG_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

eval_cfg = module.eval_cfg
eval_cfg.agent.model_settings["model_path"] = str(MODEL_PATH)
eval_cfg.agent.model_settings["vis_debug"] = VIS_DEBUG
eval_cfg.agent.model_settings["vis_debug_path"] = str(OUTPUT_PATH / "vis_debug")
eval_cfg.eval_settings["output_path"] = str(OUTPUT_PATH)
eval_cfg.eval_settings["save_video"] = SAVE_VIDEO

evaluator = Evaluator.init(eval_cfg)
original_eval_action = evaluator.eval_action


def eval_action_fp32():
    metrics = original_eval_action()
    return {name: tensor.to(dtype=torch.float32) for name, tensor in metrics.items()}


evaluator.eval_action = eval_action_fp32
print(
    "HYBRID_R2R_VAL_UNSEEN "
    f"rank={evaluator.rank} local_rank={evaluator.local_rank} "
    f"world_size={evaluator.world_size} pending_episodes={len(evaluator.env.episodes)} "
    f"save_video={SAVE_VIDEO} vis_debug={VIS_DEBUG} "
    f"model_path={MODEL_PATH} output_path={OUTPUT_PATH}",
    flush=True,
)

result = evaluator.eval()
print(f"FINAL_RESULT rank={evaluator.rank} result={result}", flush=True)

if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
