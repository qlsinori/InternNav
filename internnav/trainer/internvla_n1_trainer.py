# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# 中文训练导读：docs/internvla_n1_training_guide/README.md
# 本入口同时服务两个阶段：普通 Qwen2.5-VL 类负责 System 2 的 token CE；
# InternVLAN1ForCausalLM 类负责冻结 System 2 后的 System 1 轨迹 Flow Matching。

import logging
import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import transformers
from torchvision.transforms import v2

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl_base import replace_qwen2_vl_attention_class
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Trainer,
    TrainerCallback,
)

from internnav.dataset.internvla_n1_lerobot_dataset import make_supervised_data_module
from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.trainer.internvla_n1_argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
)


class WallClockStopAndSaveCallback(TrainerCallback):
    """Stop safely at an optimizer-step boundary after a wall-clock budget.

    All distributed ranks participate in the stop vote.  Setting should_save
    lets Trainer run its normal DeepSpeed collective checkpoint path before
    the training loop returns, avoiding partial checkpoints from SIGTERM.
    """

    def __init__(self, max_train_seconds: int, stop_file: str | None = None):
        self.max_train_seconds = max_train_seconds
        self.stop_file = stop_file
        self.started_at = None
        self.stop_announced = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.monotonic()
        if state.is_world_process_zero:
            logging.info(
                "Wall-clock training budget started: %s seconds; stop file: %s",
                self.max_train_seconds,
                self.stop_file or "disabled",
            )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.monotonic() - self.started_at
        local_stop = elapsed >= self.max_train_seconds
        if self.stop_file:
            local_stop = local_stop or os.path.exists(self.stop_file)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            backend = torch.distributed.get_backend()
            device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
            stop_vote = torch.tensor(int(local_stop), device=device)
            torch.distributed.all_reduce(stop_vote, op=torch.distributed.ReduceOp.MAX)
            local_stop = bool(stop_vote.item())

        if local_stop:
            control.should_log = True
            control.should_save = True
            control.should_training_stop = True
            if state.is_world_process_zero and not self.stop_announced:
                logging.info(
                    "Wall-clock stop requested after %.1f seconds at optimizer step %s; "
                    "forcing a DeepSpeed checkpoint before normal exit",
                    elapsed,
                    state.global_step,
                )
                self.stop_announced = True
        return control


class ResumeStateControlCallback(TrainerCallback):
    """Apply explicitly requested control settings after resume state loads.

    Transformers restores ``save_steps`` from ``trainer_state.json`` after it
    has computed the values from the current command line.  That is normally
    useful for an identical restart, but it prevents intentionally changing
    the checkpoint cadence for a much longer continuation.  Keep the override
    opt-in so ordinary runs preserve the upstream resume behavior.
    """

    def on_train_begin(self, args, state, control, **kwargs):
        forced_save_steps = os.environ.get("INTERNNAV_RESUME_SAVE_STEPS")
        if forced_save_steps:
            forced_save_steps = int(forced_save_steps)
            if forced_save_steps <= 0:
                raise ValueError("INTERNNAV_RESUME_SAVE_STEPS must be positive")
            previous_save_steps = state.save_steps
            state.save_steps = forced_save_steps
            if state.is_world_process_zero:
                logging.info(
                    "Resume checkpoint cadence override: save_steps %s -> %s",
                    previous_save_steps,
                    forced_save_steps,
                )
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if os.environ.get("INTERNNAV_FORCE_FINAL_CHECKPOINT") == "1" and state.global_step >= state.max_steps:
            control.should_save = True
        return control


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def set_model(model_args, model):
    """Apply the stage-specific freeze policy described in the training guide.

    Stage A passes all three tune flags as True and trains the Qwen VLM. Stage B
    passes them as False, then selectively re-enables the System 1 modules below.
    """
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        # model.lm_head.requires_grad = False
        for n, p in model.lm_head.named_parameters():
            p.requires_grad = False

    # Stage B 的梯度仍可“穿过”冻结的 Qwen transformer 流向 latent_queries；
    # requires_grad=False 只是不更新 Qwen 参数，并不会切断计算图。
    if 'nextdit' in model_args.system1:
        modules = [
            'action_encoder',
            'action_decoder',
            'traj_dit',
            'cond_projector',
            'memory_encoder',
            'rgb_resampler',
            'rgb_model',
        ]
        for n, p in model.model.named_parameters():
            if any(k in n for k in modules):
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True
    elif 'navdp' in model_args.system1:
        for n, p in model.model.navdp.named_parameters():
            if "rgb_model" not in n:
                p.requires_grad = True
        model.model.latent_queries.requires_grad = True


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Dataset construction happens before Trainer.__init__, so seed here to
    # make dataset sampling, prompt variants and augmentation reproducible.
    transformers.set_seed(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ColorJitter(brightness=0.2, saturation=0.2),
                v2.RandomPosterize(bits=4),
                v2.RandomAdjustSharpness(sharpness_factor=1.5),
                v2.RandomAutocontrast(),
                v2.ToPILImage(),
                v2.Resize((data_args.resize_h, data_args.resize_w)),
            ]
        )
    else:
        data_args.transform_train = v2.Resize((data_args.resize_h, data_args.resize_w))

    # 注意：这里通过 checkpoint 路径字符串选择 DualVLN 类，而不是读取一个独立 CLI 开关。
    # 因此 Stage B 的输入路径需要保留 `InternVLA-N1-System2` 这一名称。
    if 'internvla-n1-system2' in model_args.model_name_or_path.lower():
        model = InternVLAN1ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "internvla-n1"
    elif "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if data_args.model_type == "internvla-n1":
        model.get_model().initialize_vision_modules(model_args=model_args)
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    # Dataset/collator 返回的字典会被 HF Trainer 原样展开为 model(**batch)。
    # Trainer 没有自定义 compute_loss：loss 的含义完全由上面选择的模型类决定。
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)  # noqa: F821
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[ResumeStateControlCallback()],
        **data_module,
    )
    if trainer.is_world_process_zero():
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in trainer.model.parameters())
        if total_params:
            print(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )
        else:
            # ZeRO-3 may replace parameters with zero-sized local placeholders
            # before this diagnostic runs. This must never block training.
            print("Trainable parameter count unavailable after ZeRO-3 partitioning")
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    if training_args.skip_final_model_save:
        logging.info("Skipping final model save as requested (smoke-test mode)")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
