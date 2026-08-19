# 用现有数据训练 System 2：本机运行手册

## 结论先说

现有数据足够训练 System 2。正式复现使用 **R2R + RxR、8 个相机配置**；输入基础模型使用共享盘的
Qwen2.5-VL-7B-Instruct，训练结果写到个人 CKPT 空间：

```text
基础模型：/mnt/cpfs/zbl-cpfs-new/Models/Qwen2.5-VL-7B-Instruct
训练数据：/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/traj_data
建议输出：/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2
```

原始 [train_system2.sh](../../scripts/train/qwenvl_train/train_system2.sh) 是 8 节点、每节点 8 卡的
Slurm 脚本，不能直接在当前单机开发环境执行。本机使用
[train_system2_local.sh](../../scripts/train/qwenvl_train/train_system2_local.sh)。

## 1. 你在训练什么

System 2 是 Qwen2.5-VL-7B 的全参数监督微调。训练输入是 instruction、历史/当前前视 RGB，以及
需要 waypoint 时的当前朝下 RGB；标准答案仍然全是文本：

```text
pixel-goal：↓  →  加入朝下图  →  u v
turn：       ← / → / 箭头串
stop：       STOP（数据列表中复制 5 份）
```

只对 assistant 回答 token 计算 next-token Cross Entropy。System 2 阶段没有 trajectory query，
不计算 System 1 的轨迹 MSE。

## 2. 现有数据

| 数据 | scenes | episodes | frames | System 2 用法 |
|---|---:|---:|---:|---|
| R2R | 61 | 10,684 | 730,253 | 4 个相机配置 |
| RxR | 59 | 19,543 | 2,013,819 | 4 个相机配置 |
| ScaleVLN | 794 | 77,323 | 4,586,002 | 官方当前 Stage A 默认未启用 |

R2R/RxR 的 5 套 RGB、depth、pose、goal、horizon setting 均存在。官方 Stage A 的 8 个配置是：

```text
r2r_125cm_0_30       front=125cm_0deg,  look-down=125cm_30deg
r2r_125cm_0_45       front=125cm_0deg,  look-down=125cm_45deg
r2r_60cm_15_15       front=60cm_15deg,  look-down=60cm_15deg
r2r_60cm_30_30       front=60cm_30deg,  look-down=60cm_30deg
rxr_125cm_0_30       同上四种配置，数据换成 RxR
rxr_125cm_0_45
rxr_60cm_15_15
rxr_60cm_30_30
```

`--vln_dataset_use` 接受上面的配置名，不接受文件路径。新增的 `--vln_data_root` 才负责指定
`r2r/rxr/scalevln` 三个目录的共同父目录。

## 3. 安装训练附加依赖

先在同一个终端打开代理，然后安装 DeepSpeed：

```bash
cd /workspace/flow/work_space/InternNav
clashctl on
bash scripts/train/qwenvl_train/setup_system2_training.sh
```

当前 VLN 数据是逐帧 JPG/PNG，不需要 MP4 decoder。代码已将 `decord`、`torchcodec` 改为可选依赖，
也不再依赖 pandas/tabulate。若以后混入通用视频 SFT 数据，再单独安装对应 decoder。

## 4. 先做真实单场景、单步 smoke

这一步仍会加载 7B 模型并真实执行一次 forward/backward，但只扫描一个 R2R scene，只训练 1 step：

```bash
cd /workspace/flow/work_space/InternNav

CUDA_VISIBLE_DEVICES=0,1 \
VLN_DATASETS=r2r_125cm_0_30 \
VLN_SCENE_IDS=17DRP5sb8fy \
MAX_STEPS=1 \
GRAD_ACCUM_STEPS=1 \
DATALOADER_NUM_WORKERS=0 \
SAVE_STRATEGY=no \
SKIP_FINAL_MODEL_SAVE=True \
OUTPUT_DIR=/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2-smoke \
bash scripts/train/qwenvl_train/train_system2_local.sh
```

`SKIP_FINAL_MODEL_SAVE=True` 只用于 smoke，避免退出时额外写约 16 GB 完整权重；仍会保留 trainer state
和 processor。正式训练默认保存完整模型。如果输出目录中已有 `checkpoint-*`，trainer 会自动续训。

smoke 验收：

- 两个 rank 都成功加载模型和同一份单 scene dataset；
- 日志显示 vision、merger、LLM 均可训练；
- 完成 `1/1` step，loss 是有限数，不出现 NaN；
- 没有 OOM、FlashAttention ABI、NCCL、缺图或缺列错误；
- 输出目录包含 processor 和 trainer state；smoke 按上述设置不会保存完整权重。

## 5. 双卡正式训练

```bash
cd /workspace/flow/work_space/InternNav

CUDA_VISIBLE_DEVICES=0,1 \
OUTPUT_DIR=/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2 \
bash scripts/train/qwenvl_train/train_system2_local.sh
```

本机默认配置为：

- 2 个进程、双 A800；
- DeepSpeed ZeRO-3；
- BF16 + FlashAttention 2；
- 每卡 micro batch 1；
- 梯度累积 64，保持官方 global batch `1 × 64 × 2 = 128`；
- R2R + RxR 全 8 camera configs；
- 2 epochs；
- 每 5,000 optimizer steps 保存，最多保留 5 个 checkpoint；
- `report_to=none`，不依赖 W&B。

这是数百万条训练 item 的全参数 7B SFT，不是短任务。先观察 smoke 的单步耗时和显存，再估计总时长。

## 6. 如果只想先训练 R2R

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLN_DATASETS=r2r_125cm_0_30,r2r_125cm_0_45,r2r_60cm_15_15,r2r_60cm_30_30 \
GRAD_ACCUM_STEPS=64 \
OUTPUT_DIR=/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2-R2R-only \
bash scripts/train/qwenvl_train/train_system2_local.sh
```

这属于 R2R-only 实验，不等于官方完整 System 2 训练，因为官方脚本还加入了 RxR。

## 7. checkpoint 如何接到 Stage B

Stage B 的 `--model_name_or_path` 指向训练完成目录：

```text
/mnt/cpfs/zbl-cpfs-new/CKPT/flow/InternNav/InternVLA-N1-System2
```

目录名必须保留 `InternVLA-N1-System2`。当前 trainer 根据路径字符串是否包含这个名字来决定加载
自定义 DualVLN 模型类；随意改名可能使 Stage B 错误地加载原生 Qwen 类。

## 8. 已知实现细节

1. `vision_tower_lr=5e-6` 在当前优化器代码中只有同时设置 `mm_projector_lr` 才会建立独立参数组。
   为忠实保留当前官方脚本行为，本地 launcher 未额外传 `mm_projector_lr`，因此 vision 实际使用全局
   `2e-5`。这是源码实际行为，不是文档笔误。
2. `%1` 只在完整读取数据后随机保留 1%，不能替代 `VLN_SCENE_IDS` 做快速 smoke。
3. 当前 loader 每个 camera config 都会重新扫描相同的 R2R/RxR Parquet；System 2 已缩减为只读
   `action/goal/horizon` 三类必要列，System 1 才会额外读取 `pose`，但正式初始化仍会较慢。
4. 本地 launcher 自动使用空目录中的训练；只要发现 `checkpoint-*`，trainer 会自动 resume。
5. 当前运行环境如果看不到 `/dev/nvidia*`，不能启动训练。必须在分配了 GPU 的实例中执行。

## 9. 数据到 loss 的源码顺序

1. [配置名与路径映射](../../internnav/dataset/internvla_n1_lerobot_dataset.py)
2. `get_annotations_from_lerobot_data()`：meta + Parquet 四列
3. `NavPixelGoalDataset.__init__()`：分成 waypoint、turn、STOP
4. `NavPixelGoalDataset.__getitem__()`：读取 RGB，构造多轮文本
5. `preprocess_qwen_2_visual()`：只保留 assistant labels
6. `DataCollatorForSupervisedDataset`：pad 文本、拼接视觉 patch
7. [训练入口](../../internnav/trainer/internvla_n1_trainer.py)：原生 Qwen forward 返回 token CE
