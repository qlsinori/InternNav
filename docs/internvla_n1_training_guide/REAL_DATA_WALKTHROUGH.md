# 一条真实 R2R 数据怎样变成 DualVLN 的两个 loss

这篇文档不再从抽象的 `episode`、`sample` 和 `batch` 开始，而是只追踪本机上一条真实数据：

```text
raw R2R episode 515
  -> 按 scene + 唯一同文 instruction 匹配
LeRobot scene 17DRP5sb8fy / episode 0
  -> Parquet frame 4
System 2 文本真值：↓，然后 258 332
  -> pose 片段生成局部轨迹
System 1 连续真值：[6, 32, 3]
```

读完这一条，再回到[完整训练代码导读](README.md)，会更容易理解其他 R2R、RxR 和 ScaleVLN
样本只是同一套数据契约的不同实例。

如果这些词和 shape 仍然陌生，请先读
[VLN 小白版：相机 setting、frame、标签和 loss](examples/r2r_17DRP5sb8fy_ep0_frame4/parquet_schema.md)。
它从真实图片和白话词表开始，本页默认你已经理解那些基础概念。

如果你希望先从全局层级开始，请打开 [DualVLN 数据结构总图](DATA_STRUCTURE.md)。

## 1. 先运行检查脚本

原始 R2R 是压缩的 `train.json.gz`，Parquet 又是二进制列式格式。它们不能像普通 JSON 或 CSV
那样在 VS Code 中直接阅读，这不代表文件损坏。仓库中的
[inspect_vln_training_sample.py](../../scripts/inspect_vln_training_sample.py) 会只读原始文件，并把这一条
数据导出成可以直接打开的 JSON、CSV 和图片：

```bash
cd /workspace/flow/work_space/InternNav

.venv/bin/python scripts/inspect_vln_training_sample.py \
  --output-dir docs/internvla_n1_training_guide/examples/r2r_17DRP5sb8fy_ep0_frame4
```

脚本不加载 Qwen，不启动 Habitat，不使用 GPU，也不会改写共享盘数据。它只做下面几件事：

1. 解压并筛出 raw R2R episode 515；
2. 读取 LeRobot meta 和 episode 0 的 Parquet；
3. 导出 46 帧的扁平 CSV 和 frame 4 的完整 JSON；
4. 复现 dataset 的 sample 切分、history 选择和轨迹真值预处理；
5. 复制三张原始传感器图，并生成 waypoint 与轨迹预览。

若环境提示缺少 `pyarrow`，安装项目已经声明的版本后重跑：

```bash
uv pip install --python .venv/bin/python pyarrow==21.0.0
```

版本声明见 [requirements/model_requirements.txt](../../requirements/model_requirements.txt)。脚本生成后，优先打开：

- [图文检查报告](examples/r2r_17DRP5sb8fy_ep0_frame4/README.md)
- [三种传感器输入总览](examples/r2r_17DRP5sb8fy_ep0_frame4/overview.jpg)
- [解压后的 raw episode 515](examples/r2r_17DRP5sb8fy_ep0_frame4/raw_r2r_episode_515.json)
- [LeRobot info.json 本地副本](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_info.json)
- [LeRobot episodes.jsonl 本地副本](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_episodes.jsonl)
- [LeRobot tasks.jsonl 本地副本](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_tasks.jsonl)
- [LeRobot episode 0 meta](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_episode_000000.json)
- [Parquet frame 4 的全部字段](examples/r2r_17DRP5sb8fy_ep0_frame4/parquet_frame_000004.json)
- [46 帧扁平表](examples/r2r_17DRP5sb8fy_ep0_frame4/frames.csv)
- [实际 decision sample 切分](examples/r2r_17DRP5sb8fy_ep0_frame4/decision_samples.json)
- [System 1 轨迹真值 CSV](examples/r2r_17DRP5sb8fy_ep0_frame4/trajectory_gt.csv)
- [结构化样本与 loss 摘要](examples/r2r_17DRP5sb8fy_ep0_frame4/sample_summary.json)

后文的所有数字都来自这些真实导出文件，不是虚构示例。

## 2. 第一层：raw R2R episode 515 是一道导航题

在 [raw_r2r_episode_515.json](examples/r2r_17DRP5sb8fy_ep0_frame4/raw_r2r_episode_515.json) 中，关键字段是：

```text
episode_id:    515
trajectory_id: 338
scene_id:      mp3d/17DRP5sb8fy/17DRP5sb8fy.glb
instruction:   Exit the bedroom, enter the bathroom, wait at the toilet.
```

它还给出 agent 的 `start_position`、`start_rotation`、目标位置、geodesic distance 和一条
`reference_path`。这些信息足以让 Habitat 创建一道评测题，并在 episode 结束后计算 NE、SR、SPL
等指标。

但 raw episode **没有**下面这些监督训练所需的数据：

- 每个时刻看到的 RGB 和 depth；
- 每一帧的相机 4x4 pose；
- 专家逐帧执行的 action；
- 朝下图上的局部 pixel waypoint。

因此 `train.json.gz` 不能直接送进 `NavPixelGoalDataset`。它描述的是“题目是什么”，不是“专家在
每一步看到了什么、做了什么”。

## 3. raw episode 515 如何对应到 LeRobot episode 0

LeRobot 的
[episodes.jsonl 本地副本](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_episodes.jsonl)
中有这一行：

```json
{"episode_index": 0, "tasks": ["Exit the bedroom, enter the bathroom, wait at the toilet. "], "length": 46}
```

这里的 `episode_index=0` 是 scene 目录内部的 LeRobot 编号，不等于 raw R2R 的
`episode_id=515`，也不等于 `trajectory_id=338`。当前 LeRobot meta 没有保存一张显式的
`515 -> 0` ID 映射表。

检查脚本使用下面两个条件反查 raw train：

1. scene 都是 `17DRP5sb8fy`；
2. instruction 去掉首尾空格后完全相同。

在 raw R2R train 中，这个 scene 与这句 instruction 的组合只有一条匹配，因此能唯一定位到
episode 515。这里应该理解为“通过内容唯一匹配”，不能把两套 ID 直接画等号。

LeRobot scene 根目录由四类文件组成：

```text
17DRP5sb8fy/
├── meta/info.json
├── meta/episodes.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/
    ├── observation.images.rgb.125cm_0deg/
    ├── observation.images.rgb.125cm_30deg/
    └── observation.images.depth.125cm_30deg/
```

- `episodes.jsonl` 将 episode 0、instruction 和长度 46 关联起来；
- Parquet 保存 46 行结构化帧数据；
- `videos/` 下的 JPG/PNG 是每一帧真实观察；
- `info.json` 描述 scene 级 schema 和统计信息。

`info.json` 中的逻辑 feature 定义可能把标量写成 shape `[1]`；实际 Parquet schema 中
`action` 和 `relative_goal_frame_id.*` 是 scalar，`goal.*` 才是长度为 2 的 list。阅读 loader
时应以实际 Parquet 与 `table.to_pylist()` 的结果为准。

## 4. 第二层：Parquet frame 4 到底存了什么

完整原始 Parquet 有 56 列。训练这条 DualVLN 链路只读取以下几类字段：

| frame 4 字段 | 真实值 | 在 dataset 中的用途 |
|---|---|---|
| `action` | `1` | 先整体左移一帧，给离散 turn/STOP 样本使用 |
| `pose.125cm_30deg` | `4x4` float matrix | 生成 System 1 连续局部轨迹 |
| `goal.125cm_30deg` | `[258, 332]` | 生成 System 2 的 waypoint 文本 |
| `relative_goal_frame_id.125cm_30deg` | `11` | 决定 waypoint horizon 和 pose 截取范围 |
| `frame_index` | `4` | LeRobot 帧索引 |
| `episode_index` | `0` | LeRobot episode 索引 |
| `timestamp` | 约 `0.1333` | 数据索引；当前 loss 不使用 |

frame 4 的完整 pose 是：

```text
[-0.7071, -0.3536,  0.6124,  0.1768]
[-0.7071,  0.3536, -0.6124, -0.1768]
[ 0.0000, -0.8660, -0.5000,  1.2500]
[ 0.0000,  0.0000,  0.0000,  1.0000]
```

其余派生 prompt、local point goal 等列也可以在
[parquet_frame_000004.json](examples/r2r_17DRP5sb8fy_ep0_frame4/parquet_frame_000004.json) 和
[VLN 小白版及 56 列字段附录](examples/r2r_17DRP5sb8fy_ep0_frame4/parquet_schema.md) 中看到，但当前
`NavPixelGoalDataset` 不读取它们。Parquet 到 Python record 的入口是
[get_annotations_from_lerobot_data()](../../internnav/dataset/internvla_n1_lerobot_dataset.py)。

## 5. `action` 为什么要左移一帧

dataset 明确执行：

```python
actions = item["actions"][1:] + [0]
```

也就是说，构造 frame `t` 的 decision sample 时使用 Parquet frame `t+1` 的 action；最后再人工
补一个 STOP。对当前例子：

```text
Parquet frame 4: action = 1
Parquet frame 5: action = 3

loader 对 frame 4 使用的 next action = 3 = →
```

frame 4 有有效 pixel goal，所以它的主要 System 2 target 是 waypoint，而不是这个箭头。
action 左移主要影响两类情况：

- pixel goal 无效时，决定该样本是丢弃还是构造箭头序列；
- episode 末尾由 loader 明确构造 STOP 样本。

相关分支在
[NavPixelGoalDataset.__init__()](../../internnav/dataset/internvla_n1_lerobot_dataset.py) 中。

## 6. episode 0 实际被拆成多少训练样本

官方参数 `sample_step=4`，所以 loader 检查 frames `0,4,8,...,44`，然后额外创建最后一帧的
STOP。真实切分如下：

| frame | relative goal | waypoint / next action | 结果 |
|---:|---:|---|---|
| 0 | 25 | `[453,205]` | pixel-goal |
| 4 | 11 | `[258,332]` | pixel-goal，本文主例 |
| 8 | -1 | `↑` | 丢弃：无 waypoint 且下一动作是 forward |
| 12 | 13 | `[251,270]` | pixel-goal |
| 16 | 8 | `[376,357]` | pixel-goal |
| 20 | -1 | `↑` | 丢弃 |
| 24 | -1 | `→` | turn，文本 target 为 `→` |
| 28 | 15 | `[311,180]` | pixel-goal |
| 32 | 11 | `[198,231]` | pixel-goal |
| 36 | 8 | `[265,256]` | pixel-goal |
| 40 | 4 | `[318,342]` | pixel-goal |
| 44 | -1 | `↑` | 丢弃 |
| 45 | loader 创建 | `STOP` | STOP，Stage A 复制 5 份 |

所以对“这个 episode、这个相机配置”：

- Stage A：8 个 pixel-goal + 1 个 turn + 5 份 STOP，共 14 个 list item；
- Stage B：`pixel_goal_only=True`，只保留 8 个 pixel-goal item。

这也说明 episode、decision sample 和 batch 不是同一个东西：一个 46 帧 episode 会拆成许多训练
item，而 DataLoader 的一个 batch 又会组合来自不同 episode 的多个 item。

## 7. frame 4 的 history 和真实图像

frame 4 之前只有 frames `0,1,2,3`。虽然配置写的是 `num_history=8`，代码执行
`unique(linspace(0, 3, 8))` 后，实际 history 只有：

```text
[0, 1, 2, 3]
```

因此 System 2 的第一轮输入包含：

```text
instruction
+ 历史前视 RGB frames 0,1,2,3
+ 当前前视 RGB frame 4
```

对应的真实媒体可以直接打开：

- [frame 4 前视 RGB](examples/r2r_17DRP5sb8fy_ep0_frame4/front_rgb.jpg)：`125cm_0deg`
- [frame 4 朝下 RGB](examples/r2r_17DRP5sb8fy_ep0_frame4/lookdown_rgb.jpg)：`125cm_30deg`
- [带 waypoint 标记的朝下 RGB](examples/r2r_17DRP5sb8fy_ep0_frame4/lookdown_goal_overlay.jpg)
- [frame 4 原始 16-bit depth](examples/r2r_17DRP5sb8fy_ep0_frame4/depth_raw_mm.png)
- [0～5m 人眼预览](examples/r2r_17DRP5sb8fy_ep0_frame4/depth_preview_0_5m.png)

RGB 原图为 `640x480` JPEG；depth 是 `640x480`、单位为毫米的 16-bit PNG。dataset 将 depth
除以 1000 转成米，再把大于 5m 的值截断到 5m。需要注意：官方 DualVLN 的
`nextdit_async` 分支虽然收到 `traj_depths`，实际只用 RGB 构造 System 1 memory。

图像读取与 history 选择位于
[NavPixelGoalDataset.__getitem__()](../../internnav/dataset/internvla_n1_lerobot_dataset.py)。

## 8. System 2：`[258,332]` 如何变成文本 CE

### 8.1 先明确 `u v` 与内部 `[row,column]`

Parquet 的 `goal.125cm_30deg=[258,332]` 按下面的图像坐标解释：

```text
u = 258 = 水平方向 x = column
v = 332 = 垂直方向 y = row
```

dataset 保持数组原顺序，直接构造文本：

```python
f"{action[0]} {action[1]}"  # "258 332"
```

所以 Qwen 学习的是文本顺序 `u v`，即 `x y`，不是内部数组常见的 `[row,column]`。

推理时，[internvla_n1_policy.py](../../internnav/model/basemodel/internvla_n1/internvla_n1_policy.py)
从生成文本中解析出：

```text
coord = [258, 332]              # 模型文本 u,v
pixel_goal = [coord[1],coord[0]]
           = [332, 258]         # 内部 v,u，也就是 row,column
```

这次 swap 是“文本 `u v`”与“policy 输出的内部 `[row,column]`”之间的接口转换，不是把 waypoint
移到另一个位置。下游函数对坐标顺序的接口并不完全统一，继续传递或画图时必须看调用点；本文的
叠加图直接使用原始文本约定，在 OpenCV/Pillow 图像上画 `(x,y)=(u,v)=(258,332)`。

### 8.2 这条样本的两轮对话

frame 4 是有效 waypoint 样本，因此对话真值是：

```text
user:
  instruction + history RGB + current front RGB
assistant:
  ↓
user:
  current look-down RGB
assistant:
  258 332
```

`↓` 的含义是请求/触发下一张朝下观察，不是连续轨迹中的一个 `(dx,dy,d_yaw)`。

`preprocess_qwen_2_visual()` 将 system、user、图像占位和 padding 对应的 labels 设成 `-100`；
assistant 回答对应 token 才参与原生 Qwen 的 shifted causal cross entropy。因此：

- instruction 与图片是条件，不直接有 label；
- `↓` 与 `258 332` 是语言 token 分类目标；
- 这里没有二维坐标回归头，也没有 pixel L1/L2 loss；
- `258 332` 与 `259 332` 相差一个像素，并不会自动得到“更小的几何误差”，而是 token 序列预测
  是否正确的问题。

对话构造见
[NavPixelGoalDataset.__getitem__()](../../internnav/dataset/internvla_n1_lerobot_dataset.py)，label mask 见
[preprocess_qwen_2_visual()](../../internnav/dataset/internvla_n1_lerobot_dataset.py)。

## 9. System 1：同一 frame 4 如何得到 `[6,32,3]`

frame 4 的 `relative_goal_frame_id=11`。loader 为 waypoint 保留起点和终点，所以 pose slice 是：

```text
Python slice: poses[4:16]
实际全局帧: 4,5,...,15
shape:       [12,4,4]
```

随后 `goal_len=11`，默认每 2 帧选择一个局部 anchor：

```text
anchor offsets: 0,2,4,6,8,10
global frames:  4,6,8,10,12,14
F = 6
```

对每个 anchor，代码都从该 anchor 到 frame 15 的剩余 pose 独立生成一条局部轨迹：

1. 把 4x4 相机 pose 变换到当前 anchor 的机器人局部坐标系；
2. 取平面 `(x,y)`，过滤位移平方不大于 `0.05` 的小运动；
3. 用 CubicSpline 平滑并重采样 33 个位置；
4. 相邻位置作差，得到 32 个 `(dx,dy,delta_yaw)`；
5. 将 `dx,dy` 乘 4，yaw 不缩放；
6. 裁剪或补零到固定 `[32,3]`。

6 个 anchor 堆叠后的真实监督就是：

```text
traj_poses.shape = [6, 32, 3]

6  = 当前 decision 内的 anchor 数 F
32 = 每个 anchor 预测的未来轨迹 step 数
3  = normalized dx, normalized dy, delta_yaw
```

可以直接阅读：

- [完整轨迹 GT JSON](examples/r2r_17DRP5sb8fy_ep0_frame4/trajectory_gt.json)
- [逐 anchor、逐 step 的轨迹 GT CSV](examples/r2r_17DRP5sb8fy_ep0_frame4/trajectory_gt.csv)
- [首个 anchor 轨迹图](examples/r2r_17DRP5sb8fy_ep0_frame4/trajectory_first_anchor.png)

这条真实数据中，最后一个 anchor frame 14 到终点的运动在小运动过滤后得到全零轨迹；它仍然是
真实 anchor，并参与 loss。代码只有 batch 级 `video_frame_num` mask，用于屏蔽 collator 为对齐
不同样本的 `F` 而复制出的假 anchor；没有单独屏蔽一条 `[32,3]` 中的补零尾部。

轨迹预处理见
[interpolate_and_resample_trajectory()](../../internnav/dataset/internvla_n1_lerobot_dataset.py) 和
[DataCollatorForSupervisedDataset](../../internnav/dataset/internvla_n1_lerobot_dataset.py)。

## 10. `[6,32,3]` 最终怎样计算 Flow Matching loss

单样本进入 collator 前是 `[F,32,3]=[6,32,3]`。组成 batch 后，collator 将不同样本的 `F`
补到 batch 内 `Fmax`，并用 `video_frame_num` 记录真实 anchor 数。模型把 batch 和 anchor 两维展平：

```text
x0 = traj_poses.flatten(0,1)
shape = [B*Fmax,32,3]
```

对每个 anchor 轨迹采样高斯噪声 `epsilon` 与随机时间 `sigma`：

```text
x_sigma = (1-sigma)*x0 + sigma*epsilon
target  = epsilon - x0
```

NextDiT 预测相同 shape 的 velocity，损失是所有有效 anchor、32 steps、3 dimensions 上的 MSE：

```text
sum(mask * (v_pred - (epsilon-x0))^2)
------------------------------------------------
valid_anchor_count * 32 * 3
```

Stage B 的自定义 forward 虽然仍计算 `lm_head` logits，但不计算语言 CE；`labels is not None` 只用于
触发轨迹训练分支。4 个 trajectory query 也没有 token CE，它们通过这个 Flow Matching MSE
间接学习。

精确实现见
[InternVLAN1ForCausalLM.forward()](../../internnav/model/basemodel/internvla_n1/internvla_n1.py)。

## 11. 从原始字段到 loss 的总表

| 原始数据 | 在 frame 4 的真实实例 | 送到哪里 | 直接参与哪个 loss |
|---|---|---|---|
| instruction | `Exit the bedroom...` | Qwen 条件 | 不直接参与 |
| history/current RGB | frames `0,1,2,3,4` | Qwen 视觉条件 | 不直接参与 |
| look-down RGB | frame 4，`125cm_30deg` | Qwen + System 1 RGB memory | 不直接参与 |
| `goal.*` | `[258,332]` | 文本 `258 332` | Stage A token CE |
| `action` | 左移后 frame 4 为 `→` | turn/STOP 分支；本 pixel sample 不作为 target | 仅相应文本样本的 token CE |
| `relative_goal_frame_id.*` | `11` | 决定 sample 是否有效及 pose horizon | 不直接参与 |
| `pose.*` | frames 4～15 | 生成 `[6,32,3]` 的 `x0` | Stage B Flow Matching MSE |
| depth | frame 4 等 | loader/collator | `nextdit_async` 不使用 |
| frame/timestamp/index | frame 4、约 0.1333s | 数据定位 | 不直接参与 |

最重要的是，不要把两阶段误解为同一次 forward 的加权和：

```text
Stage A: instruction + RGB -> assistant 文本 -> causal token CE

Stage B: instruction + RGB + trajectory queries + pose-derived x0
       -> NextDiT velocity -> Flow Matching MSE
```

官方训练代码中不存在 `token_CE + lambda * trajectory_MSE` 这一联合 loss。

## 12. 沿源码继续阅读

现在可以带着 frame 4 的真实值按下面顺序读代码：

1. [inspect_vln_training_sample.py](../../scripts/inspect_vln_training_sample.py)：先看人类可读导出怎样生成；
2. [get_annotations_from_lerobot_data()](../../internnav/dataset/internvla_n1_lerobot_dataset.py)：meta + Parquet 怎样汇总；
3. [NavPixelGoalDataset](../../internnav/dataset/internvla_n1_lerobot_dataset.py)：episode 怎样拆成 pixel/turn/STOP；
4. [preprocess_qwen_2_visual()](../../internnav/dataset/internvla_n1_lerobot_dataset.py)：哪些 token label 被设成 `-100`；
5. [DataCollatorForSupervisedDataset](../../internnav/dataset/internvla_n1_lerobot_dataset.py)：4 个 query 与变长 `F` 怎样进入 batch；
6. [InternVLAN1ForCausalLM.forward()](../../internnav/model/basemodel/internvla_n1/internvla_n1.py)：`[B*F,32,3]` 怎样得到唯一的 Stage B loss；
7. [internvla_n1_policy.py](../../internnav/model/basemodel/internvla_n1/internvla_n1_policy.py)：推理文本 `u v` 为什么交换成内部 `[v,u]`。

如果在阅读中忘记某个抽象张量代表什么，就回到
[sample_summary.json](examples/r2r_17DRP5sb8fy_ep0_frame4/sample_summary.json) 或
[trajectory_gt.csv](examples/r2r_17DRP5sb8fy_ep0_frame4/trajectory_gt.csv)，先找到 frame 4 的真实值，再对照源码。
