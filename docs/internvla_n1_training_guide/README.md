# InternVLA-N1 / DualVLN 训练代码导读

这份文档面向“理解常规深度学习，但刚开始接触 VLN”的读者。它只解释当前仓库中与
**InternVLA-N1 (Dual System) DualVLN** 直接相关的训练链路，重点回答四个问题：

1. VLN 的一个训练样本到底包含什么；
2. 模型实际输出的是文本、像素点还是连续轨迹；
3. System 2 和 System 1 各自怎样计算 loss；
4. 数据、collator、模型和 <code>Trainer</code> 是怎样连起来的。

如果 `episode`、`frame`、`waypoint`、`camera setting` 这些词还不熟，先读
[VLN 小白版：用真实 frame 4 看懂相机 setting、数据和标签](examples/r2r_17DRP5sb8fy_ep0_frame4/parquet_schema.md)。
它先并排展示真实前视图和朝下图，再只讲当前训练真正使用的 4 个字段；56 列技术表放在附录。
之后再读[真实样本逐字段导读](REAL_DATA_WALKTHROUGH.md)和
[已经解包的 frame 4 实例](examples/r2r_17DRP5sb8fy_ep0_frame4/README.md)。对应的
[小脚本](../../scripts/inspect_vln_training_sample.py) 可以把任意同结构样本导出为 JSON、CSV 和图片，
不加载模型、不占 GPU。

如果想先建立整体结构，请直接看 [DualVLN 数据结构总图](DATA_STRUCTURE.md)；其中用 Markdown
内嵌 Mermaid 把磁盘文件、单样本、collator batch 和两个 loss 画在同一张图中。

本文对应仓库提交基线 <code>7a5c624</code>，整理日期为 2026-08-11。源码旁已经加入指向本文的
中文注释；注释和本文都不改变训练行为。

## 1. 先看这些链接

### 1.1 核心训练代码

| 阅读顺序 | 文件 | 重点 |
|---:|---|---|
| 1 | [train_system2.sh](../../scripts/train/qwenvl_train/train_system2.sh) | Stage A 的数据、模型和超参数 |
| 2 | [train_dual_system.sh](../../scripts/train/qwenvl_train/train_dual_system.sh) | Stage B 的冻结策略和轨迹参数 |
| 3 | [internvla_n1_trainer.py](../../internnav/trainer/internvla_n1_trainer.py) | 模型分支、数据模块、HF Trainer |
| 4 | [internvla_n1_lerobot_dataset.py](../../internnav/dataset/internvla_n1_lerobot_dataset.py) | VLN 样本、token label、collator、轨迹真值 |
| 5 | [internvla_n1.py](../../internnav/model/basemodel/internvla_n1/internvla_n1.py) | trajectory query、NextDiT 条件和轨迹 loss |
| 6 | [internvla_n1_arch.py](../../internnav/model/basemodel/internvla_n1/internvla_n1_arch.py) | query、RGB memory、System 1 模块构建 |
| 7 | [nextdit_crossattn_traj.py](../../internnav/model/basemodel/internvla_n1/nextdit_crossattn_traj.py) | 轨迹 DiT 主干的尺寸 |
| 8 | [qwenvl_base.py](../../internnav/trainer/qwenvl_base.py) | FlashAttention patch 与优化器参数分组 |
| 9 | [internvla_n1_argument.py](../../internnav/trainer/internvla_n1_argument.py) | CLI 参数定义 |
| 10 | [inspect_vln_training_sample.py](../../scripts/inspect_vln_training_sample.py) | 把真实 raw/Parquet/RGB/depth/轨迹监督解包成可读文件 |

System 2 的 CE 由依赖包实现，可直接查看本机安装的
[Qwen2_5_VLForConditionalGeneration.forward()](../../.venv/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)。

### 1.2 本机真实数据

| 内容 | 共享盘真实路径（目录路径不要当 Markdown 文件链接点击） |
|---|---|
| VLN-CE R2R 训练轨迹 | <code>/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/traj_data/r2r</code> |
| VLN-CE RxR 训练轨迹 | <code>/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/traj_data/rxr</code> |
| ScaleVLN 训练轨迹 | <code>/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/traj_data/scalevln</code> |
| R2R Habitat episode JSON | <code>/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/raw_data/r2r</code> |
| InternData-N1 数据说明 | <code>/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/README.md</code> |
| 已下载的最终 DualVLN 权重 | [checkpoints/InternVLA-N1-DualVLN](../../checkpoints/InternVLA-N1-DualVLN) |

VS Code 的 Markdown preview 会把以 <code>/mnt/...</code> 开头的链接错误拼到仓库路径后面。为保证
能直接点击，检查脚本已经把真实样例 scene <code>17DRP5sb8fy</code> 的必要内容复制或解包到本文目录：

- [info.json 本地副本](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_info.json)
- [episodes.jsonl 本地副本](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_episodes.jsonl)
- [tasks.jsonl 本地副本](examples/r2r_17DRP5sb8fy_ep0_frame4/lerobot_tasks.jsonl)
- [episode 0 / frame 4 的 Parquet 解包 JSON](examples/r2r_17DRP5sb8fy_ep0_frame4/parquet_frame_000004.json)
- [frame 4 前视 RGB](examples/r2r_17DRP5sb8fy_ep0_frame4/front_rgb.jpg)
- [frame 4 原始 depth](examples/r2r_17DRP5sb8fy_ep0_frame4/depth_raw_mm.png)

### 1.3 相关 Markdown

- [项目主 README](../../README.md)
- [用现有数据训练 System 2：本机运行手册](SYSTEM2_TRAINING_RUNBOOK.md)
- [DualVLN 数据结构总图](DATA_STRUCTURE.md)
- [VLN 小白版：相机 setting、真实 frame、标签和 loss](examples/r2r_17DRP5sb8fy_ep0_frame4/parquet_schema.md)
- [真实数据逐字段导读](REAL_DATA_WALKTHROUGH.md)
- [已导出的真实样本报告](examples/r2r_17DRP5sb8fy_ep0_frame4/README.md)
- [已经跑完的 R2R DualVLN 评测流程](../r2r_dualvln_reproduction/README.md)
- [共享开发机迁移计划](../../../../共享开发机迁移计划.md)

## 2. 六十秒建立整体认识

DualVLN 的训练不是一次 forward 同时优化两个 loss，而是严格的两个阶段：

~~~mermaid
flowchart TD
    A["Qwen2.5-VL-7B-Instruct"] --> B["Stage A: train_system2.sh"]
    D1["instruction + 历史/当前 RGB"] --> B
    T1["assistant 文本: STOP / 箭头 / u v"] --> B
    B --> L1["只对 assistant token 做 causal-LM CE"]
    L1 --> C["InternVLA-N1-System2 checkpoint"]
    C --> D["Stage B: train_dual_system.sh"]
    D2["pixel waypoint 样本 + 4x4 pose + 朝下 RGB"] --> D
    D --> Q["末尾追加 4 个 trajectory query"]
    Q --> S1["NextDiT System 1"]
    S1 --> L2["32x3 轨迹 Flow Matching MSE"]
    L2 --> E["InternVLA-N1-DualVLN checkpoint"]
~~~

最重要的结论是：

- Stage A 只有文本 token CE；像素坐标也是数字文本，不是二维回归头。
- Stage B 冻结 Qwen System 2，只返回连续轨迹的 Flow Matching MSE。
- Stage B 没有 <code>token_loss + lambda * traj_loss</code>；当前实现中不存在这个加权和。
- 4 个 trajectory query 是 4 个连续 latent 条件，不是 4 个动作或 4 个轨迹点。

## 3. VLN 数据的三个层级

### 3.1 Episode：任务定义

VLN 的 episode 可以理解为“一道导航题”。R2R raw episode 一般包含：

~~~text
scene_id
start_position / start_rotation
instruction_text
goal position
reference_path
~~~

本机示例可直接打开
[train.json.gz](../../data/vln_ce/raw_data/r2r/train/train.json.gz)。它供 Habitat 创建环境、放置 agent、
计算 NE/SR/SPL 等评测指标。它不含逐帧 RGB、depth 和动作序列，因而不能直接喂给本项目的
InternVLA-N1 训练 loader。

### 3.2 Trajectory：专家走过的一整条路

训练需要 demonstration。InternData-N1 把每条轨迹存为 LeRobot v2.1 结构：

~~~text
<dataset>/<scene>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   └── tasks.jsonl
├── data/chunk-000/
│   └── episode_000000.parquet
└── videos/chunk-000/
    ├── observation.images.rgb.<camera_setting>/
    └── observation.images.depth.<camera_setting>/
~~~

<code>episodes.jsonl</code> 把 episode ID、自然语言指令和长度关联起来；Parquet 保存每一帧的动作、
位姿、pixel goal 和索引；<code>videos/</code> 保存实际观察。样例 scene 有 75 个 episode、
3237 帧、30 FPS。

### 3.3 Decision sample：某个时刻应该做什么

神经网络不是每次拿整条 episode 算一个 loss。
<code>NavPixelGoalDataset</code> 每隔 <code>sample_step=4</code> 帧选择一个决策时刻，把一条
trajectory 拆成许多监督样本：

~~~text
一条 episode
  -> frame 0 的决策样本
  -> frame 4 的决策样本
  -> frame 8 的决策样本
  -> ...
~~~

因此要区分：

| 名词 | 含义 |
|---|---|
| episode | 一道完整语言导航任务 |
| trajectory | 专家完成该 episode 的逐帧 demonstration |
| decision sample | 在 trajectory 某一帧构造出的一个训练样本 |
| batch | 若干 decision sample 经 collator pad 后的张量 |

## 4. Parquet 中哪些字段真正进入训练

每种相机设置都有三组关键字段：

| 字段 | 类型/shape | 代码如何解释 |
|---|---|---|
| <code>action</code> | <code>int32</code> scalar | 离散环境动作；loader 左移一帧，让当前观察预测下一动作 |
| <code>pose.&lt;setting&gt;</code> | <code>float32 [4,4]</code> | 相机外参；用于生成当前机器人坐标系下的连续轨迹 |
| <code>goal.&lt;setting&gt;</code> | <code>int32 [2]</code> | pixel waypoint，顺序为 <code>u v = x y</code>，并按此顺序训练成文本 |
| <code>relative_goal_frame_id.&lt;setting&gt;</code> | <code>int32</code> scalar | 到局部 waypoint 的未来帧数；<code>-1</code> 表示无有效 pixel goal |
| <code>timestamp</code> | <code>float32</code> scalar | LeRobot 时间索引，本 loader 不用于 loss |
| <code>frame_index</code> 等 | <code>int64</code> scalar | LeRobot 数据索引 |

相机设置名例如：

- <code>125cm_0deg</code>：相机高度 125 cm、水平视角；
- <code>125cm_30deg</code>：高度 125 cm、向下 30°；
- <code>60cm_15deg</code>：高度 60 cm、向下 15°。

代码中的 <code>height/pitch_1/pitch_2</code> 分别选择前视图和生成 waypoint/轨迹所用的朝下视图。

动作文本映射为：

~~~text
0 -> STOP
1 -> ↑
2 -> ←
3 -> →
5 -> ↓
~~~

其中 <code>↓</code> 主要是对话里让 agent 再看一张朝下图的符号。训练数据先把
<code>actions[1:] + [0]</code>，使 frame <code>t</code> 对应接下来要执行的动作，而 episode
最后自然落到 STOP。

## 5. 一条轨迹怎样产生三类 System 2 样本

数据拆分逻辑在 <code>NavPixelGoalDataset.__init__()</code>，对话构造在 <code>__getitem__()</code>。

### 5.1 Pixel waypoint

当 <code>relative_goal_frame_id >= 3</code> 时，构造两轮对话：

~~~text
user:
  instruction + 最多 8 张历史前视 RGB + 当前前视 RGB
assistant:
  ↓
user:
  当前朝下 RGB
assistant:
  u v
~~~

真实 frame 4 的最后输出是 <code>258 332</code>，它是两个数字组成的文本：
<code>u=258</code> 是图像横坐标 <code>x</code>，<code>v=332</code> 是图像纵坐标 <code>y</code>。
代码直接输出 Parquet <code>goal</code> 数组的两个元素。推理 policy 解析后会交换成内部使用的
<code>[row,column]=[v,u]</code>，见
[internvla_n1_policy.py](../../internnav/model/basemodel/internvla_n1/internvla_n1_policy.py)。这里没有
<code>Linear(hidden, 2)</code> 坐标头，也没有 L1/L2 pixel-distance loss。Qwen tokenizer
会把这串字符切成 token，再像普通语言一样预测。

### 5.2 Turn

当 pixel goal 为 <code>-1</code> 且下一动作不是 forward 时，loader 最多向后看
<code>num_future_steps=4</code> 个动作，到再次 forward 为止，并把转向序列转成字符串，例如：

~~~text
assistant: ←←→
~~~

pixel goal 无效但下一动作恰好是 forward 的时刻会被丢弃，因为这里希望 forward 主要由可见
waypoint 表达。

### 5.3 STOP

每个 episode 的最后一帧构造：

~~~text
assistant: STOP
~~~

代码把 STOP 样本复制 5 次，以提高它在 System 2 SFT 中的采样比例。这是数据重采样，不是 loss
函数里的 STOP 权重。

当 <code>pixel_goal_only=False</code> 时三类都保留；当 <code>pixel_goal_only=True</code> 时
只保留第一类，因为只有 pixel waypoint 样本带有 System 1 所需的连续位姿片段。

## 6. 阶段 A：训练 System 2

### 6.1 启动配置

入口是 [train_system2.sh](../../scripts/train/qwenvl_train/train_system2.sh)：

| 参数 | 官方值 | 意义 |
|---|---:|---|
| base model | <code>Qwen/Qwen2.5-VL-7B-Instruct</code> | Qwen VLM |
| datasets | R2R + RxR，8 个相机组合 | 不包含 raw episode JSON |
| <code>pixel_goal_only</code> | <code>False</code> | waypoint、turn、STOP 全保留 |
| <code>system1</code> | <code>none</code> | 不构造 NextDiT |
| tune flags | 全部 <code>True</code> | vision、merger、LLM、lm_head 全训练 |
| epochs | 2 | 无在线 Habitat rollout |
| global LR | <code>2e-5</code> | 实际优化器 LR，特殊情况见第 9 节 |
| dtype | BF16 | 配合 FlashAttention 2 |

模型路径包含 <code>qwen2.5</code>，所以 trainer 实例化 Transformers 原生
<code>Qwen2_5_VLForConditionalGeneration</code>，不是自定义 DualVLN forward。

### 6.2 Token 与 label

<code>preprocess_qwen_2_visual()</code> 做两件关键事：

1. 把每个 <code>&lt;image&gt;</code> 展开为
   <code>vision_start + image_pad x patch数量 + vision_end</code>；
2. 构造与 <code>input_ids</code> 同长的 <code>labels</code>。

label mask 如下：

~~~text
system message                    -> -100
user instruction                 -> -100
历史/当前图像占位 token          -> -100
assistant chat header 前 3 token -> -100
assistant 的 ↓ / STOP / 箭头 / u v -> 真实 token id
batch padding                    -> -100
~~~

<code>CrossEntropyLoss</code> 默认忽略 <code>-100</code>，所以图像和 user prompt 都只是条件，
不直接产生 loss。

### 6.3 Batch 形式

令 <code>B</code> 为 batch size，<code>L</code> 为 pad 后文本长度：

| 张量 | shape | 说明 |
|---|---|---|
| <code>input_ids</code> | <code>[B,L]</code> | 文本和视觉占位 token |
| <code>labels</code> | <code>[B,L]</code> | 仅 assistant 内容位置不是 <code>-100</code> |
| <code>attention_mask</code> | <code>[B,L]</code> | 非 padding 为 True |
| <code>position_ids</code> | <code>[3,B,L]</code> | Qwen2.5-VL 多模态 RoPE |
| <code>pixel_values</code> | <code>[sum(image patches),D_patch]</code> | 动态视觉 patch，不能按普通 <code>[B,C,H,W]</code> 理解 |
| <code>image_grid_thw</code> | <code>[N_image,3]</code> | 每张图的 temporal/height/width patch 网格 |

视觉 patch 沿第 0 维拼接，<code>image_grid_thw</code> 告诉 Qwen 哪些 patch 属于哪张图。

### 6.4 System 2 loss

Qwen forward 先得到：

~~~text
logits: [B, L, vocab_size]
~~~

然后做标准 causal shift：

~~~python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
loss = CrossEntropyLoss()(shift_logits, shift_labels)
~~~

公式为：

\[
L_{S2}=-\frac{1}{|M|}\sum_{t\in M}
\log p_\theta(y_t\mid y_{<t},\text{instruction},\text{images})
\]

<code>M</code> 是 <code>labels != -100</code> 的 assistant token 位置集合。

这意味着真值 <code>258 332</code> 和预测 <code>259 332</code> 不会按“像素差 1”得到较小 loss；
只要 tokenizer 序列不同，就是 token 分类错误。坐标、STOP 和箭头共用同一个词表和同一个 CE。

## 7. 阶段 B：训练 DualVLN / System 1

### 7.1 启动配置和冻结

入口是 [train_dual_system.sh](../../scripts/train/qwenvl_train/train_dual_system.sh)：

| 参数 | 官方值 | 意义 |
|---|---:|---|
| input checkpoint | <code>InternVLA-N1-System2</code> | Stage A 的结果 |
| <code>system1</code> | <code>nextdit_async</code> | RGB memory + NextDiT |
| datasets | R2R/RxR/ScaleVLN 六组，各 <code>%30</code> | 每个配置随机抽 30% |
| <code>pixel_goal_only</code> | <code>True</code> | 只留能构造连续轨迹监督的 waypoint 样本 |
| tune flags | 全部 <code>False</code> | 先冻结 Qwen vision/LLM/lm_head |
| epochs | 3 | 训练 System 1 |
| LR | <code>1e-4</code> | cosine，最低 <code>1e-5</code> |

<code>set_model()</code> 随后重新打开这些参数：

- <code>latent_queries</code>
- <code>action_encoder</code> / <code>action_decoder</code>
- <code>traj_dit</code>
- <code>cond_projector</code>
- <code>memory_encoder</code>
- <code>rgb_resampler</code>
- <code>rgb_model</code>（DepthAnything 中复用的 DINOv2 RGB backbone）

Qwen 参数不更新，但冻结模块仍在计算图中；轨迹 loss 的梯度可以穿过冻结的 Qwen 运算，更新输入端的
<code>latent_queries</code>。

### 7.2 连续轨迹真值怎样生成

对一个有效 waypoint，loader 从当前帧截取到 future goal 的 <code>pose</code> 片段，然后：

1. 用 <code>get_trajectory_relative_to_frame()</code> 把 4x4 相机外参变到当前机器人局部坐标系；
2. 取二维位置 <code>(x,y)</code>；原 pose 的 yaw 不直接作为标签；
3. 删除相邻位移平方不大于 <code>0.05</code> 的小运动段；
4. 若点足够多，用 CubicSpline 平滑；
5. 以 <code>interval=0.1</code> 的名义尺度重采样 33 个二维位置点；
6. 相邻位置做差，得到 32 个 <code>(dx,dy,delta_yaw)</code>；
7. <code>dx,dy *= 4</code>，yaw 不缩放；
8. 裁剪或补零为固定 <code>[32,3]</code>。

最终一个 anchor frame 的监督是：

~~~text
traj_poses: [32, 3]
             32 个未来 step
             3 = normalized dx, normalized dy, delta_yaw
~~~

短轨迹补出的 32-step 尾部没有单独的 step mask，仍参与 loss。推理把平移维度除以 4 恢复尺度，见
[vln_utils.py](../../internnav/model/utils/vln_utils.py)。

实现细节：函数计算 <code>total_distance = sample_length * interval</code>，再对 33 点使用
<code>linspace</code>，所以完整路径的相邻采样间距略大于严格的 0.1 m；阅读源码时应以实现为准。

### 7.3 一个样本为什么含多个 anchor

从当前帧到 waypoint 的朝下图像序列每隔 2 帧取一个；如果超过约 12 个，动态增大间隔。因此单样本
额外返回：

| 张量 | 单样本 shape |
|---|---|
| <code>traj_images</code> | <code>[F,224,224,3]</code>，值域约 <code>[0,1]</code> |
| <code>traj_depths</code> | <code>[F,224,224]</code>，米制并截断到 5 m |
| <code>traj_poses</code> | <code>[F,32,3]</code> |

<code>F</code> 是该样本选中的 anchor 数。collator 按 batch 内最大 <code>Fmax</code> 复制最后一帧
补齐，并保存真实的 <code>video_frame_num</code>。模型的 <code>loss_mask</code> 只屏蔽这些
batch 对齐产生的假 anchor。

### 7.4 四个 trajectory query

Dual collator 在完整对话末尾追加 4 个 ID 为 <code>151667</code> 的占位 token，并记录开始位置
<code>t_s_pos</code>：

~~~text
[instruction / images / assistant waypoint text] + [Q1 Q2 Q3 Q4]
~~~

forward 不使用词表中 ID 151667 的 embedding，而是替换为：

~~~text
latent_queries: [1,4,3584]
~~~

经过冻结的 Qwen transformer 后，截取这四个位置的 hidden states：

~~~text
[B,4,3584]
~~~

这些 query：

- 不是四个文本 token 的分类结果；
- 不是四个动作；
- 不是 32-step 轨迹中的前四步；
- 没有直接 label；
- 通过最终轨迹 MSE 间接学习如何压缩高层计划。

训练时 query 位于真实 waypoint 文本之后；推理时位于 System 2 自己生成的 waypoint 文本之后。
因为 Qwen 是 causal transformer，后一个 query 可以看见前面的 query。

当前 collator 把 query 数硬编码为 4，而 CLI 同时暴露 <code>--n_query</code>。最终 checkpoint 也是
<code>n_query=4</code>；直接修改 CLI 会造成 dataset 与模型不一致。

## 8. System 1 的 Flow Matching loss

### 8.1 RGB memory 条件

记 batch 内补齐后的 anchor 数为 <code>F</code>。每个 anchor 使用两张朝下 RGB：

1. 最初产生 pixel waypoint 时刻的图；
2. 当前 anchor 时刻的图。

<code>nextdit_async</code> 的 shape 流为：

~~~text
RGB pair                         [B*F,2,3,224,224]
DINOv2-S/14 patch features       [B*F,2,256,384]
flatten two views                [B*F,512,384]
MemoryEncoder                    [B*F,512,384]
concat(raw, encoded)             [B*F,512,768]
Q-Former / resampler             [B*F,32,768]
~~~

trajectory query 条件为：

~~~text
Qwen query hidden                [B,4,3584]
repeat for anchors               [B*F,4,3584]
cond_projector                   [B*F,4,768]
~~~

二者拼接得到 NextDiT cross-attention 条件：

~~~text
latents = [memory tokens, trajectory query tokens]
shape   = [B*F, 32+4, 768]
~~~

这里又出现一个“32”：Q-Former 的 32 个 memory token 与未来轨迹的 32 step 恰好同数，但两者没有
一一对应关系。

虽然 dataset 和 collator 都提供 <code>traj_depths</code>，<code>nextdit_async</code> 分支没有
读取它；它只使用 RGB。<code>navdp_async</code> 分支才同时读取 RGB-D。这正是评测表写 RGB、公共
evaluator 仍创建 depth sensor 不矛盾的原因。

### 8.2 带噪轨迹和预测目标

真实轨迹：

~~~text
x0 = traj_poses.flatten(B,F)     [B*F,32,3]
~~~

对每条轨迹采样高斯噪声 <code>epsilon</code> 和 FlowMatch scheduler 的随机时间步。当前 Diffusers
默认 scheduler 有 1000 个训练时刻。代码构造：

\[
x_\sigma=(1-\sigma)x_0+\sigma\epsilon
\]

轨迹主干：

~~~text
x_sigma                           [B*F,32,3]
action_encoder                    [B*F,32,384]
+ sinusoidal trajectory position  [B*F,32,384]
12-layer NextDiT                  [B*F,32,384]
action_decoder                    [B*F,32,3]
~~~

监督目标不是 DDPM 常见的纯噪声 <code>epsilon</code>，而是直线路径的 velocity：

\[
v^*=\epsilon-x_0
\]

代码变量叫 <code>noise_pred</code>，但数学意义是 <code>v_pred</code>。

### 8.3 精确 loss

令 <code>m_j</code> 表示第 <code>j</code> 个 anchor 是否真实存在：

\[
L_{traj}=
\frac{
\sum_{j,t,d}m_j\left(v_{\theta,j,t,d}-v^*_{j,t,d}\right)^2
}{
(\sum_jm_j)\times32\times3
}
\]

也就是对所有有效 anchor、32 个未来 step、3 个轨迹维度求均方误差。

<code>dx,dy</code> 在预处理时乘了 4，所以相对于原始米制误差，它们在平方项中的尺度等效放大
16 倍；代码没有再给 <code>dx/dy/yaw</code> 设置显式 loss 权重。

### 8.4 为什么这里没有语言 CE

自定义 <code>InternVLAN1ForCausalLM.forward()</code> 确实仍计算：

~~~text
logits = lm_head(hidden_states)
~~~

但之后没有 shift logits，也没有调用 <code>CrossEntropyLoss</code>。只要
<code>labels is not None</code>，代码就进入轨迹分支，最终把 <code>loss=L_traj</code> 放入
<code>CausalLMOutputWithPast</code>。

因此 Dual batch 中追加到 <code>labels</code> 的四个 query token ID 只是保持序列对齐，绝不会形成
query token CE。如果把 turn/STOP 样本混入这个阶段，它们没有 <code>traj_images/traj_poses</code>，
custom forward 会缺少必要输入；这就是官方脚本必须设置 <code>pixel_goal_only=True</code> 的原因。

## 9. 优化器、梯度与 checkpoint

### 9.1 Trainer 如何拿到 loss

项目没有为这条链路重写 <code>compute_loss()</code>：

~~~text
Dataset.__getitem__
  -> DataCollatorForSupervisedDataset
  -> Trainer 将 batch 展开成 model(**batch)
  -> model 返回 outputs.loss
  -> Trainer backward
~~~

模型类决定 <code>outputs.loss</code> 的含义：原生 Qwen 类返回 token CE，自定义 InternVLAN1 类
返回轨迹 MSE。

### 9.2 优化器 monkey patch

导入 [qwenvl_base.py](../../internnav/trainer/qwenvl_base.py) 时，项目会全局替换
<code>Trainer.create_optimizer</code>。

一个需要特别注意的源码行为是：<code>vision_tower_lr</code> 的独立参数组被嵌套在
<code>mm_projector_lr is not None</code> 分支内。官方 System 2 脚本只传了：

~~~text
learning_rate=2e-5
vision_tower_lr=5e-6
mm_projector_lr=None
~~~

所以按当前代码实际会落入普通参数分组，vision tower 使用全局 <code>2e-5</code>，声明的
<code>5e-6</code> 不生效。本文只记录真实行为，没有擅自修复它。

### 9.3 分布式配置

两个官方脚本都按 Slurm 集群编写：8 节点 x 每节点 8 GPU，共 64 个进程。

~~~text
per_device_train_batch_size = 2
gradient_accumulation_steps = 1
world_size = 64
global batch = 2 x 1 x 64 = 128
~~~

DeepSpeed 配置是 [zero2.json](../../scripts/train/qwenvl_train/zero2.json)：ZeRO Stage 2、BF16。
这和先前双卡评测中的 <code>torchrun</code> 数据并行不是同一次运行；训练脚本的资源规模也不能直接
照搬到当前单台共享开发机。

### 9.4 保存和恢复

- <code>eval_strategy="no"</code>：训练中不运行验证集。
- 每 5000 step 保存一次，最多保留 5 份。
- 输出目录只要存在任意 <code>checkpoint-*</code>，就自动
  <code>resume_from_checkpoint=True</code>。
- 训练结束保存 Trainer state、image processor 和模型。

## 10. 参数如何对应到算法

| 参数 | 影响位置 | 实际含义 |
|---|---|---|
| <code>vln_dataset_use</code> | dataset init | 逗号分隔的 LeRobot 训练配置名 |
| <code>%30</code> | <code>parse_sampling_rate()</code> | 对该配置随机抽样 30% |
| <code>sample_step=4</code> | episode 拆样本 | 每隔 4 帧构造一个决策样本 |
| <code>num_history=8</code> | prompt | 最多选择 8 张历史前视图 |
| <code>num_future_steps=4</code> | turn sample | 最多聚合 4 个未来离散转向动作 |
| <code>predict_step_num=32</code> | trajectory GT | 每条连续轨迹固定 32 step |
| <code>n_query=4</code> | latent plan | 4 个 trajectory query；collator 目前硬编码为 4 |
| <code>pixel_goal_only=False</code> | Stage A | waypoint + turn + STOP |
| <code>pixel_goal_only=True</code> | Stage B | 仅 waypoint + trajectory GT |
| <code>system1=none</code> | Stage A | 使用原生 Qwen token CE |
| <code>system1=nextdit_async</code> | Stage B | RGB memory + NextDiT flow loss |
| <code>resize_h/w=384</code> | 前视 S2 图像增强 | 不是 System 1 的 224x224 RGB 尺寸 |
| <code>max_pixels/min_pixels</code> | Qwen image processor | 控制动态视觉 token 数 |

数据抽样发生在 dataset 初始化期间，且使用 Python <code>random.sample()</code>；官方代码在这一位置
之前没有显式固定 Python seed。再加上随机 prompt conjunction 和图像增强，直接重跑不保证抽到完全
相同的样本。

## 11. 推荐的源码阅读顺序

### 11.1 第一遍：只追 System 2

1. 在 [train_system2.sh](../../scripts/train/qwenvl_train/train_system2.sh) 看参数。
2. 在 [internvla_n1_trainer.py](../../internnav/trainer/internvla_n1_trainer.py) 找
   <code>Qwen2_5_VLForConditionalGeneration</code> 分支。
3. 在 dataset 的 <code>NavPixelGoalDataset.__init__()</code> 看三类样本怎样加入列表。
4. 在 <code>__getitem__()</code> 看 prompt 和 target 字符串。
5. 在 <code>preprocess_qwen_2_visual()</code> 看 <code>-100</code> mask。
6. 在 <code>DataCollatorForSupervisedDataset.__call__()</code> 看 pad/concat。
7. 最后看安装包 Qwen forward 的 shift CE。

第一遍先忽略 <code>traj_images</code>、query 和 NextDiT。

### 11.2 第二遍：只追 System 1

1. 在 [train_dual_system.sh](../../scripts/train/qwenvl_train/train_dual_system.sh) 看冻结和
   <code>pixel_goal_only=True</code>。
2. 在 <code>set_model()</code> 看哪些参数重新打开梯度。
3. 在 <code>initialize_vision_modules()</code> 看 System 1 如何创建。
4. 在 dataset 的位姿函数看 <code>[32,3]</code> 真值怎样生成。
5. 在 collator 看 4 个 query 和 <code>[B,F,32,3]</code> batch。
6. 在 custom forward 看 query hidden、RGB memory、带噪轨迹和 MSE。
7. 最后看 [generate_traj()](../../internnav/model/basemodel/internvla_n1/internvla_n1.py)，理解训练的
   velocity 网络怎样在推理时通过 10 步 scheduler 产生候选轨迹。

## 12. 当前机器上“能读”和“能直接训练”的差别

目前代码、最终 DualVLN 权重和完整共享训练数据都能读取，但官方训练 shell **不能原样直接启动**：

| 检查项 | 当前状态 |
|---|---|
| 最终 <code>InternVLA-N1-DualVLN</code> checkpoint | 已有，可做推理/评测 |
| Stage B 所需 <code>checkpoints/InternVLA-N1-System2</code> | 当前仓库缺少 |
| <code>data/vln_ce/raw_data/r2r</code> | 已接好，但只用于 Habitat 评测 |
| 代码预期 <code>traj_data/r2r\|rxr\|scalevln</code> | 仓库根下缺少相对路径 |
| 共享盘 LeRobot 训练数据本体 | 已有，见第 1 节链接 |
| 当前 <code>.venv</code> 的数据依赖 | 已有 <code>pyarrow 18.1.0</code>，本页检查脚本可读 Parquet；<code>pandas/decord/torchcodec</code> 仍缺，完整训练 dataset 尚不能直接导入 |
| Slurm <code>srun</code> 8x8 环境 | 当前单机环境不能照搬 |

所以本文的状态是“训练代码已完整整理并可阅读”，不是“训练环境已经安装并跑通”。本次没有创建数据
软链接、没有安装训练依赖，也没有启动大规模训练。

另一个易混淆文件
[train_system2_vlln.sh](../../scripts/train/qwenvl_train/train_system2_vlln.sh) 面向 VL-LN/IIGN，
它引用的 <code>internvla_vlln_trainer.py</code> 当前仓库不存在，不是本次 DualVLN 阅读主线。

## 13. VLN-CE、VLN-PE 和自采数据怎样对应

### 13.1 本文主线：VLN-CE DualVLN

本文解释的是 <code>vln_ce/traj_data/{r2r,rxr,scalevln}</code>。它的 rich schema 同时提供多视角
RGB/depth、4x4 pose、pixel goal、relative goal frame 和离散动作，因此可以同时监督 System 2
与 System 1。

### 13.2 VLN-PE 的 CMA/RDP 是另一条训练链

[scripts/train/base_train](../../scripts/train/base_train) 服务 Seq2Seq、CMA、RDP、NavDP 等基线，
不要与 <code>qwenvl_train</code> 混在一起读。

- CMA trainer 通常把离散 action CE 与 progress MSE 组合，见
  [cma_trainer.py](../../internnav/trainer/cma_trainer.py)。
- RDP trainer 使用连续轨迹噪声预测及 progress/stop-progress 监督，见
  [rdp_trainer.py](../../internnav/trainer/rdp_trainer.py)。
- 本机 VLN-PE 数据位于
  <code>/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_pe</code>。

这些 baseline 的输入 schema 和 loss 都不是本文第 6～8 节的 DualVLN 两阶段 loss。

### 13.3 自采 PointNav/ObjectNav 不能只改一个路径就训练 DualVLN

若自采数据要进入当前 <code>NavPixelGoalDataset</code>，至少需要生成等价字段：

~~~text
每个 episode 的 instruction/task text
同步的前视与朝下 RGB
朝下 depth（当前 nextdit_async 不用，但 schema/collator 会读取）
逐帧 4x4 camera pose
逐帧 pixel goal (u,v)
relative_goal_frame_id
离散 turn / stop action
LeRobot meta + parquet + videos 目录
~~~

现有 [vlnce2lerobot.py](../../scripts/dataset_converters/vlnce2lerobot.py) 的公开转换路径主要写
RGB + action，不能自动补出 pixel goal、relative goal frame 和 4x4 pose，因此不足以直接满足
DualVLN loader。

PointNav 只有目标坐标、ObjectNav 只有目标类别时，还需要先定义怎样形成语言 prompt、怎样选择可见
pixel waypoint，并通过规划器或专家轨迹生成逐帧监督。标准 Habitat episode JSON 本身同样不等价于
训练 demonstration。

## 14. 最容易误解的十件事

1. raw R2R episode 是评测题目，不是 InternVLA 的逐帧训练数据。
2. <code>"258 332"</code> 是按 <code>u v = x y</code> 顺序输出的文本 token，不是二维回归输出。
3. Stage A 不训练连续轨迹；Stage B 不计算 token CE。
4. 4 个 trajectory query 不等于 4 个动作。
5. 轨迹长度的 32、Q-Former memory token 的 32、推理候选轨迹数的 32 是三件事。
6. <code>noise_pred</code> 实际预测 Flow Matching velocity <code>epsilon-x0</code>。
7. <code>loss_mask</code> 只屏蔽补齐的 anchor，不屏蔽 32-step 轨迹的补零尾部。
8. <code>nextdit_async</code> 加载 DepthAnything 权重，但此分支实际只使用其 RGB backbone。
9. <code>%30</code> 是随机数据抽样比例，不是相机角度或 loss 权重。
10. <code>vision_tower_lr=5e-6</code> 按当前优化器分支实际不生效。

## 15. 训练输出和推理输出不要混为一谈

| 阶段 | forward 的主要输出 | loss 用什么 | 推理时看到什么 |
|---|---|---|---|
| System 2 | <code>logits [B,L,V]</code> | assistant token CE | 字符串 STOP、箭头或 <code>u v</code> |
| System 1 训练 | <code>noise_pred [B*F,32,3]</code> | velocity MSE | 训练时不会直接输出最终可执行动作 |
| System 1 推理 | 32 条候选、每条 32x3 | 无 loss | scheduler 去噪后的局部连续轨迹 |
| Habitat agent | 离散 action 序列 | 无训练 loss | STOP/FORWARD/LEFT/RIGHT 等环境动作 |

推理的完整调用、双卡 episode 分片和最终 NE/OS/SR/SPL 计算已经单独记录在
[R2R DualVLN 复现文档](../r2r_dualvln_reproduction/README.md)，可以在读完本文后继续阅读。

## 16. 一句话总结

DualVLN 先把 VLN 学成一个“看历史图像、用文本说出下一步高层 waypoint/符号动作”的 Qwen 模型，
再冻结它，用四个 latent query 抽取高层计划，训练一个以 RGB memory 为条件、预测 32-step 局部运动
flow velocity 的 NextDiT。前者用 token CE，后者用 masked trajectory MSE；二者是两个训练阶段，
不是一个联合加权 loss。
