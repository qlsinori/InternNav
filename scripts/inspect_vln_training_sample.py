#!/usr/bin/env python3
"""Export one real VLN training sample into files that VS Code can open.

This is deliberately a data-only inspector: it does not import InternNav's
dataset module, instantiate Qwen, start Habitat, or allocate a GPU.  The
trajectory helper functions below mirror the preprocessing in
``internnav/dataset/internvla_n1_lerobot_dataset.py`` so that the derived
``[F, 32, 3]`` target can be inspected without the full training stack.

Default example:

    .venv/bin/python scripts/inspect_vln_training_sample.py

The default is the real R2R scene ``17DRP5sb8fy``, LeRobot episode 0, frame 4,
and camera configuration ``r2r_125cm_0_30`` available on this machine.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline

try:
    import pyarrow
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pyarrow is required to inspect the original Parquet file.\n"
        "Install the project-declared dependency with:\n"
        "  uv pip install --python .venv/bin/python pyarrow==21.0.0"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_R2R = REPO_ROOT / "data/vln_ce/raw_data/r2r/train/train.json.gz"
DEFAULT_TRAJ_ROOT = Path("/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/traj_data/r2r")
ACTION_TEXT = {-1: "START/INVALID", 0: "STOP", 1: "↑", 2: "←", 3: "→", 5: "↓"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a real raw-R2R -> LeRobot -> DualVLN decision sample without loading a model."
    )
    parser.add_argument("--raw-r2r", type=Path, default=DEFAULT_RAW_R2R)
    parser.add_argument("--traj-root", type=Path, default=DEFAULT_TRAJ_ROOT)
    parser.add_argument("--scene", default="17DRP5sb8fy")
    parser.add_argument("--episode", type=int, default=0, help="Scene-local LeRobot episode_index.")
    parser.add_argument("--frame", type=int, default=4, help="Decision frame to explain and export.")
    parser.add_argument("--front-setting", default="125cm_0deg")
    parser.add_argument("--lookdown-setting", default="125cm_30deg")
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--num-future-steps", type=int, default=4)
    parser.add_argument("--predict-steps", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to logs/vln_data_inspection/<scene>_ep..._frame....",
    )
    parser.add_argument("--no-images", action="store_true", help="Skip image copies and previews.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_record(path: Path, key: str, value: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get(key) == value:
                return record
    raise ValueError(f"No record with {key}={value} in {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def schema_example(value: Any) -> str:
    """Format one Parquet value compactly for the annotated schema report."""

    if isinstance(value, list) and len(value) == 4 and all(isinstance(row, list) for row in value):
        translation = [round(float(value[row][3]), 6) for row in range(3)]
        return f"4×4 矩阵；平移列前三项={translation}（完整矩阵见 parquet_frame_*.json）"
    rendered = json.dumps(value, ensure_ascii=False)
    if len(rendered) > 180:
        return rendered[:177] + "..."
    return rendered


def schema_column_comment(
    name: str, value: Any, selected_setting: str, frame_index: int
) -> tuple[str, str, str]:
    """Return logical shape, Chinese semantics, and current-loader use for one column."""

    if name == "action":
        return (
            "标量（meta/info.json 逻辑声明为 [1]）",
            "到达本行 observation 时记录的原始离散动作。本 episode 实际值为 -1=START/INVALID、"
            "1=↑/forward、2=←/left、3=→/right；loader 在左移后给末尾人工补 0=STOP。代码的"
            "对话映射还定义 5=↓/look-down，但它是模型的两轮输出符号，不是本 episode 存储的基础动作。"
            "训练 loader 做 actions[1:] + [0]，所以决策帧 t 的“下一动作”来自原始行 t+1。",
            "使用；主要构造转向/STOP 样本。若本帧有有效 pixel goal，System 2 监督仍是 ↓ 后接 u v，"
            "不是这一个 action 数字。",
        )

    for prefix in ("pose.", "goal.", "relative_goal_frame_id."):
        if name.startswith(prefix):
            setting = name[len(prefix) :]
            is_selected = setting == selected_setting
            if prefix == "pose.":
                meaning = (
                    f"相机配置 {setting} 的 4×4 float32 变换矩阵；平移量以米计。训练代码把未来一段 "
                    "pose 交给 get_trajectory_relative_to_frame()，转成各 anchor 局部坐标中的轨迹。"
                    "源码注释称它为 T_world2camera；仅凭 Parquet schema 不应继续推断矩阵方向。"
                )
                use = (
                    "使用：为 System 1 派生 [F,32,3] Flow 轨迹 GT。"
                    if is_selected
                    else f"本配置不使用；选择 look-down setting={setting} 的配置时才使用。"
                )
                return "[4,4]（Arrow 本身是可变长嵌套 list）", meaning, use
            if prefix == "goal.":
                meaning = (
                    f"相机配置 {setting} 中的局部 waypoint 像素坐标 [u,v]：u 是横向列/x（0～639），"
                    "v 是纵向行/y（0～479）；[-1,-1] 表示该设置下没有有效可见 waypoint。"
                )
                use = (
                    "使用：直接成为 System 2 在 ↓ 后生成的文本 `u v`，按 token CE 训练，不是二维回归。"
                    if is_selected
                    else f"本配置不使用；选择 look-down setting={setting} 的配置时才使用。"
                )
                return "[2]", meaning, use

            meaning = (
                f"相机配置 {setting} 的 waypoint 未来帧偏移 h，不是绝对 frame ID。h=-1 表示无有效 "
                f"pixel goal；有效时当前帧 {frame_index} 使用 pose[t:t+h+1]，h<3 的样本会被丢弃。"
            )
            use = (
                "使用：决定样本类型、pose 切片终点和 System 1 anchor 数量；它本身不是回归标签。"
                if is_selected
                else f"本配置不使用；选择 look-down setting={setting} 的配置时才使用。"
            )
            return "标量（未来 frame offset）", meaning, use

    index_comments = {
        "timestamp": (
            "标量，秒",
            "episode 内时间戳，通常等于 frame_index / FPS；本例 FPS=30。",
        ),
        "frame_index": ("标量", "episode 内从 0 开始的帧号。"),
        "episode_index": (
            "标量",
            "当前 scene 的 LeRobot episode 编号；它不等于 raw R2R episode_id。",
        ),
        "index": ("标量", "该 scene 级 LeRobot 数据集内跨 episode 累计的全局行号。"),
        "task_index": (
            "标量",
            "指向 meta/tasks.jsonl 的 task_index，用来把整数还原为任务文本。",
        ),
    }
    if name in index_comments:
        shape, meaning = index_comments[name]
        return (
            shape,
            meaning,
            "当前 InternVLA loader 不使用；instruction/episode/length 来自 meta/episodes.jsonl，"
            "帧顺序直接采用 Parquet 行顺序。",
        )

    if name.startswith("local_point_goal_prompt."):
        setting = name.removeprefix("local_point_goal_prompt.")
        return (
            "字符串标量",
            f"额外 enrichment：把 {setting} 的 local BEV target 和允许动作写进自然语言模板。",
            "当前 InternVLA loader 完全不读取；它使用 episode instruction 在 __getitem__() 中另建 prompt。",
        )
    if name.startswith("local_point_goal."):
        setting = name.removeprefix("local_point_goal.")
        return (
            "[2]，float64",
            f"额外 enrichment：从当前机器人坐标看 {setting} 局部 waypoint 的 BEV [x,y]，单位米；"
            "对应 prompt 把当前位置写成 (0,0)、heading 写成 (1,0)。",
            "当前 InternVLA loader 完全不读取；检查脚本只用它核对由 pose 派生的轨迹位移。",
        )
    if name.startswith("final_point_goal."):
        setting = name.removeprefix("final_point_goal.")
        return (
            "[2]，float64",
            f"额外 enrichment：从当前机器人坐标看 episode 最后一帧在 {setting} 下的 BEV [x,y]，单位米；"
            "它不是五个不同的 Habitat goal。",
            "当前 InternVLA loader 完全不读取。",
        )
    if name.startswith("goal_prompt_pixel."):
        setting = name.removeprefix("goal_prompt_pixel.")
        return (
            "字符串标量",
            f"额外 enrichment：{setting} 的简短 prompt 前缀，写入 current/local/final BEV point。",
            "当前 InternVLA loader 完全不读取。",
        )
    if name.startswith("goal_prompt."):
        setting = name.removeprefix("goal_prompt.")
        return (
            "字符串标量",
            f"额外 enrichment：{setting} 的完整两阶段模板，先选符号动作，若选择 ↓ 再输出 u v。",
            "当前 InternVLA loader 完全不读取；正式训练 prompt 由代码重新构造。",
        )
    if name.startswith("action_pixel_goal_type."):
        setting = name.removeprefix("action_pixel_goal_type.")
        return (
            "字符串标量",
            f"额外 enrichment：描述 action_pixel_goal.{setting} 的类别，真实数据可见 "
            "forward/left/right/invalid。仓库缺少生成这 35 个 enrichment 字段的原脚本，故这里不臆测更多规则。",
            "当前 InternVLA loader 完全不读取。",
        )
    if name.startswith("action_pixel_goal."):
        setting = name.removeprefix("action_pixel_goal.")
        return (
            "[2]，int64",
            f"额外 enrichment：{setting} 的 action-conditioned 像素目标。实测 forward 且 waypoint 有效时"
            "常等于 goal，left/right 常取图像左右边缘，invalid 为 [-1,-1]；它与 goal 不是全局同义字段。",
            "当前 InternVLA loader 完全不读取，不能拿它替代 goal.<setting>。",
        )

    return "见 Arrow 类型", "本仓库没有此列的专门语义注释。", "当前 loader 未显式索引此列。"


def markdown_cell(value: Any) -> str:
    """Escape a value for a GitHub-Flavored Markdown table cell."""

    return str(value).replace("|", "\\|").replace("\n", "<br>")


def write_parquet_schema_markdown(
    path: Path,
    raw_schema_path: Path,
    table: Any,
    rows: list[dict[str, Any]],
    selected_row: dict[str, Any],
    parquet_path: Path,
    info_feature_names: set[str],
    front_setting: str,
    selected_setting: str,
    instruction: str,
) -> None:
    """Write a beginner-first guide plus a complete 56-column reference."""

    instruction = instruction.strip()
    field_names = table.schema.names
    field_numbers = {name: index for index, name in enumerate(field_names, start=1)}
    frame_index = int(selected_row["frame_index"])
    next_action = rows[frame_index + 1]["action"] if frame_index + 1 < len(rows) else 0
    pose_name = f"pose.{selected_setting}"
    goal_name = f"goal.{selected_setting}"
    horizon_name = f"relative_goal_frame_id.{selected_setting}"
    goal_uv = selected_row[goal_name]
    horizon = int(selected_row[horizon_name])

    def field_label(name: str) -> str:
        return f"[{field_numbers[name]:02d}] `{name}`"

    def value_summary(name: str) -> str:
        value = selected_row[name]
        if isinstance(value, str) and len(value) > 60:
            return f"字符串，共 {len(value)} 字符；全文见 frame JSON"
        return f"`{schema_example(value)}`"

    def add_header(lines: list[str], columns: list[str]) -> None:
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("|" + "|".join("---" for _ in columns) + "|")

    def add_row(lines: list[str], values: list[Any]) -> None:
        lines.append("| " + " | ".join(markdown_cell(value) for value in values) + " |")

    instruction_explanation = (
        "离开卧室，进入卫生间，并在马桶旁停下"
        if instruction.strip() == "Exit the bedroom, enter the bathroom, wait at the toilet."
        else "按照这句自然语言描述，从起点走到目的地"
    )

    lines = [
        "# VLN 小白版：用一个真实 frame 看懂相机 setting、数据和标签",
        "",
        f"> 这份 Markdown 由 `scripts/inspect_vln_training_sample.py` 根据真实 Parquet 和 frame {frame_index} 生成。",
        "> 它是教学文档，不是训练输入。前半部分按小白顺序讲故事；56 列技术表放在最后，第一次阅读不用看。",
        "",
        "**第一次阅读建议：读到“7. 到这里你应该懂什么”就停。** 后面只是遇到字段时再查的字典。",
        "",
        "## 1. 先把这条数据想成机器人做一道题",
        "",
        "人先给机器人一句路线指令：",
        "",
        f"> `{instruction}`",
        "",
        f"它的大意是：**{instruction_explanation}**。机器人从开始走到最后停下的完整过程叫一个",
        f"**episode（一次完整导航）**。本例 episode 有 **{len(rows)} 个时刻**，编号为 frame 0～{len(rows) - 1}。",
        "",
        f"这里挑出的 **frame {frame_index}** 是从 0 开始数的第 **{frame_index + 1} 张**观察，时间约为",
        f"`{frame_index}/30 = {frame_index / 30:.3f}` 秒。它不是“第 4 个 episode”，也不是模型标签。",
        "",
        "```text",
        "一整道导航题（episode）",
        f"├── 路线指令：{instruction}",
        f"├── frame 0：第 1 个时刻",
        "├── frame 1：第 2 个时刻",
        "├── ...",
        f"├── frame {frame_index}：本页研究的当前时刻",
        "├── ...",
        f"└── frame {len(rows) - 1}：最后一个时刻",
        "```",
        "",
        "一条 episode 会被训练代码拆成很多道小题。每道小题都在问：**机器人看到当前画面后，",
        "接下来该往哪里走？** frame 4 就是其中一道小题的起点。",
        "",
        "## 2. `camera setting` 到底是什么",
        "",
        "`camera setting` 可以直译成**相机拍摄设置**。这里它只描述两件事：",
        "",
        "```text",
        "125cm_30deg",
        "│     └──── 镜头向下俯视 30 度（deg = degree，度）",
        "└────────── 镜头离地 125 厘米",
        "```",
        "",
        f"所以 `{front_setting}` 是镜头离地 125 厘米、基本水平看前方；",
        f"`{selected_setting}` 是镜头同样离地 125 厘米、但向下俯视 30 度。",
        "它们**不是帧编号，不是机器人转了 30 度，也不是学习率一类的训练超参数**。",
        "在 Habitat 这类模拟数据里，可以把它理解为：在同一个机器人位置，用几台安装姿态不同的",
        "虚拟相机各拍一张照片。",
        "",
        f"### 同一个 frame {frame_index}，为什么有好几张图",
        "",
        f"下面两张照片来自完全相同的时刻和机器人位置，只是镜头朝向不同。点击图片可打开原图。",
        "",
        f"| `{front_setting}`：水平前视 | `{selected_setting}`：向下 30° |",
        "|---|---|",
        f"| [![frame {frame_index} 前视](media_rgb_{front_setting}_frame{frame_index:06d}.jpg)](media_rgb_{front_setting}_frame{frame_index:06d}.jpg) | [![frame {frame_index} 朝下](media_rgb_{selected_setting}_frame{frame_index:06d}.jpg)](media_rgb_{selected_setting}_frame{frame_index:06d}.jpg) |",
        "| 看房间、门、走廊，帮助理解语言指令 | 看近处地面，方便指出下一处落脚点 |",
        "",
        "前视和朝下的分工是：",
        "",
        "- **front（前视）**：回答“我在哪里，门和卫生间大概在哪边”。本样本给 System 2 看",
        f"  历史前视 frame 0～{frame_index - 1}，再看当前 frame {frame_index}。",
        "- **look-down（朝下）**：回答“眼下具体踩向地面的哪个点”。图中的目标点会写成像素坐标",
        f"  `{goal_uv[0]} {goal_uv[1]}`；这张图也作为 System 1 生成局部路线的视觉条件。",
        "- `↓` 在模型回答中表示“请切到朝下图继续定位”，**不是让机器人向下移动**。",
        "",
        "本 scene 实际保存 5 个 setting，每个 setting 都有一张彩色 RGB 图和一张 depth 距离图，",
        "因此同一 frame 共对应 10 个媒体流：",
        "",
        f"| setting | 白话解释 | 同一 frame {frame_index} 的 RGB 原图 | 本配置是否用 |",
        "|---|---|---|---|",
        f"| `125cm_0deg` | 高 125 cm，水平前视 | [打开](media_rgb_125cm_0deg_frame{frame_index:06d}.jpg) | **用作 front** |",
        f"| `125cm_30deg` | 高 125 cm，向下 30° | [打开](media_rgb_125cm_30deg_frame{frame_index:06d}.jpg) | **用作 look-down** |",
        f"| `125cm_45deg` | 高 125 cm，向下 45° | [打开](media_rgb_125cm_45deg_frame{frame_index:06d}.jpg) | 本配置不用 |",
        f"| `60cm_15deg` | 高 60 cm，向下 15° | [打开](media_rgb_60cm_15deg_frame{frame_index:06d}.jpg) | 本配置不用 |",
        f"| `60cm_30deg` | 高 60 cm，向下 30° | [打开](media_rgb_60cm_30deg_frame{frame_index:06d}.jpg) | 本配置不用 |",
        "",
        "### setting 和 dataset config 不是同一个东西",
        "",
        "一个 `setting` 只代表一个镜头视角，例如 `125cm_30deg`。训练名称",
        "`r2r_125cm_0_30` 则代表一整套数据搭配：",
        "",
        "```text",
        "r2r_125cm_0_30",
        "│   │     │  └── look-down 的俯角：30°",
        "│   │     └───── front 的俯角：0°",
        "│   └─────────── 相机高度：125 cm",
        "└─────────────── 数据集：R2R",
        "",
        "也就是：front = 125cm_0deg，look-down = 125cm_30deg",
        "```",
        "",
        "代码里的 `pitch_1` 就是 front 俯角，`pitch_2` 就是 look-down 俯角。",
        "如果训练脚本写成 `r2r_125cm_0_30%30`，最后的 `%30` 表示随机保留 30% 样本，",
        "与相机的 30° 无关。可直接查看",
        "[相机配置定义源码](../../../../internnav/dataset/internvla_n1_lerobot_dataset.py) 和",
        "[Dual System 训练脚本](../../../../scripts/train/qwenvl_train/train_dual_system.sh)。",
        "",
        "```mermaid",
        "flowchart LR",
        f"    T[\"同一时刻 frame {frame_index}<br/>机器人位置没变\"] --> F[\"{front_setting}<br/>水平前视 RGB\"]",
        f"    T --> L[\"{selected_setting}<br/>朝下 RGB\"]",
        f"    T --> D[\"{selected_setting}<br/>朝下 depth\"]",
        "    F --> Q[\"System 2<br/>理解场景和指令\"]",
        f"    L --> P[\"指出地面目标点<br/>{goal_uv[0]} {goal_uv[1]}\"]",
        "    L --> S[\"System 1<br/>生成局部路线\"]",
        "    D -.-> N[\"loader 会读取；<br/>当前 nextdit_async 分支不用于 loss\"]",
        "```",
        "",
        "一句话记忆：**`125cm_0deg/30deg` 说的是镜头怎么摆；`r2r_125cm_0_30` 说的是",
        "一次训练样本把哪一个前视镜头和哪一个朝下镜头配在一起。**",
        "",
        "## 3. 先认识这些词，再看字段",
        "",
        "| 词 | 完全不专业的解释 |",
        "|---|---|",
        "| VLN | Vision-Language Navigation：机器人一边看图，一边照着人话找路 |",
        "| instruction | 人给机器人的整句路线指令，不是分类标签 |",
        "| episode | 从起点执行一条 instruction，直到结束的一整次导航 |",
        "| frame / observation | episode 某一时刻机器人看到的图和当时记录的数据 |",
        "| decision sample | 训练代码从某个 frame 加工出的一道“下一步怎么走”小题 |",
        "| camera setting | 相机离地高度和俯视角度，例如 `125cm_30deg` |",
        "| waypoint | 不是最终终点，而是眼下先走到的一个中途落脚点 |",
        "| label / GT | 标准答案；训练时拿模型答案和它比较 |",
        "| loss | 模型答案错了多少的分数；训练就是设法让这个数变小 |",
        "| loader / Dataset | 从硬盘读取数据并把它加工成模型输入与标准答案的代码 |",
        "| Parquet | 一种二进制表格；本例一行对应一个 frame，类似程序高效读取的 Excel |",
        "| pose | 相机当时的位置和朝向，可先把它理解成更完整的“GPS + 指南针” |",
        "| pixel / `[u,v]` | 图像中的一个小点；`u` 从左向右，`v` 从上向下 |",
        "| depth | 每个像素离相机多远的距离图，不是普通彩色照片 |",
        "| token | 模型读写文本时用的基本小块；数字和箭头最终也会变成 token |",
        "| CE / MSE | 两种比较“模型答案”和“标准答案”的算法；前者适合文字，后者适合连续数值 |",
        "| BEV | bird's-eye view，像从天花板向下看的一张局部平面地图 |",
        "| Arrow type | Parquet 在磁盘里怎样保存数字；只查数据格式时才需要看 |",
        "| enrichment | 后处理额外加进表里的辅助字段；当前训练不一定使用 |",
        "",
        "## 4. 一道训练小题的数据分别放在哪里",
        "",
        "不要误以为 56 列包含了所有内容。instruction、图片和数字表分别住在不同文件里：",
        "",
        "```text",
        "episode 0 / frame 4 这道小题",
        "├── 人说了什么",
        "│   └── meta/episodes.jsonl：instruction + episode 有多少帧",
        "├── 机器人看到了什么",
        "│   └── videos/...：不同 camera setting 的 RGB / depth 图片",
        "└── 这一刻的位置和标准答案线索",
        "    └── data/.../episode_000000.parquet：一帧一行的数字表",
        "```",
        "",
        "本页已把不方便直接看的容器抽成普通文件：",
        "",
        f"- [本 episode 的 instruction 和长度](lerobot_episode_000000.json)；",
        f"- [frame {frame_index} 的全部 56 列真实值](parquet_frame_{frame_index:06d}.json)；",
        f"- [同一 frame 的前视 RGB](media_rgb_{front_setting}_frame{frame_index:06d}.jpg)；",
        f"- [同一 frame 的朝下 RGB](media_rgb_{selected_setting}_frame{frame_index:06d}.jpg)；",
        f"- [在朝下图上画出 `{goal_uv}` 的版本](lookdown_goal_overlay.jpg)；",
        "- [把三种关键图放在一起的总览](overview.jpg)。",
        "",
        "## 5. 56 列先别背：当前配置真正取标准答案只看四列",
        "",
        "训练代码虽然把整张 56 列表读进内存，但当前 camera config 后续真正索引的只有：",
        "",
        "```text",
        "action                              到达当前画面前做了什么动作",
        f"{pose_name:<36} 相机在哪里、朝哪里",
        f"{goal_name:<36} 朝下图里的下一落脚点",
        f"{horizon_name:<36} 还要经过多少帧到这个落脚点",
        "```",
        "",
        "### 5.1 `action`：动作记录",
        "",
        f"frame {frame_index} 这一行的 `action={selected_row['action']}` 表示机器人**为了到达这张画面**执行的是前进。",
        f"训练要问的是“看到这张画面后做什么”，所以代码把动作向前挪一格，读取下一行的",
        f"`action={next_action}`，也就是右转。这个动作挪位叫 action shift（动作左移）。",
        "",
        "不过当前 frame 有一个有效 waypoint，因此 System 2 不把右转箭头当最终文字答案；它改学",
        f"“请求朝下图，然后指出 `{goal_uv[0]} {goal_uv[1]}`”。只有没有有效 waypoint 的转向样本才直接学箭头。",
        "",
        f"### 5.2 `{goal_name}`：朝下图中的目标点",
        "",
        f"真实值是 `{goal_uv}`，顺序为 `[u,v]`：",
        "",
        "```text",
        "(0,0) ───────────── u / 横向 / 向右 ────────────> (639,0)",
        "  │",
        "  │",
        "  v  v / 纵向 / 向下",
        "  │",
        "(0,479)",
        "",
        f"本例 u={goal_uv[0]}：从左边向右数 {goal_uv[0]} 个像素",
        f"本例 v={goal_uv[1]}：从上边向下数 {goal_uv[1]} 个像素",
        "```",
        "",
        f"请直接打开 [带十字标记的真实朝下图](lookdown_goal_overlay.jpg)。这个点是**局部 waypoint**，",
        "不是卫生间马桶这个最终任务终点，也不是以米为单位的坐标。",
        "",
        f"### 5.3 `{horizon_name}`：多久到 waypoint",
        "",
        f"真实值 `{horizon}` 的意思是：从当前 frame {frame_index} 再向未来走 `{horizon}` 帧，会到 frame",
        f"`{frame_index + horizon}`。所以代码取 frame {frame_index}～{frame_index + horizon} 的 pose，Python 切片写作",
        f"`[{frame_index}:{frame_index + horizon + 1}]`。它是“向未来数几格”，不是绝对 frame ID。",
        "",
        f"### 5.4 `{pose_name}`：机器人走过的空间位置",
        "",
        "每个 frame 的 pose 是一个 `4×4` 数字矩阵，用来同时记录三维位置和朝向。你现在不必手算矩阵；",
        "只需知道：连续看 frame 4～15 的 pose，就能还原机器人这段真实路线。训练代码再把这条路线",
        "变成“向前多少、向侧面多少、转头多少”的连续运动标准答案。",
        "",
        "## 6. 这四列最后怎样变成 Dual System 的两种答案",
        "",
        "可以把双系统记成：**System 2 是看图和读指令的“大脑”，System 1 是把意图变成平滑路线的‘腿’。**",
        "",
        "### System 2：学会用文字回答下一步",
        "",
        f"1. 它先看历史前视 frame 0～{frame_index - 1}、当前前视 frame {frame_index} 和 instruction。",
        "2. 标准答案先是 `↓`，意思是“我需要朝下图来精确找落脚点”。",
        f"3. 加入当前 `{selected_setting}` 朝下图后，标准答案是文本 `{goal_uv[0]} {goal_uv[1]}`。",
        "4. 训练只检查模型自己回答的这些文字小块（token），不要求它复述 instruction，也不把图片当答案。",
        "5. CE loss 可以理解成：正确文字的概率越低，扣分越多。",
        "",
        "### System 1：学会画出连续局部路线",
        "",
        f"1. 从 pose 中取 frame {frame_index}～{frame_index + horizon} 的真实移动。",
        f"2. 每隔 2 帧选一个重新规划起点，本例为 `{list(range(frame_index, frame_index + horizon, 2))}`。这些起点叫 anchor。",
        "3. 从每个 anchor 往后，把路线整理成 32 个小步；每步 3 个数：向前/侧面位移和转角。",
        "4. 本例最终标准答案 shape 是 `[6,32,3]`：6 个 anchor × 32 个小步 × 每步 3 个数。",
        "5. 训练先给真实路线加噪声，再让 NextDiT 学会去掉噪声、恢复正确运动方向；MSE loss",
        "   可以理解成逐个数字比较预测与标准答案，差得越远扣分越多。",
        "",
        "这两种 loss 属于两个训练阶段：System 2 阶段学文本 CE，Dual/System 1 阶段学轨迹 Flow MSE；",
        "不是把两项随手加在同一次 forward 里。",
        "",
        "```text",
        "instruction + 前视图片",
        "          │",
        "          ▼",
        f"System 2 标准答案：↓  →  看朝下图  →  {goal_uv[0]} {goal_uv[1]}",
        "                                             │",
        "                                             ▼",
        "pose frame 4～15 ───────────────> System 1 标准答案 [6,32,3] 局部路线",
        "```",
        "",
        "## 7. 到这里你应该懂什么",
        "",
        "读完前面，只要能回答下面 6 句，就已经足够继续看训练代码：",
        "",
        "1. 一个 episode 是一次完整导航；一个 frame 只是其中一个时刻。",
        "2. 同一 frame 可以从不同 camera setting 同时拍摄，机器人位置并没有变化。",
        f"3. 当前配置用 `{front_setting}` 看前方，用 `{selected_setting}` 看近处地面。",
        f"4. `[u,v]=[{goal_uv[0]},{goal_uv[1]}]` 是朝下图中的中途落脚点，不是最终终点。",
        f"5. `{horizon}` 表示从 frame {frame_index} 向未来数 {horizon} 帧，不是 frame ID。",
        "6. System 2 学文字/坐标，System 1 学连续路线。",
        "",
        "**第一次阅读可以在这里停下。** 下面是为了逐字段排查代码和数据而保留的技术字典。",
        "",
        "---",
        "",
        "## 8. 技术追踪：frame 4 的四列怎样变成监督",
        "",
        "这一节把上面的白话和源码术语一一对应。遇到 `loader`、`CE`、`GT` 时可回看第 3 节词表。",
        "",
        "可直接打开的相关文件：",
        "",
        f"- [frame {frame_index} 的全部 56 列实值](parquet_frame_{frame_index:06d}.json)",
        "- [episode 0 全部 46×56 数据](parquet_episode_000000_all_46x56.jsonl)",
        "- [46 帧教学表](frames.csv)",
        "- [meta/info.json](lerobot_info.json)",
        f"- [未经注释的原始 Arrow schema]({raw_schema_path.name})",
        "- [本样本和 loss 摘要](sample_summary.json)",
        "- [Dataset 读取和标签构造源码](../../../../internnav/dataset/internvla_n1_lerobot_dataset.py)",
        "- [推理时 `u v → [v,u]` 的源码](../../../../internnav/model/basemodel/internvla_n1/internvla_n1_policy.py)",
        "",
        "原始二进制文件位于：",
        "",
        f"```text\n{parquet_path.resolve()}\n```",
        "",
        f"文件有 **{table.num_columns} 个顶层列**。但是当前 `front={front_setting}, look-down={selected_setting}` "
        "配置实际用来构造训练样本的只有：",
        "",
        "```text",
        "action",
        pose_name,
        goal_name,
        horizon_name,
        "```",
        "",
        "当前 `pq.read_table()` 没有限制 `columns=`，所以 56 列仍会全部读进内存；“只用四列”表示",
        "后续训练逻辑只显式索引这四列。换一个 camera config 时，会改用对应 setting 的三列。",
        f"`meta/info.json` 声明了 {len(info_feature_names)} 列，实际 Parquet 另外包含 "
        f"{len(field_names) - len(info_feature_names)} 个后处理 enrichment 字段。",
        "",
    ]
    add_header(lines, ["字段", f"frame {frame_index} 实值", "处理", "最终作用"])
    add_row(
        lines,
        [
            "`action`",
            f"本行 `{selected_row['action']}`，下一行 `{next_action}`",
            "执行 `actions[1:] + [0]`，让当前观察对齐下一动作",
            "用于 turn/STOP 分支；本帧有有效 waypoint，所以最终输出不是这个箭头",
        ],
    )
    add_row(
        lines,
        [
            f"`{pose_name}`",
            "`[4,4]`",
            f"截取 frame {frame_index}～{frame_index + horizon}，转成 anchor 局部轨迹并重采样",
            "System 1 的 `[F,32,3]` Flow trajectory GT",
        ],
    )
    add_row(
        lines,
        [
            f"`{goal_name}`",
            f"`{goal_uv}`",
            "保持 `[u,v]` 顺序并转成文本",
            f"System 2 在 `↓` 后的 token CE 标签 `{goal_uv[0]} {goal_uv[1]}`",
        ],
    )
    add_row(
        lines,
        [
            f"`{horizon_name}`",
            f"`{horizon}`",
            f"解释为未来 offset；pose slice 为 `[{frame_index}:{frame_index + horizon + 1}]`",
            "决定样本有效性、轨迹窗口和 anchor 数；本身不是回归标签",
        ],
    )
    lines.extend(
        [
            "",
            f"因此本帧 System 2 的监督是：先生成 `↓`，看到朝下图后再生成 `{goal_uv[0]} {goal_uv[1]}`。",
            "",
            "## 9. 为什么会有 56 列",
            "",
            "同一条路线为 5 种 camera setting 都预先保存了 pose、goal 和 horizon；数据制作流程还追加了",
            "一些备用目标和 prompt。所以列数看起来很多，并不表示模型同时预测 56 个答案。当前配置只",
            "从所选 look-down setting 取 3 列，再加 `action`，其余字段第一次阅读都可以跳过。",
            "",
        ]
    )
    add_header(lines, ["组", "列号", "数量", "内容", "当前 loader"])
    for values in [
        ["A", "1", "1", "离散 action", "使用"],
        ["B", "2–16", "15", "5 × (`pose`, `goal`, `relative_goal_frame_id`)", "每个配置使用所选 setting 的 3 列"],
        ["C", "17–21", "5", "LeRobot 时间和索引", "不使用"],
        ["D", "22–31", "10", "5 × (`local_point_goal`, prompt)", "不使用"],
        ["E", "32–36", "5", "5 × `final_point_goal`", "不使用"],
        ["F", "37–41", "5", "5 × 完整 `goal_prompt`", "不使用"],
        ["G", "42–56", "15", "5 × (`action_pixel_goal`, type, prompt)", "不使用"],
        ["合计", "1–56", "56", "21 个基础字段 + 35 个 enrichment 字段", "当前配置语义使用 4 列"],
    ]:
        add_row(lines, values)

    lines.extend(
        [
            "",
            "## 附录 A：56 列逐列查字典",
            "",
            "下面保证 56 列一列不少，适合排查代码时搜索字段名。`Arrow 类型` 只是在说明硬盘如何保存",
            "数字；`当前 loader` 是在说明训练读取代码是否真正把这一列加工成输入或标签。无需通读。",
        ]
    )

    groups = [
        (
            "### A1. 离散动作（列 1）",
            0,
            1,
            "`action` 是到达当前 observation 时记录的动作；loader 左移后才得到当前观察对应的下一动作。",
        ),
        (
            "### A2. 五种相机设置的 pose / goal / horizon（列 2–16）",
            1,
            16,
            "五套 setting 属于同一条轨迹，不是五个 episode。每个配置只选择一套 look-down setting。",
        ),
        (
            "### A3. LeRobot 时间和索引（列 17–21）",
            16,
            21,
            "这些字段用于数据格式记账；当前 loader 的 instruction、episode ID 和 length 来自 `meta/episodes.jsonl`。",
        ),
        (
            "### A4. local point goal 与 prompt（列 22–31）",
            21,
            31,
            "`local_point_goal` 是机器人局部 BEV `[x,y]`，单位米；配套 prompt 把同一数值写入问句。当前 loader 不读取。",
        ),
        (
            "### A5. final point goal（列 32–36）",
            31,
            36,
            "表示从当前机器人坐标看 episode 最后一帧位置的 BEV `[x,y]`，单位米；不是五个不同 Habitat goal。",
        ),
        (
            "### A6. 完整两阶段 prompt（列 37–41）",
            36,
            41,
            "模板写入 local/final BEV point，并要求先选动作、若选 `↓` 再输出 `u v`。正式 loader 不读取这些字符串。",
        ),
        (
            "### A7. action-conditioned pixel goal（列 42–56）",
            41,
            56,
            "这是另一组 enrichment。它与正式使用的 `goal.<setting>` 不是全局同义字段，当前 loader 完全不读取。",
        ),
    ]
    covered_names: list[str] = []
    for title, start, end, introduction in groups:
        lines.extend(["", title, "", introduction, ""])
        add_header(
            lines,
            ["# / 字段", "磁盘类型 / 形状（可跳过）", f"frame {frame_index} 实值", "白话含义", "训练读取代码是否用"],
        )
        for name in field_names[start:end]:
            field = table.schema.field(name)
            logical_shape, meaning, loader_use = schema_column_comment(
                name, selected_row[name], selected_setting, frame_index
            )
            source = "meta/info.json 已声明" if name in info_feature_names else "Parquet enrichment；meta 未声明"
            add_row(
                lines,
                [
                    field_label(name),
                    f"`{field.type}`；{logical_shape}<br>来源：{source}",
                    value_summary(name),
                    meaning,
                    loader_use,
                ],
            )
            covered_names.append(name)

    representative_prompts = [
        f"local_point_goal_prompt.{selected_setting}",
        f"goal_prompt.{selected_setting}",
        f"goal_prompt_pixel.{selected_setting}",
    ]
    lines.extend(
        [
            "",
            "<details>",
            f"<summary>展开 frame {frame_index} / {selected_setting} 的三种真实 prompt 全文</summary>",
            "",
        ]
    )
    for name in representative_prompts:
        lines.extend(
            [
                f"#### {field_label(name)}",
                "",
                "```text",
                selected_row[name],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "</details>",
            "",
            "## 附录 B：最容易误解的地方",
            "",
            "1. **56 是顶层列数。** Raw schema 中缩进的 `child/element` 是 list 子类型，不是额外列。",
            "2. **Arrow list 不强制固定 shape。** `[4,4]` 和 `[2]` 是 meta 逻辑约定及真实数据形状。",
            "3. **`goal=[u,v]`。** `u` 是横向列/x，`v` 是纵向行/y，不是 `[row,column]`。",
            "4. **horizon 是未来 offset。** frame 4 的 11 对应 frame 15，pose slice 是 `[4:16]`。",
            "5. **action 会左移。** frame 4 原值为 1，而它的下一动作来自 frame 5 的 3。",
            "6. **有效 waypoint 优先形成坐标监督。** 所以本帧标签是 `↓`、`258 332`，不是 `→`。",
            "7. **`goal` 不等于 `action_pixel_goal`。** 转向帧中后者常是图像左右边缘点。",
            "8. **`goal=[-1,-1]` 只表示图像 waypoint 无效。** 不表示导航任务或 BEV goal 不存在。",
            "9. **`meta/info.json` 只声明前 21 列。** 完整 56 列应以 Parquet Arrow schema 为准。",
            "10. **后 35 列不进入当前 loss。** 当前 loader 的 prompt 是根据 episode instruction 重新构造的。",
            "11. **RGB/depth 不在 Parquet。** 它们位于 `videos/chunk-000/observation.images.*`。",
            "",
            "## 附录 C：后 35 列的来源边界",
            "",
            "共享盘中的下面这个 shell 记录了使用 `--write-parquet` 后处理 R2R/RxR/ScaleVLN：",
            "",
            "```text",
            "/mnt/cpfs/zbl-cpfs-new/open_data/InternData-N1/vln_ce/traj_data/data_convert.sh",
            "```",
            "",
            "它记录调用的自动标注脚本是：",
            "",
            "```text",
            "/x2robot_v2/morris/dualvln_post/scripts/dataset_converters/vln_data_auto_labeling_pipeline.py",
            "```",
            "",
            "这个私有 Python 文件当前不在原绝对路径，也不在公共 InternNav 仓库。因此 local/final 坐标可以用",
            "pose 数值复核，action pixel 的规律可以从 46 行数据观察，但自动选点阈值、可见性判断和投影实现",
            "无法进行源码级确认。",
            "",
            f"未经注释的类型定义保存在 [parquet_schema.txt]({raw_schema_path.name})。",
            "",
        ]
    )

    if covered_names != field_names:
        raise AssertionError("Markdown guide did not cover the Parquet columns exactly once and in order")
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def scene_name(scene_id: str) -> str:
    return Path(scene_id).stem


def find_raw_episode(raw_data: dict[str, Any], scene: str, instruction: str) -> tuple[int, dict[str, Any], int]:
    candidates: list[tuple[int, dict[str, Any]]] = []
    for array_index, episode in enumerate(raw_data["episodes"]):
        if scene_name(episode["scene_id"]) != scene:
            continue
        if episode["instruction"]["instruction_text"].strip() == instruction.strip():
            candidates.append((array_index, episode))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one raw R2R match for scene={scene!r} and instruction={instruction!r}, "
            f"found {len(candidates)}"
        )
    array_index, episode = candidates[0]
    scene_count = sum(scene_name(item["scene_id"]) == scene for item in raw_data["episodes"])
    return array_index, episode, scene_count


def classify_decision(
    rows: list[dict[str, Any]], frame: int, setting: str, num_future_steps: int
) -> dict[str, Any]:
    """Mirror the sample-selection branch in NavPixelGoalDataset.__init__."""

    goal_key = f"goal.{setting}"
    relative_key = f"relative_goal_frame_id.{setting}"
    raw_actions = [int(row["action"]) for row in rows]
    next_actions = raw_actions[1:] + [0]
    row = rows[frame]
    goal_uv = [int(value) for value in row[goal_key]]
    relative_goal_frame_id = int(row[relative_key])

    result: dict[str, Any] = {
        "frame": frame,
        "raw_action": raw_actions[frame],
        "raw_action_text": ACTION_TEXT.get(raw_actions[frame], str(raw_actions[frame])),
        "next_action_after_loader_shift": next_actions[frame],
        "next_action_text": ACTION_TEXT.get(next_actions[frame], str(next_actions[frame])),
        "goal_uv": goal_uv,
        "relative_goal_frame_id": relative_goal_frame_id,
    }

    if relative_goal_frame_id == -1:
        if next_actions[frame] == 1:
            result.update(target_type="discarded", reason="invalid pixel goal and next action is forward")
            return result
        turn_actions: list[int] = []
        for index in range(frame, min(len(rows), frame + num_future_steps)):
            if next_actions[index] == 1:
                break
            turn_actions.append(next_actions[index])
        result.update(
            target_type="turn",
            turn_actions=turn_actions,
            system2_target="".join(ACTION_TEXT[action] for action in turn_actions),
        )
        return result

    if relative_goal_frame_id < 3:
        result.update(target_type="discarded", reason="pixel goal horizon is shorter than 3 frames")
        return result

    # Parquet stores image coordinates as u v = horizontal x, vertical y.
    # The model learns exactly the same textual order.  Inference later swaps
    # it to internal [row, column] for downstream image indexing.
    result.update(
        target_type="pixel_goal",
        system2_targets=["↓", f"{goal_uv[0]} {goal_uv[1]}"],
        pose_slice=[frame, frame + relative_goal_frame_id + 1],
    )
    return result


def build_decision_table(
    rows: list[dict[str, Any]], setting: str, sample_step: int, num_future_steps: int
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    num_rounds = len(rows) // sample_step
    for round_index in range(num_rounds + 1):
        frame = round_index * sample_step
        if frame == len(rows) or frame == len(rows) - 1:
            continue
        decisions.append(classify_decision(rows, frame, setting, num_future_steps))

    # STOP is not read from the last Parquet action.  The loader creates this
    # item explicitly, then Stage A inserts it into the list five times.
    decisions.append(
        {
            "frame": len(rows) - 1,
            "target_type": "stop",
            "system2_target": "STOP",
            "stage_a_repetitions": 5,
            "note": "created by the loader; not taken from the final raw action",
        }
    )
    return decisions


def history_frames(frame: int, num_history: int) -> list[int]:
    if frame == 0:
        return []
    return np.unique(np.linspace(0, frame - 1, num_history, dtype=np.int32)).tolist()


def get_trajectory_relative_to_frame(extrinsics: np.ndarray, camera_deg: float) -> np.ndarray:
    """Mirror get_trajectory_relative_to_frame() from the training dataset."""

    robot_to_camera = np.array(
        [[[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    camera_rad = np.radians(camera_deg)
    pitch_transform = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.cos(-camera_rad), -np.sin(-camera_rad), 0.0],
                [0.0, np.sin(-camera_rad), np.cos(-camera_rad), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    robot_to_camera = np.matmul(robot_to_camera, pitch_transform)
    camera_to_robot = np.linalg.inv(robot_to_camera)
    extrinsics_robot = np.matmul(extrinsics, camera_to_robot)
    relative_to_ref = np.matmul(np.linalg.inv(extrinsics_robot[0])[None, :, :], extrinsics_robot)
    translations = relative_to_ref[:, :2, 3]
    yaws = np.arctan2(relative_to_ref[:, 1, 0], relative_to_ref[:, 0, 0])
    return np.concatenate((translations, yaws[:, None]), axis=-1)


def smooth_and_resample_trajectory(points: np.ndarray, sample_length: int = 33, interval: float = 0.1) -> np.ndarray:
    """Mirror the training implementation, including its sample_length*interval convention."""

    total_distance = sample_length * interval
    if len(points) == 0:
        return np.zeros((sample_length, 2))
    if len(points) == 1:
        return np.tile(points[0], (sample_length, 1))

    differences = np.diff(points, axis=0)
    cumulative = np.insert(np.cumsum(np.sqrt(np.sum(differences**2, axis=1))), 0, 0)
    if len(points) > 3:
        spline_x = CubicSpline(cumulative, points[:, 0])
        spline_y = CubicSpline(cumulative, points[:, 1])
        dense_distances = np.linspace(0, cumulative[-1], max(50, len(points) * 2))
        smoothed = np.column_stack((spline_x(dense_distances), spline_y(dense_distances)))
        smooth_diff = np.diff(smoothed, axis=0)
        smooth_cumulative = np.insert(np.cumsum(np.sqrt(np.sum(smooth_diff**2, axis=1))), 0, 0)
    else:
        smoothed = points
        smooth_cumulative = cumulative

    targets = np.linspace(0, total_distance, sample_length)
    resampled = np.zeros((sample_length, 2))
    for index, target in enumerate(targets):
        if target >= smooth_cumulative[-1]:
            resampled[index] = smoothed[-1]
            continue
        segment = np.searchsorted(smooth_cumulative, target, side="right") - 1
        start_distance = smooth_cumulative[segment]
        end_distance = smooth_cumulative[segment + 1]
        ratio = (target - start_distance) / (end_distance - start_distance)
        resampled[index] = smoothed[segment] + ratio * (smoothed[segment + 1] - smoothed[segment])
    return resampled


def xy_to_delta_xyt(xy_actions: np.ndarray) -> np.ndarray:
    vectors = np.diff(xy_actions, axis=0)
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])
    delta_yaw = (np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi
    delta_yaw = np.concatenate([[yaw[0]], delta_yaw])
    return np.concatenate([vectors, delta_yaw[:, None]], axis=1)


def interpolate_and_resample_trajectory(absolute_trajectory: np.ndarray, predict_steps: int) -> np.ndarray:
    trajectory_xy = absolute_trajectory[..., :2]
    steps = trajectory_xy[1:] - trajectory_xy[:-1]
    keep = (steps**2).sum(axis=-1) > 0.05
    filtered = np.concatenate([np.array([[0.0, 0.0]]), trajectory_xy[1:][keep]], axis=0)
    resampled = smooth_and_resample_trajectory(filtered, sample_length=predict_steps + 1)
    delta_xyt = xy_to_delta_xyt(resampled)
    delta_xyt[:, 0:2] *= 4
    if len(delta_xyt) >= predict_steps:
        return delta_xyt[:predict_steps]
    return np.pad(delta_xyt, ((0, predict_steps - len(delta_xyt)), (0, 0)))


def camera_pitch(setting: str) -> int:
    match = re.fullmatch(r"\d+cm_(-?\d+)deg", setting)
    if not match:
        raise ValueError(f"Cannot extract camera pitch from setting {setting!r}")
    return int(match.group(1))


def derive_trajectory_targets(
    rows: list[dict[str, Any]], frame: int, setting: str, horizon: int, predict_steps: int
) -> tuple[list[int], np.ndarray]:
    pose_key = f"pose.{setting}"
    poses = np.asarray([row[pose_key] for row in rows[frame : frame + horizon + 1]], dtype=np.float32)
    interval = 2
    anchor_offsets = np.arange(0, horizon, interval)
    if len(anchor_offsets) > 12:
        interval = int(np.ceil(horizon / 12))
        anchor_offsets = np.arange(0, horizon, interval)

    targets = []
    for offset in anchor_offsets:
        relative = get_trajectory_relative_to_frame(poses[offset:], camera_pitch(setting))
        targets.append(interpolate_and_resample_trajectory(relative, predict_steps))
    return [frame + int(offset) for offset in anchor_offsets], np.stack(targets)


def write_frames_csv(path: Path, rows: list[dict[str, Any]], setting: str, decisions: list[dict[str, Any]]) -> None:
    decision_by_frame = {item["frame"]: item for item in decisions}
    action_values = [int(row["action"]) for row in rows]
    next_actions = action_values[1:] + [0]
    goal_key = f"goal.{setting}"
    relative_key = f"relative_goal_frame_id.{setting}"
    fieldnames = [
        "frame",
        "timestamp",
        "raw_action",
        "raw_action_text",
        "next_action_after_shift",
        "next_action_text",
        "goal_u",
        "goal_v",
        "relative_goal_frame_id",
        "decision_target_type",
        "system2_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            goal = row[goal_key]
            decision = decision_by_frame.get(index, {})
            target = decision.get("system2_target", decision.get("system2_targets", ""))
            writer.writerow(
                {
                    "frame": index,
                    "timestamp": row["timestamp"],
                    "raw_action": action_values[index],
                    "raw_action_text": ACTION_TEXT.get(action_values[index], action_values[index]),
                    "next_action_after_shift": next_actions[index],
                    "next_action_text": ACTION_TEXT.get(next_actions[index], next_actions[index]),
                    "goal_u": goal[0],
                    "goal_v": goal[1],
                    "relative_goal_frame_id": row[relative_key],
                    "decision_target_type": decision.get("target_type", "not sampled at sample_step"),
                    "system2_target": json.dumps(target, ensure_ascii=False),
                }
            )


def write_trajectory_csv(path: Path, anchors: list[int], targets: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["anchor_frame", "step", "dx_normalized", "dy_normalized", "delta_yaw", "dx_m", "dy_m"]
        )
        for anchor, trajectory in zip(anchors, targets):
            for step, (dx, dy, delta_yaw) in enumerate(trajectory):
                writer.writerow([anchor, step, dx, dy, delta_yaw, dx / 4, dy / 4])


def make_depth_preview(source: Path, destination: Path) -> tuple[int, int]:
    depth_mm = np.asarray(Image.open(source), dtype=np.uint16)
    clipped_m = np.clip(depth_mm.astype(np.float32) / 1000.0, 0.0, 5.0)
    preview = np.round((1.0 - clipped_m / 5.0) * 255.0).astype(np.uint8)
    preview[depth_mm == 0] = 0
    Image.fromarray(preview).save(destination)
    valid = depth_mm[depth_mm > 0]
    return int(valid.min()) if valid.size else 0, int(valid.max()) if valid.size else 0


def mark_goal(source: Path, destination: Path, goal_uv: list[int]) -> None:
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    u, v = goal_uv
    radius = 14
    draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=(255, 0, 0), width=5)
    draw.line((u - 22, v, u + 22, v), fill=(255, 255, 0), width=3)
    draw.line((u, v - 22, u, v + 22), fill=(255, 255, 0), width=3)
    draw.rectangle((8, 8, 220, 34), fill=(0, 0, 0))
    draw.text((14, 13), f"goal u={u}, v={v}", fill=(255, 255, 255))
    image.save(destination, quality=95)


def make_overview(front: Path, goal_overlay: Path, depth_preview: Path, destination: Path) -> None:
    panels = []
    for title, path in [("front RGB", front), ("look-down RGB + goal", goal_overlay), ("depth preview (0-5m)", depth_preview)]:
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 360), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (500, 400), "white")
        panel.paste(image, ((500 - image.width) // 2, 32))
        ImageDraw.Draw(panel).text((12, 10), title, fill="black")
        panels.append(panel)
    overview = Image.new("RGB", (sum(panel.width for panel in panels), 400), "white")
    x_offset = 0
    for panel in panels:
        overview.paste(panel, (x_offset, 0))
        x_offset += panel.width
    overview.save(destination, quality=92)


def make_trajectory_plot(trajectory: np.ndarray, destination: Path) -> None:
    # Translation labels are normalized by 4 in the training target.
    positions = np.vstack([np.zeros(2), np.cumsum(trajectory[:, :2] / 4.0, axis=0)])
    lateral = positions[:, 1]
    forward = positions[:, 0]
    width = height = 640
    margin = 60
    lateral_min, lateral_max = float(lateral.min()), float(lateral.max())
    forward_min, forward_max = float(forward.min()), float(forward.max())
    lateral_span = max(lateral_max - lateral_min, 0.5)
    forward_span = max(forward_max - forward_min, 0.5)
    scale = min((width - 2 * margin) / lateral_span, (height - 2 * margin) / forward_span)

    def project(point: np.ndarray) -> tuple[int, int]:
        x = margin + (float(point[1]) - lateral_min) * scale
        y = height - margin - (float(point[0]) - forward_min) * scale
        return round(x), round(y)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    projected = [project(point) for point in positions]
    draw.line(projected, fill=(35, 90, 200), width=5)
    for index, point in enumerate(projected):
        radius = 5 if index not in (0, len(projected) - 1) else 9
        color = (30, 160, 70) if index == 0 else (220, 45, 45) if index == len(projected) - 1 else (35, 90, 200)
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=color)
    draw.text((12, 12), "first anchor GT: forward x / lateral y (meters)", fill="black")
    draw.text((12, 34), f"end = ({forward[-1]:.4f}, {lateral[-1]:.4f})", fill="black")
    image.save(destination)


def export_images(
    scene_root: Path,
    episode: int,
    frame: int,
    front_setting: str,
    lookdown_setting: str,
    goal_uv: list[int],
    output_dir: Path,
) -> tuple[dict[str, Path], tuple[int, int]]:
    chunk = f"chunk-{episode // 1000:03d}"
    media_root = scene_root / "videos" / chunk
    stem = f"episode_{episode:06d}_{frame}"
    original_front = media_root / f"observation.images.rgb.{front_setting}" / f"{stem}.jpg"
    original_lookdown = media_root / f"observation.images.rgb.{lookdown_setting}" / f"{stem}.jpg"
    original_depth = media_root / f"observation.images.depth.{lookdown_setting}" / f"{stem}.png"
    for path in (original_front, original_lookdown, original_depth):
        if not path.exists():
            raise FileNotFoundError(path)

    outputs = {
        "front_rgb": output_dir / "front_rgb.jpg",
        "lookdown_rgb": output_dir / "lookdown_rgb.jpg",
        "depth_raw": output_dir / "depth_raw_mm.png",
        "lookdown_goal_overlay": output_dir / "lookdown_goal_overlay.jpg",
        "depth_preview": output_dir / "depth_preview_0_5m.png",
        "overview": output_dir / "overview.jpg",
    }
    shutil.copy2(original_front, outputs["front_rgb"])
    shutil.copy2(original_lookdown, outputs["lookdown_rgb"])
    shutil.copy2(original_depth, outputs["depth_raw"])
    mark_goal(original_lookdown, outputs["lookdown_goal_overlay"], goal_uv)
    depth_range = make_depth_preview(original_depth, outputs["depth_preview"])
    make_overview(outputs["front_rgb"], outputs["lookdown_goal_overlay"], outputs["depth_preview"], outputs["overview"])
    outputs.update(original_front=original_front, original_lookdown=original_lookdown, original_depth=original_depth)
    return outputs, depth_range


def export_all_media_stream_samples(
    scene_root: Path, episode: int, frame: int, output_dir: Path
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Copy one directly viewable frame from every RGB/depth stream in the real scene."""

    chunk = f"chunk-{episode // 1000:03d}"
    media_root = scene_root / "videos" / chunk
    source_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for stream_dir in sorted(path for path in media_root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"observation\.images\.(rgb|depth)\.(.+)", stream_dir.name)
        if match is None:
            continue
        kind, setting = match.groups()
        suffix = "jpg" if kind == "rgb" else "png"
        source = stream_dir / f"episode_{episode:06d}_{frame}.{suffix}"
        if not source.exists():
            raise FileNotFoundError(source)

        if kind == "rgb":
            local = output_dir / f"media_rgb_{setting}_frame{frame:06d}.jpg"
        else:
            local = output_dir / f"media_depth_{setting}_frame{frame:06d}_raw_mm.png"
        shutil.copy2(source, local)

        with Image.open(source) as image:
            size = list(image.size)
            mode = image.mode
        record: dict[str, Any] = {
            "stream": stream_dir.name,
            "kind": kind,
            "camera_setting": setting,
            "episode_frame_count": sum(1 for _ in stream_dir.glob(f"episode_{episode:06d}_*.{suffix}")),
            "sample_frame": frame,
            "shape_wh": size,
            "pillow_mode": mode,
            "source_path": str(source.resolve()),
            "local_sample": local.name,
            "local_sample_relation": "local_sample 是 source_path 的逐字节副本，未做缩放或转码",
        }
        if kind == "depth":
            preview = output_dir / f"media_depth_{setting}_frame{frame:06d}_preview_0_5m.png"
            minimum_mm, maximum_mm = make_depth_preview(source, preview)
            record.update(
                units="millimeters in raw PNG",
                nonzero_range_mm=[minimum_mm, maximum_mm],
                local_preview=preview.name,
                local_preview_relation="由原始 16-bit 毫米 depth 派生的 0～5m 灰度预览；不是训练输入",
            )
        source_paths.append(source)
        records.append(record)

    if len(records) != 10:
        raise ValueError(f"Expected 10 RGB/depth streams in {media_root}, found {len(records)}")
    return records, source_paths


def markdown_link(label: str, path: str | Path) -> str:
    target = path.as_posix() if isinstance(path, Path) else path
    return f"[{label}]({target})"


def main() -> None:
    args = parse_args()
    scene_root = args.traj_root / args.scene
    if not scene_root.is_dir():
        raise SystemExit(f"Scene directory does not exist: {scene_root}\nUse --traj-root to select another copy.")
    if not args.raw_r2r.is_file():
        raise SystemExit(f"Raw R2R file does not exist: {args.raw_r2r}")

    output_dir = args.output_dir or (
        REPO_ROOT
        / "logs/vln_data_inspection"
        / f"{args.scene}_ep{args.episode:06d}_frame{args.frame:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_meta_path = scene_root / "meta/episodes.jsonl"
    episode_stats_path = scene_root / "meta/episodes_stats.jsonl"
    task_meta_path = scene_root / "meta/tasks.jsonl"
    info_path = scene_root / "meta/info.json"
    parquet_path = (
        scene_root / "data" / f"chunk-{args.episode // 1000:03d}" / f"episode_{args.episode:06d}.parquet"
    )
    for path in (episode_meta_path, episode_stats_path, task_meta_path, info_path, parquet_path):
        if not path.exists():
            raise FileNotFoundError(path)

    episode_meta = read_jsonl_record(episode_meta_path, "episode_index", args.episode)
    if not episode_meta.get("tasks"):
        raise ValueError(f"Episode {args.episode} has no task text")
    instruction = episode_meta["tasks"][0]
    episode_stats = read_jsonl_record(episode_stats_path, "episode_index", args.episode)
    task_matches = [
        record for record in read_jsonl(task_meta_path) if record.get("task", "").strip() == instruction.strip()
    ]
    if not task_matches:
        raise ValueError(f"No task record matches episode instruction {instruction!r}")
    task_record = task_matches[0]
    info_metadata = read_json(info_path)
    info_feature_names = set(info_metadata.get("features", {}))
    raw_data = read_gzip_json(args.raw_r2r)
    raw_array_index, raw_episode, raw_scene_count = find_raw_episode(raw_data, args.scene, instruction)

    parquet_file = pq.ParquetFile(parquet_path)
    table = parquet_file.read()
    rows = table.to_pylist()
    if not 0 <= args.frame < len(rows):
        raise ValueError(f"frame must be in [0, {len(rows) - 1}], got {args.frame}")
    selected_row = rows[args.frame]

    decisions = build_decision_table(rows, args.lookdown_setting, args.sample_step, args.num_future_steps)
    selected_decision = next((item for item in decisions if item["frame"] == args.frame), None)
    if selected_decision is None:
        sampled_frames = [item["frame"] for item in decisions]
        raise ValueError(
            f"frame {args.frame} is not a training decision for sample_step={args.sample_step}; "
            f"choose one of {sampled_frames}"
        )
    if selected_decision["target_type"] != "pixel_goal":
        raise ValueError(
            f"The detailed trajectory export requires a pixel-goal frame; frame {args.frame} is "
            f"{selected_decision['target_type']!r}. Try the default --frame 4."
        )

    horizon = int(selected_decision["relative_goal_frame_id"])
    anchors, trajectory_targets = derive_trajectory_targets(
        rows, args.frame, args.lookdown_setting, horizon, args.predict_steps
    )
    goal_uv = selected_decision["goal_uv"]
    history = history_frames(args.frame, args.num_history)

    # These files are deliberately plain JSON/CSV/TXT so VS Code can open them,
    # unlike the source .json.gz and .parquet containers.
    raw_export = output_dir / f"raw_r2r_episode_{raw_episode['episode_id']}.json"
    info_export = output_dir / "lerobot_info.json"
    episodes_jsonl_export = output_dir / "lerobot_episodes.jsonl"
    episodes_stats_jsonl_export = output_dir / "lerobot_episodes_stats.jsonl"
    tasks_jsonl_export = output_dir / "lerobot_tasks.jsonl"
    lerobot_export = output_dir / f"lerobot_episode_{args.episode:06d}.json"
    lerobot_stats_export = output_dir / f"lerobot_episode_{args.episode:06d}_stats.json"
    lerobot_task_export = output_dir / f"lerobot_task_{task_record['task_index']:06d}.json"
    frame_export = output_dir / f"parquet_frame_{args.frame:06d}.json"
    all_frames_export = output_dir / f"parquet_episode_{args.episode:06d}_all_46x56.jsonl"
    decisions_export = output_dir / "decision_samples.json"
    trajectory_export = output_dir / "trajectory_gt.json"
    frames_csv = output_dir / "frames.csv"
    trajectory_csv = output_dir / "trajectory_gt.csv"
    schema_export = output_dir / "parquet_schema.txt"
    schema_markdown_export = output_dir / "parquet_schema.md"
    media_streams_export = output_dir / "media_streams.json"
    source_tree_export = output_dir / "source_data_tree.json"

    write_json(raw_export, {"raw_array_index": raw_array_index, **raw_episode})
    # Keep exact, workspace-local copies of the source meta files.  Markdown
    # previews often mis-resolve links beginning with /mnt/cpfs as repo paths.
    shutil.copy2(info_path, info_export)
    shutil.copy2(episode_meta_path, episodes_jsonl_export)
    shutil.copy2(episode_stats_path, episodes_stats_jsonl_export)
    shutil.copy2(task_meta_path, tasks_jsonl_export)
    write_json(lerobot_export, episode_meta)
    write_json(lerobot_stats_export, episode_stats)
    write_json(lerobot_task_export, task_record)
    write_json(frame_export, selected_row)
    write_jsonl(all_frames_export, rows)
    write_json(decisions_export, decisions)
    write_json(
        trajectory_export,
        {
            "shape": list(trajectory_targets.shape),
            "anchor_frames": anchors,
            "value_order": ["dx_times_4", "dy_times_4", "delta_yaw"],
            "targets": trajectory_targets.tolist(),
        },
    )
    schema_export.write_text(str(table.schema) + "\n", encoding="utf-8")
    write_parquet_schema_markdown(
        schema_markdown_export,
        schema_export,
        table,
        rows,
        selected_row,
        parquet_path,
        info_feature_names,
        args.front_setting,
        args.lookdown_setting,
        instruction,
    )
    write_frames_csv(frames_csv, rows, args.lookdown_setting, decisions)
    write_trajectory_csv(trajectory_csv, anchors, trajectory_targets)

    image_outputs: dict[str, Path] = {}
    media_records: list[dict[str, Any]] = []
    media_source_paths: list[Path] = []
    depth_range = (0, 0)
    if not args.no_images:
        image_outputs, depth_range = export_images(
            scene_root,
            args.episode,
            args.frame,
            args.front_setting,
            args.lookdown_setting,
            goal_uv,
            output_dir,
        )
        make_trajectory_plot(trajectory_targets[0], output_dir / "trajectory_first_anchor.png")
        media_records, media_source_paths = export_all_media_stream_samples(
            scene_root, args.episode, args.frame, output_dir
        )
    write_json(
        media_streams_export,
        {
            "scene": args.scene,
            "episode_index": args.episode,
            "sample_frame": args.frame,
            "stream_count": len(media_records),
            "streams": media_records,
        },
    )

    target_counts = {
        kind: sum(decision["target_type"] == kind for decision in decisions)
        for kind in ("pixel_goal", "turn", "discarded", "stop")
    }
    stage_a_items = target_counts["pixel_goal"] + target_counts["turn"] + 5 * target_counts["stop"]
    stage_b_items = target_counts["pixel_goal"]
    first_displacement = np.sum(trajectory_targets[0, :, :2], axis=0) / 4.0
    local_key = f"local_point_goal.{args.lookdown_setting}"
    local_point_goal = selected_row.get(local_key)

    write_json(
        source_tree_export,
        {
            "_provenance_comment": {
                "file_role": (
                    "这是教学用的来源索引，由检查脚本汇总生成；它不是原始 R2R/LeRobot 文件，"
                    "训练代码也不会读取 source_data_tree.json"
                ),
                "no_single_original_file": (
                    "本文件没有一个同名的“原来文件”。它同时索引 raw R2R 任务、LeRobot meta、"
                    "Parquet、RGB/depth，以及脚本从这些输入计算出的训练标签"
                ),
                "generated_by": str(Path(__file__).resolve()),
                "reproduce_command": (
                    ".venv/bin/python scripts/inspect_vln_training_sample.py "
                    f"--scene {args.scene} --episode {args.episode} --frame {args.frame} "
                    f"--output-dir {output_dir}"
                ),
                "path_legend": {
                    "source_container/source_root/source_path": "共享盘上的原始数据位置",
                    "viewable_export/local_sample": "为便于在 VS Code 打开而放在本目录的副本或解包文件",
                    "selected_record/selected_frame_export": "从较大的原始容器中抽出的本例记录",
                    "derived_training_view": "按 Dataset 预处理规则计算出的教学结果，原始磁盘上没有这些文件",
                },
                "complete_hash_manifest": "manifest.json 记录每个原始输入的绝对路径、字节数和 SHA-256",
            },
            "raw_r2r_task_definition": {
                "_comment": (
                    f"来源于下面 gzip 解压后的 episodes[{raw_array_index}]。"
                    "viewable_export 只抽出这一条 episode，并额外标注 raw_array_index，不是整个 train.json.gz"
                ),
                "source_container": str(args.raw_r2r.resolve()),
                "source_record": {
                    "json_path": f"episodes[{raw_array_index}]",
                    "raw_array_index_zero_based": raw_array_index,
                    "episode_id": raw_episode["episode_id"],
                    "trajectory_id": raw_episode["trajectory_id"],
                },
                "viewable_export": raw_export.name,
                "viewable_export_relation": (
                    "从 source_container 解压并抽取 source_record；保留原字段，另加 raw_array_index"
                ),
                "contains": [
                    "episode_id and trajectory_id",
                    "scene_id and natural-language instruction",
                    "start_position and start_rotation",
                    "goal and reference_path",
                ],
                "does_not_contain": ["per-frame RGB/depth", "per-frame pose", "per-frame action"],
            },
            "lerobot_scene": {
                "_comment": (
                    "这是逐帧 demonstration 的原始 scene 目录。它与上面的 raw episode 没有显式 ID 映射；"
                    "本例通过 scene=17DRP5sb8fy 且 instruction 唯一完全相同建立对应关系"
                ),
                "source_root": str(scene_root.resolve()),
                "raw_r2r_match": {
                    "rule": "same scene + unique exact instruction",
                    "raw_episode_id": raw_episode["episode_id"],
                    "lerobot_episode_index": args.episode,
                    "ids_are_not_equal_or_interchangeable": True,
                },
                "meta": {
                    "info.json": {
                        "_comment": "viewable_export 是 source_path 的逐字节副本",
                        "source_path": str(info_path.resolve()),
                        "viewable_export": info_export.name,
                        "viewable_export_relation": "byte-for-byte copy",
                        "contains": "scene totals, fps, file templates, declared feature schema",
                    },
                    "episodes.jsonl": {
                        "_comment": (
                            "viewable_export 是完整源文件的逐字节副本；selected_record 是其中 "
                            f"episode_index={args.episode} 的一行重新格式化为 JSON"
                        ),
                        "source_path": str(episode_meta_path.resolve()),
                        "viewable_export": episodes_jsonl_export.name,
                        "viewable_export_relation": "byte-for-byte copy of all JSONL records",
                        "selected_record": lerobot_export.name,
                        "selected_record_locator": f"record where episode_index == {args.episode}",
                        "contains": "scene-local episode_index, instruction list, frame length",
                    },
                    "episodes_stats.jsonl": {
                        "_comment": (
                            "viewable_export 是完整源文件的逐字节副本；selected_record 是其中 "
                            f"episode_index={args.episode} 的统计记录"
                        ),
                        "source_path": str(episode_stats_path.resolve()),
                        "viewable_export": episodes_stats_jsonl_export.name,
                        "viewable_export_relation": "byte-for-byte copy of all JSONL records",
                        "selected_record": lerobot_stats_export.name,
                        "selected_record_locator": f"record where episode_index == {args.episode}",
                        "contains": "per-episode min/max/mean/std/count for declared features",
                    },
                    "tasks.jsonl": {
                        "_comment": (
                            "viewable_export 是完整源文件的逐字节副本；selected_record 是与本 episode "
                            "instruction 完全相同的 task 记录"
                        ),
                        "source_path": str(task_meta_path.resolve()),
                        "viewable_export": tasks_jsonl_export.name,
                        "viewable_export_relation": "byte-for-byte copy of all JSONL records",
                        "selected_record": lerobot_task_export.name,
                        "selected_record_locator": f"record where task_index == {task_record['task_index']}",
                        "contains": "task_index to instruction text mapping",
                    },
                },
                "data": {
                    parquet_path.name: {
                        "_comment": (
                            "source_path 是训练 loader 读取的二进制 Parquet 原件。JSON/JSONL/CSV 都是"
                            "为了阅读而导出的视图，不是另一个原始数据源"
                        ),
                        "source_path": str(parquet_path.resolve()),
                        "rows": len(rows),
                        "columns": table.num_columns,
                        "all_rows_all_columns_export": all_frames_export.name,
                        "all_rows_all_columns_export_relation": (
                            "Parquet 全部行列经 PyArrow 读取后转为可读 JSONL；每行对应一个 frame"
                        ),
                        "selected_frame_export": frame_export.name,
                        "selected_frame_locator": {
                            "zero_based_row_index": args.frame,
                            "frame_index": selected_row["frame_index"],
                        },
                        "schema_export": schema_export.name,
                        "schema_export_relation": (
                            "未经注释的原始 Arrow schema，供程序类型核对"
                        ),
                        "schema_markdown_export": schema_markdown_export.name,
                        "schema_markdown_export_relation": (
                            "VLN 小白版 Markdown：先解释 camera setting、真实 frame、标签和 loss，"
                            "附录再覆盖 56 列含义、frame 实值、来源和当前 loader 使用情况"
                        ),
                        "human_summary_export": frames_csv.name,
                        "human_summary_export_relation": "从 56 列中抽取少量教学字段并加入 decision 解释",
                    }
                },
                "videos": {
                    "_comment": (
                        "source_root 下是逐帧原始媒体。每个 stream 的 local_sample 是 frame 4 原件副本；"
                        "只有 local_preview 是为了看 depth 而生成的派生图"
                    ),
                    "source_root": str((scene_root / "videos").resolve()),
                    "layout": "10 directories = 5 camera settings x RGB/depth",
                    "stream_index_export": media_streams_export.name,
                    "streams": media_records,
                },
            },
            "derived_training_view": {
                "_comment": (
                    "以下内容由检查脚本根据 Parquet pose/action/goal 和 Dataset 规则计算。"
                    "它们没有对应的单个原始文件，也不是训练集磁盘格式"
                ),
                "primary_source": str(parquet_path.resolve()),
                "reference_implementation": str(
                    (REPO_ROOT / "internnav/dataset/internvla_n1_lerobot_dataset.py").resolve()
                ),
                "decision_split": decisions_export.name,
                "decision_split_relation": (
                    "由 episode 的 action、goal、relative_goal_frame_id 按采样规则划分 pixel-goal/turn/discarded/stop"
                ),
                "selected_system2_targets": selected_decision["system2_targets"],
                "selected_system2_targets_relation": (
                    f"↓ 是 look-down 请求符号；{goal_uv[0]} {goal_uv[1]} 来自 frame {args.frame} 的 "
                    f"goal.{args.lookdown_setting}=[u,v]"
                ),
                "trajectory_ground_truth": trajectory_export.name,
                "trajectory_ground_truth_csv": trajectory_csv.name,
                "trajectory_ground_truth_relation": (
                    "由 pose 窗口做相对坐标变换、插值并按 anchor 整理成 [6,32,3]；JSON 与 CSV 是同一结果"
                ),
            },
        },
    )

    summary = {
        "source_relationship": {
            "raw_match_rule": "same scene plus unique exact instruction; no explicit ID map is stored in LeRobot meta",
            "raw_r2r_array_index": raw_array_index,
            "raw_episode_id": raw_episode["episode_id"],
            "raw_trajectory_id": raw_episode["trajectory_id"],
            "raw_episodes_in_scene": raw_scene_count,
            "lerobot_episode_index": args.episode,
            "lerobot_length": episode_meta["length"],
        },
        "selected_decision": {
            **selected_decision,
            "history_frames": history,
            "current_front_frame": args.frame,
            "lookdown_frame": args.frame,
            "coordinate_convention": "Parquet/model text use u v = x y; policy swaps to internal row-column later",
            "trajectory_anchor_frames": anchors,
            "trajectory_target_shape": list(trajectory_targets.shape),
            "first_anchor_sum_dxdy_div4": first_displacement.tolist(),
            "parquet_local_point_goal": local_point_goal,
        },
        "episode_camera_config_sample_counts": {
            "scope": "this one LeRobot episode and this one camera configuration only",
            "camera_config": f"front={args.front_setting}, lookdown={args.lookdown_setting}",
            **target_counts,
            "stage_a_items_after_stop_x5": stage_a_items,
            "stage_b_pixel_goal_items": stage_b_items,
        },
        "loss_contract": {
            "stage_a": "assistant tokens '↓' and 'u v' use shifted causal cross entropy; images/instruction are masked",
            "stage_b_x0_shape": list(trajectory_targets.shape),
            "stage_b_text_cross_entropy": False,
            "flow_input": "x_sigma = (1-sigma)*x0 + sigma*epsilon",
            "flow_target": "epsilon - x0",
            "loss": "sum(mask * (v_pred - (epsilon - x0))^2) / (valid_anchor_count * 32 * 3)",
            "mask_scope": "only collator-added fake anchors are masked; all 32x3 values, including zero tails, participate",
        },
    }
    summary_export = output_dir / "sample_summary.json"
    write_json(summary_export, summary)

    manifest_sources = [
        args.raw_r2r,
        info_path,
        episode_meta_path,
        episode_stats_path,
        task_meta_path,
        parquet_path,
        *media_source_paths,
    ]
    manifest_sources = list(dict.fromkeys(manifest_sources))
    manifest = {
        "generated_by": str(Path(__file__).resolve()),
        "pyarrow_version": pyarrow.__version__,
        "sources": [source_entry(path) for path in manifest_sources],
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    manifest_export = output_dir / "manifest.json"
    write_json(manifest_export, manifest)

    dataset_source = REPO_ROOT / "internnav/dataset/internvla_n1_lerobot_dataset.py"
    policy_source = REPO_ROOT / "internnav/model/basemodel/internvla_n1/internvla_n1_policy.py"
    loss_source = REPO_ROOT / "internnav/model/basemodel/internvla_n1/internvla_n1.py"
    dataset_source_link = Path(os.path.relpath(dataset_source, output_dir))
    policy_source_link = Path(os.path.relpath(policy_source, output_dir))
    loss_source_link = Path(os.path.relpath(loss_source, output_dir))
    report_lines = [
        "# 真实 R2R → LeRobot → DualVLN 样本检查结果",
        "",
        f"由 `{Path(__file__).resolve()}` 生成。这里只读原始数据，没有加载模型或修改数据集。",
        "",
        f"**如果你刚开始学 VLN，请先读 {markdown_link('VLN 小白版：相机 setting、frame、标签和 loss', schema_markdown_export.name)}。**",
        "它从两张真实相机图片讲起，技术性的 56 列表在文末，不需要先看。",
        "",
        "## 先看可视化",
        "",
    ]
    if image_outputs:
        report_lines.extend(
            [
                "![front / look-down waypoint / depth](overview.jpg)",
                "",
                "![first trajectory target](trajectory_first_anchor.png)",
                "",
            ]
        )
    report_lines.extend(
        [
            "## 这一条真实数据",
            "",
            f"- raw R2R：数组下标 `{raw_array_index}`，episode_id `{raw_episode['episode_id']}`，"
            f"trajectory_id `{raw_episode['trajectory_id']}`。",
            f"- LeRobot：scene `{args.scene}`，episode_index `{args.episode}`，共 `{len(rows)}` 帧。",
            f"- instruction：`{instruction.strip()}`",
            f"- 当前 decision：frame `{args.frame}`，history `{history}`。",
            f"- Qwen 这一条实际看到 `{len(history)}` 张历史前视图、1 张当前前视图和 1 张当前朝下图，"
            f"共 `{len(history) + 2}` 张图。",
            f"- Parquet waypoint：`{goal_uv}`，即 `u={goal_uv[0]}`（横向 x），"
            f"`v={goal_uv[1]}`（纵向 y）。",
            f"- System 2 真值：先输出 `↓`，看朝下图，再输出文本 `{goal_uv[0]} {goal_uv[1]}`。",
            f"- System 1 真值：anchor frames `{anchors}`，shape `{list(trajectory_targets.shape)}`。",
            f"- 只计算本 episode、本相机配置时：Stage A 有 `{stage_a_items}` 个 item，"
            f"Stage B 有 `{stage_b_items}` 个 item；这不是整个训练集大小。",
            f"- 首个 anchor 的 32 步平移求和后除以 4：`{first_displacement.tolist()}`；"
            f"Parquet local point goal：`{local_point_goal}`。",
            "",
            "raw 与 LeRobot 的 ID 不能直接画等号；这里按同一 scene 和唯一完全相同的 instruction 匹配。",
            "",
            "## 可以直接点击的解包文件",
            "",
            f"- {markdown_link('解压后的 raw R2R episode', raw_export.name)}",
            f"- {markdown_link('LeRobot info.json 本地副本', info_export.name)}",
            f"- {markdown_link('LeRobot episodes.jsonl 本地副本', episodes_jsonl_export.name)}",
            f"- {markdown_link('LeRobot episodes_stats.jsonl 本地副本', episodes_stats_jsonl_export.name)}",
            f"- {markdown_link('LeRobot tasks.jsonl 本地副本', tasks_jsonl_export.name)}",
            f"- {markdown_link('LeRobot episode meta', lerobot_export.name)}",
            f"- {markdown_link('LeRobot episode 统计', lerobot_stats_export.name)}",
            f"- {markdown_link('LeRobot task 记录', lerobot_task_export.name)}",
            f"- {markdown_link('Parquet 全部 46×56 内容 JSONL', all_frames_export.name)}",
            f"- {markdown_link('这一帧的全部 56 个 Parquet 字段', frame_export.name)}",
            f"- {markdown_link('VLN 小白版：相机 setting、frame、标签和 56 列字典', schema_markdown_export.name)}",
            f"- {markdown_link('未经注释的原始 Arrow schema', schema_export.name)}",
            f"- {markdown_link('46 帧扁平表', frames_csv.name)}",
            f"- {markdown_link('10 个 RGB/depth 流索引', media_streams_export.name)}",
            f"- {markdown_link('源数据结构 JSON', source_tree_export.name)}",
            f"- {markdown_link('实际 decision 切分', decisions_export.name)}",
            f"- {markdown_link('轨迹 GT JSON', trajectory_export.name)}",
            f"- {markdown_link('轨迹 GT CSV', trajectory_csv.name)}",
            f"- {markdown_link('结构化解释与 loss 公式', summary_export.name)}",
            f"- {markdown_link('原始文件路径与 SHA-256', manifest_export.name)}",
            "",
        ]
    )
    if media_records:
        report_lines.extend(["## 10 个真实相机流的同一 frame 4", ""])
        for record in media_records:
            label = f"{record['kind']} · {record['camera_setting']}"
            report_lines.append(f"- {markdown_link(label, record['local_sample'])}")
            if record.get("local_preview"):
                report_lines.append(
                    f"  - {markdown_link('0～5m 人眼预览', record['local_preview'])}"
                )
        report_lines.append("")
    report_lines.extend(
        [
            "## 对应源码",
            "",
            f"- {markdown_link('Parquet → decision sample → target', dataset_source_link)}",
            f"- {markdown_link('u v 文本转内部 row-column', policy_source_link)}",
            f"- {markdown_link('System 1 Flow Matching loss', loss_source_link)}",
            "",
            "## Depth 说明",
            "",
            f"原始 depth 是 16-bit 毫米 PNG；本帧非零范围 `{depth_range[0]}..{depth_range[1]}` mm。"
            "`depth_preview_0_5m.png` 只是方便人眼查看的 0～5m 灰度预览，训练仍读取原始 PNG，"
            "除以 1000 后截断到 5m；但当前 `nextdit_async` forward/loss 不使用 depth。",
            "",
        ]
    )
    report_path = output_dir / "README.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Refresh manifest once README and manifest itself exist.
    manifest["outputs"] = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    write_json(manifest_export, manifest)

    print(f"Wrote inspectable sample to: {output_dir.resolve()}")
    print(f"Open this report: {report_path.resolve()}")
    print(
        f"raw episode {raw_episode['episode_id']} -> LeRobot episode {args.episode}, "
        f"frame {args.frame}, System 2 target {selected_decision['system2_targets']}, "
        f"System 1 target shape {tuple(trajectory_targets.shape)}"
    )


if __name__ == "__main__":
    main()
