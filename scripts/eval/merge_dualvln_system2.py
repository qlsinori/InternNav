#!/usr/bin/env python3
"""Build a DualVLN checkpoint with System 2 weights from another checkpoint.

The released DualVLN checkpoint stores Qwen/System 2 and System 1 in one
state dict.  A standalone System 2 checkpoint is therefore merged by replacing
every shared tensor while retaining tensors that exist only in DualVLN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


INDEX_NAME = "model.safetensors.index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-checkpoint", type=Path, required=True)
    parser.add_argument("--system2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_index(root: Path) -> dict:
    path = root / INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shapes(root: Path, weight_map: dict[str, str]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)
    for shard, keys in sorted(by_shard.items()):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                shapes[key] = list(handle.get_slice(key).get_shape())
    return shapes


def main() -> None:
    args = parse_args()
    dual_root = args.dual_checkpoint.resolve()
    system2_root = args.system2_checkpoint.resolve()
    output_root = args.output_dir.resolve()

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")

    dual_index = read_index(dual_root)
    system2_index = read_index(system2_root)
    dual_map = dual_index["weight_map"]
    system2_map = system2_index["weight_map"]
    dual_keys = set(dual_map)
    system2_keys = set(system2_map)

    missing_from_dual = sorted(system2_keys - dual_keys)
    if missing_from_dual:
        raise ValueError(
            f"System 2 has {len(missing_from_dual)} keys absent from DualVLN; "
            f"first: {missing_from_dual[:5]}"
        )

    dual_shapes = tensor_shapes(dual_root, dual_map)
    system2_shapes = tensor_shapes(system2_root, system2_map)
    mismatched = {
        key: (dual_shapes[key], system2_shapes[key])
        for key in system2_keys
        if dual_shapes[key] != system2_shapes[key]
    }
    if mismatched:
        first = next(iter(mismatched.items()))
        raise ValueError(f"Shape mismatch for {len(mismatched)} keys; first: {first}")

    temp_root = output_root.parent / f".{output_root.name}.tmp-{os.getpid()}"
    temp_root.mkdir(parents=True, exist_ok=False)
    copied_shared_shards: list[str] = []

    try:
        for source in dual_root.iterdir():
            if not source.is_file() or source.suffix == ".safetensors":
                continue
            shutil.copy2(source, temp_root / source.name)

        dual_by_shard: dict[str, list[str]] = {}
        system2_by_shard: dict[str, list[str]] = {}
        for key, shard in dual_map.items():
            dual_by_shard.setdefault(shard, []).append(key)
        for key, shard in system2_map.items():
            system2_by_shard.setdefault(shard, []).append(key)

        for shard, keys in sorted(dual_by_shard.items()):
            keys = sorted(keys)
            can_copy_system2_shard = (
                set(keys) == set(system2_by_shard.get(shard, []))
                and all(system2_map[key] == shard for key in keys)
            )
            target = temp_root / shard
            if can_copy_system2_shard:
                shutil.copy2(system2_root / shard, target)
                copied_shared_shards.append(shard)
                print(f"copied complete System 2 shard: {shard}", flush=True)
                continue

            required_system2_shards = sorted({system2_map[key] for key in keys if key in system2_map})
            with ExitStack() as stack:
                dual_handle = stack.enter_context(
                    safe_open(dual_root / shard, framework="pt", device="cpu")
                )
                system2_handles = {
                    name: stack.enter_context(
                        safe_open(system2_root / name, framework="pt", device="cpu")
                    )
                    for name in required_system2_shards
                }
                tensors = {}
                for key in keys:
                    if key in system2_map:
                        tensors[key] = system2_handles[system2_map[key]].get_tensor(key)
                    else:
                        tensors[key] = dual_handle.get_tensor(key)
                save_file(tensors, target, metadata={"format": "pt"})
            print(f"merged shard: {shard} ({len(keys)} tensors)", flush=True)

        output_shapes = tensor_shapes(temp_root, dual_map)
        if output_shapes != dual_shapes:
            raise RuntimeError("Merged checkpoint key/shape verification failed")

        provenance = {
            "format": "DualVLN checkpoint with shared Qwen/System2 tensors overridden",
            "dual_checkpoint": str(dual_root),
            "system2_checkpoint": str(system2_root),
            "dual_index_sha256": sha256(dual_root / INDEX_NAME),
            "system2_index_sha256": sha256(system2_root / INDEX_NAME),
            "dual_config_sha256": sha256(dual_root / "config.json"),
            "system2_config_sha256": sha256(system2_root / "config.json"),
            "shared_system2_tensors_overridden": len(system2_keys),
            "dual_only_system1_and_latent_tensors_retained": len(dual_keys - system2_keys),
            "copied_complete_system2_shards": copied_shared_shards,
            "output_shards": sorted(dual_by_shard),
        }
        (temp_root / "HYBRID_PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        temp_root.rename(output_root)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    print(f"hybrid checkpoint ready: {output_root}", flush=True)


if __name__ == "__main__":
    main()
