"""Export all PyTorch checkpoints in a directory to ONNX for browser inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from omokai.checkpoint import load_checkpoint, list_checkpoints


def export_one(ckpt_path: Path, out_path: Path) -> dict:
    model, config, metadata = load_checkpoint(ckpt_path, device="cpu")
    model.eval()
    board_size = config.rules.board_size
    in_planes = config.network.input_planes
    dummy = torch.zeros(1, in_planes, board_size, board_size, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    return {
        "name": ckpt_path.stem,
        "file": out_path.name,
        "board_size": board_size,
        "input_planes": in_planes,
        "channels": config.network.channels,
        "blocks": config.network.blocks,
        "exactly_five": config.rules.exactly_five,
        "size_bytes": out_path.stat().st_size,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="checkpoints/rocm_24h")
    parser.add_argument("--out-dir", default="web/models")
    parser.add_argument("--manifest", default="web/models/manifest.json")
    parser.add_argument("--include-latest", action="store_true", default=True)
    parser.add_argument("--best-iteration", type=int, default=None,
                        help="Mark this iter_XXXX.pt as the recommended best in the manifest")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list_checkpoints(ckpt_dir)
    if not paths:
        raise SystemExit(f"no checkpoints found in {ckpt_dir}")

    entries = []
    for path in paths:
        out = out_dir / f"{path.stem}.onnx"
        print(f"exporting {path.name} -> {out.name}")
        entry = export_one(path, out)
        entries.append(entry)

    if args.best_iteration is not None:
        target_iter = args.best_iteration
        for entry in entries:
            iteration = entry.get("metadata", {}).get("iteration")
            entry["recommended"] = (
                entry["name"] == "best"
                or iteration == target_iter
                or entry["name"] == f"iter_{target_iter:04d}"
            )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"checkpoints": entries}, indent=2))
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
