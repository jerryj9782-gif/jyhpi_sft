"""Prepare exactly the installed NAVSIM v1 navtest filter; never fit test statistics."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
import yaml
from navpi.checkpoints import sha256
from navpi.data import CAMERAS, SENSOR_CAMERAS, read_rows
from prepare_data import group_name, local_trajectory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--navsim-root", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, default=Path("assets/navsim/train.jsonl"))
    p.add_argument("--val-manifest", type=Path, default=Path("assets/navsim/val.jsonl"))
    p.add_argument("--output", type=Path, default=Path("assets/navtest"))
    a = p.parse_args()
    source = a.navsim_root / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    config = yaml.safe_load(source.read_text())
    if (config["num_history_frames"], config["num_future_frames"], config["frame_interval"],
        config["has_route"], config["max_scenes"]) != (4, 10, 1, True, None):
        raise ValueError("Expected NAVSIM v1 official navtest scene filter")
    expected = set(config["tokens"])
    if not expected or len(expected) != len(config["tokens"]):
        raise ValueError("Empty or duplicate official tokens")
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(a.output)
    used = read_rows(a.train_manifest) + read_rows(a.val_manifest)
    used_tokens, used_groups = {r["token"] for r in used}, {r["group"] for r in used}
    rows, irregular = {}, []
    for log in sorted(config["log_names"]):
        with (a.root / "navsim_logs/test" / f"{log}.pkl").open("rb") as stream:
            frames = pickle.load(stream)  # trusted local NAVSIM dataset only
        group = group_name(log)
        for index in range(3, len(frames) - 10):
            frame = frames[index]
            token = frame["token"]
            if token not in expected or not frame["roadblock_ids"]:
                continue
            if token in rows or token in used_tokens or group in used_groups:
                raise ValueError(f"Duplicate token or train/val/test leakage: {token}")
            dynamics = np.asarray(frame["ego_dynamic_state"], dtype=np.float32)
            images = {dest: str(Path("sensor_blobs/test") / frame["cams"][camera]["data_path"])
                      for camera, dest in zip(SENSOR_CAMERAS, CAMERAS, strict=True)}
            for path in images.values():
                if not (a.root / path).is_file():
                    raise FileNotFoundError(a.root / path)
            actions = local_trajectory(frames, index)
            if dynamics.shape != (4,) or not np.isfinite(dynamics).all() or not np.isfinite(actions).all():
                raise ValueError(f"Invalid state or trajectory: {token}")
            times = np.array([f["timestamp"] for f in frames[index:index + 9]])
            if not np.all(np.abs(np.diff(times) / 1e6 - .5) < .05):
                irregular.append(token)  # retain official coverage, report irregular timestamps
            rows[token] = dict(token=token, group=group, images=images,
                state=[float(np.linalg.norm(dynamics[:2])), *dynamics.tolist()],
                prompt="Drive safely. Recorded navigation command: " +
                    " ".join(map(str, np.asarray(frame["driving_command"], int))),
                actions=actions.tolist())
    if set(rows) != expected:
        raise ValueError(f"Missing official tokens: {sorted(expected - set(rows))[:20]}")
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = a.output / "navtest.jsonl"
    manifest.write_text("".join(json.dumps(rows[t]) + "\n" for t in sorted(rows)))
    audit = dict(protocol="NAVSIM v1 navtest; exact official tokens", samples=len(rows),
        official_filter_sha256=sha256(source), manifest_sha256=sha256(manifest),
        train_manifest_sha256=sha256(a.train_manifest), val_manifest_sha256=sha256(a.val_manifest),
        token_overlap=0, recording_overlap=0, images_checked=True,
        irregular_timestamps=irregular, normalization="Use checkpoint statistics, no test fitting")
    (a.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Prepared all {len(rows)} official tokens in {manifest}")


if __name__ == "__main__":
    main()
