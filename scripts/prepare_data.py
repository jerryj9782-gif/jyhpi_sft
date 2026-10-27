"""Prepare only current F0/L0/R0 inputs and eight ego-relative future poses.

Import old manifests without depending on openpi_old at training time, or build
custom recording-disjoint train/val manifests from trusted NAVSIM pickle logs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from navpi.data import CAMERAS, SENSOR_CAMERAS, image_path, read_rows
from navpi.checkpoints import sha256


def group_name(stem):
    if "_veh-" not in stem:
        raise ValueError(f"Unrecognized original-recording name: {stem}")
    prefix, rest = stem.split("_veh-", 1)
    return prefix + "_veh-" + rest.split("_")[0]


def local_trajectory(frames, index):
    def pose(frame):
        w, x, y, z = np.asarray(frame["ego2global_rotation"], np.float64)
        # Same intrinsic yaw convention as NAVSIM's pyquaternion yaw_pitch_roll.
        yaw = np.arctan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z))
        return np.array([*frame["ego2global_translation"][:2], yaw])
    origin = pose(frames[index])
    future = np.stack([pose(f) for f in frames[index + 1:index + 9]])
    delta = future[:, :2] - origin[:2]
    c, s = np.cos(origin[2]), np.sin(origin[2])
    xy = delta @ np.array([[c, -s], [s, c]])
    heading = (future[:, 2] - origin[2] + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack((xy, heading)).astype(np.float32)


def make_rows(root, source, stride, test_groups):
    result = {"train": [], "val": []}
    seen = {}
    skipped = Counter()
    for filename in sorted((root / "navsim_logs" / source).glob("*.pkl")):
        group = group_name(filename.stem)
        if group in test_groups:
            skipped["test_recording"] += 1
            continue
        split = "val" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
        # Only load dataset pickle files that you trust.
        with filename.open("rb") as stream:
            frames = pickle.load(stream)
        for index in range(3, len(frames) - 8, stride):
            frame = frames[index]
            timestamps = np.array([f["timestamp"] for f in frames[index:index + 9]])
            if not np.all(np.abs(np.diff(timestamps) / 1e6 - .5) < .05):
                skipped["time_discontinuity"] += 1
                continue
            dynamics = np.asarray(frame["ego_dynamic_state"], np.float32)
            if dynamics.shape != (4,):
                raise ValueError("Expected vx,vy,ax,ay")
            images = {dest: str(Path("sensor_blobs") / source / frame["cams"][src]["data_path"])
                      for src, dest in zip(SENSOR_CAMERAS, CAMERAS, strict=True)}
            row = {"token": frame["token"], "group": group, "images": images,
                "state": [float(np.linalg.norm(dynamics[:2])), *dynamics.tolist()],
                "prompt": "Drive safely. Recorded navigation command: " +
                    " ".join(map(str, np.asarray(frame["driving_command"], int))),
                "actions": local_trajectory(frames, index).tolist()}
            previous = seen.get(row["token"])
            if previous is not None:
                if previous != row:
                    raise ValueError(f"Conflicting duplicate token: {row['token']}")
                skipped["duplicate_token"] += 1
                continue
            seen[row["token"]] = row
            result[split].append(row)
    return result, dict(skipped)


def export(rows, root, output, test_groups, *, check_images=False, provenance=None):
    if any(output.glob("*.json*")):
        raise FileExistsError(f"Refusing to overwrite manifests in {output}")
    tokens, groups, images_seen = {}, {}, {}
    for split in ("train", "val"):
        if not rows[split]:
            raise ValueError(f"Empty {split}")
        tokens[split], groups[split], images_seen[split] = set(), set(), set()
        for row in rows[split]:
            if row["token"] in tokens[split]:
                raise ValueError(f"Duplicate token {row['token']}")
            tokens[split].add(row["token"]); groups[split].add(row["group"])
            if row["group"] in test_groups:
                raise ValueError(f"Test recording leaked into {split}: {row['group']}")
            converted = {}
            for camera in CAMERAS:
                path = image_path(root, row["images"][camera])
                parts = path.parts
                if "sensor_blobs" not in parts:
                    raise ValueError(f"No sensor_blobs component in {path}")
                converted[camera] = str(Path(*parts[parts.index("sensor_blobs"):]))
                if check_images and not (root / converted[camera]).is_file():
                    raise FileNotFoundError(root / converted[camera])
            row["images"] = converted
            images_seen[split].update(converted.values())
            if np.asarray(row["state"]).shape != (5,) or np.asarray(row["actions"]).shape != (8, 3):
                raise ValueError(f"Bad shapes: {row['token']}")
            if not all(np.isfinite(row[key]).all() for key in ("state", "actions")):
                raise ValueError(f"Nonfinite entry: {row['token']}")
    for sets in (tokens, groups, images_seen):
        if sets["train"] & sets["val"]:
            raise ValueError("Train/validation leakage")
    stats = {}
    for key in ("state", "actions"):
        values = np.asarray([r[key] for r in rows["train"]], np.float64)
        values = values.reshape(-1, values.shape[-1])
        stats[key] = {"mean": values.mean(0).tolist(), "std": values.std(0).tolist(),
                      "q01": np.quantile(values, .01, axis=0).tolist(),
                      "q99": np.quantile(values, .99, axis=0).tolist()}
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        with (output / f"{split}.jsonl").open("w") as stream:
            for row in rows[split]:
                stream.write(json.dumps(row) + "\n")
    (output / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}, indent=2))
    audit = {"cameras": list(CAMERAS), "horizon": 8, "dt": .5,
             "normalization": "train only", "split": "custom recording-disjoint; not official leaderboard split",
             "test_recordings_checked": len(test_groups), "images_checked": check_images,
             "provenance": provenance,
             "overlap": {"tokens": 0, "recordings": 0, "images": 0},
             "samples": {s: len(rows[s]) for s in rows},
             "groups": {s: sorted(groups[s]) for s in groups},
             "sha256": {p.name: sha256(p) for p in output.glob("*.json*")}}
    (output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k not in ("groups", "sha256")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--from-manifests", type=Path)
    parser.add_argument("--source-split", default="trainval")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("assets/navsim"))
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    if args.source_split == "test" or args.stride < 1:
        parser.error("Use a training source split and stride >= 1")
    files = list((args.root / "navsim_logs/test").glob("*.pkl"))
    if not files:
        parser.error("Test log filenames required to check recording exclusion (test labels are not read)")
    test_groups = {group_name(p.stem) for p in files}
    if args.from_manifests:
        rows = {s: read_rows(args.from_manifests / f"{s}.jsonl") for s in ("train", "val")}
        provenance = {"method": "import", "source": str(args.from_manifests)}
    else:
        rows, skipped = make_rows(args.root, args.source_split, args.stride, test_groups)
        provenance = {"method": "raw_logs", "stride": args.stride, "skipped": skipped}
    export(rows, args.root, args.output, test_groups, check_images=args.check_images, provenance=provenance)


if __name__ == "__main__":
    main()
