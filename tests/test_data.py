import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import sentencepiece as spm
import torch
from navpi.data import CAMERAS, CachedDataset, ManifestDataset, Normalizer, Tokenizer, resize_image, to_device
from prepare_data import export, local_trajectory


def test_local_trajectory_rotation_and_heading_wrap():
    def frame(x, y, yaw):
        return {"ego2global_translation": [x, y, 0], "ego2global_rotation": [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]}
    frames = [frame(10, 20 + i, math.pi / 2) for i in range(9)]
    trajectory = local_trajectory(frames, 0)
    np.testing.assert_allclose(trajectory[:, 0], np.arange(1, 9), atol=1e-6)
    np.testing.assert_allclose(trajectory[:, 1:], 0, atol=1e-6)
    frames = [frame(i, 0, 3.13 if i == 0 else -3.13) for i in range(9)]
    assert np.max(np.abs(local_trajectory(frames, 0)[:, 2])) < .03


def test_letterbox_keeps_geometry_and_uint8():
    image = Image.fromarray(np.full((20, 40, 3), 255, np.uint8))
    output = resize_image(image, 28)
    assert output.shape == (3, 28, 28) and output.dtype == np.uint8
    assert np.all(output[:, :7] == 0) and np.all(output[:, 7:21] == 255)


def test_data_pipeline_reads_only_three_cameras(tmp_path):
    image = tmp_path / "front.jpg"
    Image.fromarray(np.full((20, 40, 3), 127, np.uint8)).save(image)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("Task: Drive safely. Recorded navigation command: 0 1 0 0, State: 128 128 128 128 128; Action: \n") * 30)
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / "tokenizer"),
                                  vocab_size=128, hard_vocab_limit=False, model_type="char", minloglevel=2)
    stats = {"norm_stats": {key: {"q01": [-1.] * n, "q99": [1.] * n} for key, n in (("state", 5), ("actions", 3))}}
    stats_file = tmp_path / "stats.json"; stats_file.write_text(json.dumps(stats))
    row = {"token": "a", "group": "a", "images": {c: str(image) for c in CAMERAS},
           "state": [0.] * 5, "actions": [[0., 0., .2]] * 8, "prompt": "Drive safely."}
    row["images"]["cam_rear_center"] = "THIS_FILE_MUST_NOT_BE_READ.jpg"
    manifest = tmp_path / "train.jsonl"; manifest.write_text(json.dumps(row) + "\n")
    dataset = ManifestDataset(manifest, tmp_path, stats_file, tmp_path / "tokenizer.model", image_size=28, max_text_len=256)
    sample = dataset[0]
    assert sample["images"].shape == (3, 3, 28, 28)
    batch = to_device({k: v[None] for k, v in sample.items()}, torch.device("cpu"))
    assert batch["images"].min() >= -1 and batch["images"].max() <= 1
    normalizer = Normalizer(stats_file)
    np.testing.assert_allclose(normalizer.decode_actions(sample["actions"].numpy()), row["actions"], atol=1e-6)
    cached = tmp_path / "cache"; cached.mkdir()
    for key, value in sample.items():
        np.save(cached / f"{key}.npy", value.numpy()[None])
    (cached / "metadata.json").write_text(json.dumps({"samples": 1, "stats_sha256": "test"}))
    loaded = CachedDataset(cached, expected={"stats_sha256": "test"})[0]
    for key in sample:
        torch.testing.assert_close(sample[key], loaded[key])
    with pytest.raises(ValueError, match="Stale"):
        CachedDataset(cached, expected={"stats_sha256": "wrong"})
    with pytest.raises(ValueError, match="refusing truncation"):
        Tokenizer(tmp_path / "tokenizer.model", max_len=2).encode("Drive safely.", np.zeros(5))


@pytest.mark.parametrize("leak", ["group", "token", "image", "test"])
def test_split_leakage_rejected(tmp_path, leak):
    def row(token, group, camera):
        return {"token": token, "group": group, "images": {c: f"sensor_blobs/{camera}/{c}.jpg" for c in CAMERAS},
                "state": [0.] * 5, "actions": [[0., 0., 0.]] * 8, "prompt": "Drive"}
    rows = {"train": [row("one", "group1", "first")], "val": [row("two", "group2", "second")]}
    if leak == "group": rows["val"][0]["group"] = "group1"
    if leak == "token": rows["val"][0]["token"] = "one"
    if leak == "image": rows["val"][0]["images"] = rows["train"][0]["images"]
    with pytest.raises(ValueError):
        export(rows, tmp_path, tmp_path / "out", {"group1"} if leak == "test" else set())


def test_stats_use_train_only(tmp_path):
    def row(token, group, action):
        return {"token": token, "group": group, "images": {c: f"sensor_blobs/{group}/{c}.jpg" for c in CAMERAS},
                "state": [action] * 5, "actions": [[action] * 3] * 8, "prompt": "Drive"}
    rows = {"train": [row("a", "a", 1.)], "val": [row("b", "b", 999.)]}
    export(rows, tmp_path, tmp_path / "out", set())
    stats = json.loads((tmp_path / "out/norm_stats.json").read_text())
    assert stats["norm_stats"]["actions"]["mean"] == [1., 1., 1.]
