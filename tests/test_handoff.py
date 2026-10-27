import base64
import hashlib
import importlib
import sys
from types import ModuleType, SimpleNamespace as NS
import numpy as np
import pytest
from navpi.download import verify_object
from score_navsim import check_coverage


def test_download_rejects_corrupt_or_incomplete_files(tmp_path):
    target = tmp_path / "object"
    payload = b"official checkpoint object fixture"
    metadata = {"size": len(payload), "md5Hash": base64.b64encode(hashlib.md5(payload).digest()).decode()}
    assert not verify_object(target, metadata)
    target.write_bytes(payload)
    assert verify_object(target, metadata)
    target.write_bytes(b"X" * len(payload))
    assert not verify_object(target, metadata)
    target.write_bytes(payload[:4])
    assert not verify_object(target, metadata)


def test_pdm_coverage_rejects_missing_duplicate_invalid_predictions():
    manifest = [{"token": "a"}, {"token": "b"}]
    rows = [{"token": t, "trajectory": np.zeros((8, 3)).tolist()} for t in ("a", "b")]
    check_coverage(rows, manifest)
    for invalid in (rows[:1], rows[::-1], rows + rows[:1],
                    [{**rows[0], "trajectory": [[float("nan")]*3]*8}, rows[1]]):
        with pytest.raises(ValueError):
            check_coverage(invalid, manifest)


def test_navsim_adapter_contract_uses_latest_three_cameras(monkeypatch):
    # Dependency-isolated contract test; not an official NAVSIM scoring run.
    class Base:
        def __init__(self, requires_scene): assert not requires_scene
    class Sensor:
        @staticmethod
        def build_no_sensors(): return NS(cam_f0=[], cam_l0=[], cam_r0=[], cam_b0=[])
    for name, values in {
        "navsim.agents.abstract_agent": {"AbstractAgent": Base},
        "navsim.common.dataclasses": {"SensorConfig": Sensor, "Trajectory": lambda poses, sampling: NS(poses=poses, sampling=sampling)},
        "nuplan.planning.simulation.trajectory.trajectory_sampling": {"TrajectorySampling": lambda **kw: NS(**kw)},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("navpi.navsim_agent", None)
    adapter = importlib.import_module("navpi.navsim_agent")
    agent = adapter.NavPiAgent("unused")
    sensors = agent.get_sensor_config()
    assert sensors.cam_f0 == sensors.cam_l0 == sensors.cam_r0 == [3] and sensors.cam_b0 == []
    captured = {}
    def predict(images, state, prompt, seed):
        captured.update(images=images, state=state, prompt=prompt)
        return np.zeros((8, 3), np.float32)
    agent.policy = NS(predict=predict)
    ego = NS(ego_velocity=np.array([3., 4.]), ego_acceleration=np.array([1., 2.]), driving_command=[0, 1, 0, 0])
    cams = NS(cam_f0=NS(image="front"), cam_l0=NS(image="left"), cam_r0=NS(image="right"))
    trajectory = agent.compute_trajectory(NS(ego_statuses=[None, ego], cameras=[None, cams]))
    assert captured["state"] == [5., 3., 4., 1., 2.]
    assert list(captured["images"].values()) == ["front", "left", "right"]
    assert captured["prompt"].endswith("0 1 0 0")
    assert trajectory.poses.shape == (8, 3) and trajectory.sampling.interval_length == .5
    sys.modules.pop("navpi.navsim_agent", None)
