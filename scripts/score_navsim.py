"""CPU NAVSIM v1 official PDM scoring of exported predictions, in the NAVSIM environment.

No navpi/torch/model import required. Missing tokens/caches or scoring errors abort
instead of silently averaging only successful scenes. Use trusted metric caches.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np

_SCORING = None


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_coverage(predictions, manifest):
    expected = [row["token"] for row in manifest]
    actual = [row["token"] for row in predictions]
    if not expected or len(set(expected)) != len(expected) or actual != expected:
        raise ValueError("Prediction tokens must exactly match the complete manifest in order")
    for row in predictions:
        poses = np.asarray(row["trajectory"], dtype=np.float64)
        if poses.shape != (8, 3) or not np.isfinite(poses).all():
            raise ValueError(f"Invalid trajectory: {row['token']}")


def initialize(root, cache):
    global _SCORING
    sys.path.insert(0, root)
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    _SCORING = MetricCacheLoader(Path(cache)), sampling, PDMSimulator(sampling), PDMScorer(sampling)


def score_one(row):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    loader, sampling, simulator, scorer = _SCORING
    result = pdm_score(loader.get_from_token(row["token"]),
        Trajectory(np.asarray(row["trajectory"], np.float64), TrajectorySampling(num_poses=8, interval_length=.5)),
        sampling, simulator, scorer)
    fields = {key: float(value) for key, value in dataclasses.asdict(result).items()}
    if not all(np.isfinite(value) for value in fields.values()):
        raise FloatingPointError(f"Nonfinite PDM result: {row['token']}")
    return {"token": row["token"], **fields}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--navsim-root", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.workers < 1:
        p.error("workers must be positive")
    predictions = read_rows(a.predictions)
    check_coverage(predictions, read_rows(a.manifest))
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize, initargs=(str(a.navsim_root.resolve()), str(a.cache.resolve()))) as pool:
        with (a.output / "scores.jsonl").open("w") as stream:
            for row in pool.map(score_one, predictions, chunksize=16):
                records.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    means = {key: float(np.mean([r[key] for r in records])) for key in records[0] if key != "token"}
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    summary = dict(protocol="NAVSIM v1 PDM; not v2 EPDMS", scenes=len(records), failures=0,
        metrics=means, pdm_score_percent=100 * means["score"],
        manifest_sha256=digest(a.manifest), predictions_sha256=digest(a.predictions),
        scorer_sha256=digest(a.navsim_root / "navsim/evaluate/pdm_score.py"))
    (a.output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
