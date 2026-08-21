#!/usr/bin/env python3
"""After the strong sweep: pull latest, pick a regularizer, full-visibility refit."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "outputs" / "3c391_regularization_sweep_strong"
FULLVIS = ROOT / "outputs" / "3c391_regularization_fullvis"
SELECTION = SWEEP / "selection.json"
SWEEP_PID = int(os.environ.get("SL1MJAX_SWEEP_PID", "139669"))


def _wait_for_sweep() -> None:
    print(f"waiting for sweep pid {SWEEP_PID}", flush=True)
    while True:
        try:
            os.kill(SWEEP_PID, 0)
        except OSError:
            break
        time.sleep(60)
    print("sweep process has exited", flush=True)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def _update_tree() -> dict[str, str]:
    status = _git("status", "--short").stdout
    print("pre-pull status:\n", status, flush=True)
    tracked = [
        "pyproject.toml",
        "uv.lock",
        "scripts/image_3c391_portable.py",
        "src/sl1mjax/imaging.py",
        "src/sl1mjax/inference.py",
        "src/sl1mjax/objective.py",
        "src/sl1mjax/sky.py",
        "src/sl1mjax/split.py",
    ]
    existing = [path for path in tracked if (ROOT / path).exists()]
    stash = _git("stash", "push", "-m", "bacchus-pre-pull", "--", *existing)
    print(stash.stdout, stash.stderr, flush=True)
    sweep_script = ROOT / "scripts" / "run_3c391_regularization_sweep.py"
    backup = Path("/tmp/sl1mjax-bacchus-backup")
    backup.mkdir(exist_ok=True)
    listed = _git("ls-files", "scripts/run_3c391_regularization_sweep.py")
    if sweep_script.exists() and not listed.stdout.strip():
        sweep_script.rename(backup / "run_3c391_regularization_sweep.py")
    pull = _git("pull", "--ff-only", "origin", "main")
    print(pull.stdout, pull.stderr, flush=True)
    pyproject = ROOT / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    old = '"jax>=0.4"'
    new = '"jax[cuda13]>=0.4"'
    restored_cuda = False
    if old in text and new not in text:
        pyproject.write_text(text.replace(old, new, 1), encoding="utf-8")
        restored_cuda = True
    head = _git("rev-parse", "--short", "HEAD").stdout.strip()
    return {"head": head, "restored_cuda": str(restored_cuda)}


def _finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _select() -> dict[str, object]:
    rows = list(csv.DictReader((SWEEP / "results.csv").open(encoding="utf-8")))
    candidates = []
    for row in rows:
        if row.get("status") != "ok" or row.get("split_strategy") != "random_row":
            continue
        train = _finite(row.get("casa_train_loss", ""))
        holdout = _finite(row.get("casa_holdout_loss", ""))
        if train is None or holdout is None or train <= 0:
            continue
        ratio = holdout / train
        step = _finite(row.get("casa_best_step", "")) or 0.0
        candidates.append(
            {
                "pixel_model": row["pixel_model"],
                "sparsity_weight": float(row["sparsity_weight"]),
                "smoothness_weight": float(row["smoothness_weight"]),
                "train_loss": train,
                "holdout_loss": holdout,
                "ratio": ratio,
                "best_step": step,
                "total_flux": _finite(row.get("casa_total_flux", "")) or 0.0,
            }
        )
    matched = [
        item
        for item in candidates
        if 0.7 <= item["ratio"] <= 1.3 and item["best_step"] >= 400
    ]
    pool = matched or candidates
    pool.sort(key=lambda item: (item["holdout_loss"], abs(item["ratio"] - 1.0)))
    if not pool:
        raise RuntimeError("no successful random_row cases in the strong sweep")
    chosen = pool[0]
    payload = {
        "chosen": chosen,
        "matched_count": len(matched),
        "candidate_count": len(candidates),
        "candidates": pool,
        "rule": (
            "random_row only; prefer holdout/train in [0.7, 1.3] and "
            "best_step >= 400; then lowest CASA-corrected holdout loss"
        ),
    }
    SELECTION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return chosen


def _run_fullvis(chosen: dict[str, object]) -> None:
    FULLVIS.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "image_3c391_portable.py"),
        str(ROOT / "outputs" / "3c391_imaging_fixture.zarr"),
        "--output",
        str(FULLVIS),
        "--size",
        "128",
        "--steps",
        "1000",
        "--learning-rate",
        "0.03",
        "--pixel-model",
        str(chosen["pixel_model"]),
        "--sparsity-weight",
        str(chosen["sparsity_weight"]),
        "--smoothness-weight",
        str(chosen["smoothness_weight"]),
        "--holdout-fraction",
        "0",
        "--split-strategy",
        "random_row",
        "--validation-interval",
        "20",
        "--precision",
        "float32",
        "--visibility-tile-size",
        "4096",
        "--pixel-tile-size",
        "4096",
    ]
    log_path = FULLVIS / "run.log"
    print("running", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"full-visibility refit failed; see {log_path}")


def _score_if_possible() -> None:
    casa_model = ROOT / "outputs" / "3c391_casa_imaging_128" / "3c391_c1_multiscale.model.fits"
    reconstruction = FULLVIS / "casa_corrected_reconstruction.fits"
    if not casa_model.exists() or not reconstruction.exists():
        print("skipping CASA visibility score; missing model or reconstruction")
        return
    output = ROOT / "outputs" / "3c391_casa_clean_visibility_score"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "score_3c391_casa_clean_visibilities.py"),
        str(ROOT / "outputs" / "3c391_imaging_fixture.zarr"),
        "--casa-model",
        str(casa_model),
        "--sl1mjax-reconstruction",
        str(reconstruction),
        "--output",
        str(output),
        "--holdout-fraction",
        "0",
    ]
    log_path = output / "run.log"
    output.mkdir(parents=True, exist_ok=True)
    print("running", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"CASA visibility score failed; see {log_path}")


def main() -> int:
    _wait_for_sweep()
    git_info = _update_tree()
    chosen = _select()
    print("selected", json.dumps(chosen, indent=2, sort_keys=True), flush=True)
    _run_fullvis(chosen)
    try:
        _score_if_possible()
    except RuntimeError as error:
        print(error, flush=True)
    report = {
        "git": git_info,
        "selection": json.loads(SELECTION.read_text(encoding="utf-8")),
        "fullvis_summary": (
            json.loads((FULLVIS / "summary.json").read_text(encoding="utf-8"))
            if (FULLVIS / "summary.json").exists()
            else None
        ),
    }
    (ROOT / "outputs" / "overnight_3c391_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("wrote outputs/overnight_3c391_report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
