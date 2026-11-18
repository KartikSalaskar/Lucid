
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.ml.common import PREPROC_VERSION

from .calibration import (
    softmax,
    fit_temperature,
    report,
    reliability_diagram,
)
from research.trust.eval_signal import signal_error_auc


EXPECTED_CLASSES = ["NORMAL", "PNEUMONIA"]


def validate_metadata(val, test):
    required = (
        "split",
        "checkpoint_sha256",
        "splits_sha256",
        "classes",
        "preprocessing_version",
    )

    for name, data in (("validation", val), ("test", test)):
        available_keys = data.files if hasattr(data, "files") else data.keys()

        for key in required:
            if key not in available_keys:
                raise ValueError(f"Missing {key} in {name} data")

    if str(val["split"]) != "val":
        raise ValueError("Expected validation split")

    if str(test["split"]) != "test":
        raise ValueError("Expected test split")

    if str(val["checkpoint_sha256"]) != str(test["checkpoint_sha256"]):
        raise ValueError("Checkpoint mismatch")

    if str(val["splits_sha256"]) != str(test["splits_sha256"]):
        raise ValueError("Split file hash mismatch")

    for name, data in (("validation", val), ("test", test)):
        if data["classes"].tolist() != EXPECTED_CLASSES:
            raise ValueError(f"Unexpected class order in {name} data")

        if str(data["preprocessing_version"]) != str(PREPROC_VERSION):
            raise ValueError(
                f"Preprocessing version mismatch in {name} data"
            )

    return True


def validate_external(data, fingerprint, metadata):
    required = (
        "checkpoint_sha256",
        "splits_sha256",
        "classes",
        "preprocessing_version",
    )

    for key in required:
        if key not in data.files:
            raise ValueError(f"Missing {key} in external data")

    if str(data["checkpoint_sha256"]) != fingerprint:
        raise ValueError("External checkpoint mismatch")

    if str(data["splits_sha256"]) != metadata["splits_sha256"]:
        raise ValueError("External split file hash mismatch")

    if data["classes"].tolist() != EXPECTED_CLASSES:
        raise ValueError("Unexpected class order in external data")

    if str(data["preprocessing_version"]) != str(PREPROC_VERSION):
        raise ValueError("External preprocessing version mismatch")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--val")
    ap.add_argument("--test")
    ap.add_argument("--external")

    a = ap.parse_args()

    out = Path("results/trust/chitthara") / (
        "demo" if a.demo else "real"
    )
    out.mkdir(parents=True, exist_ok=True)

    if a.demo:
        rng = np.random.default_rng(42)

        def sample():
            z = rng.normal(size=(4000, 2)) * 3
            y = (
                rng.random(len(z)) < softmax(z, 2)[:, 1]
            ).astype(int)
            return z, y

        zv, yv = sample()
        zt, yt = sample()

        fingerprint = "synthetic"
        metadata = {
            "splits_sha256": "synthetic",
            "classes": EXPECTED_CLASSES,
            "preprocessing_version": PREPROC_VERSION,
        }

    else:
        if not a.val or not a.test:
            ap.error("Need --val and --test input NPZs")

        with np.load(a.val) as v, np.load(a.test) as t:
            validate_metadata(v, t)

            fingerprint = str(v["checkpoint_sha256"])
            metadata = {
                "splits_sha256": str(v["splits_sha256"]),
                "classes": v["classes"].tolist(),
                "preprocessing_version": str(v["preprocessing_version"]),
            }

            zv, yv = v["logits"], v["labels"]
            zt, yt = t["logits"], t["labels"]

    T = fit_temperature(zv, yv)

    before = report(zt, yt, 1)
    after = report(zt, yt, T)

    wrong = (zt.argmax(1) != yt).astype(int)

    result = dict(
        T=T,
        before=before,
        after=after,
        error_auc_before=signal_error_auc(
            1 - softmax(zt).max(1), wrong
        ),
        error_auc_after=signal_error_auc(
            1 - softmax(zt, T).max(1), wrong
        ),
        random_auc=signal_error_auc(
            np.random.default_rng(42).random(len(yt)), wrong
        ),
    )

    for name, temp in [("before", 1), ("after", T)]:
        reliability_diagram(
            softmax(zt, temp),
            yt,
            out / f"reliability_{name}.png",
        )

    if a.external:
        with np.load(a.external) as e:
            validate_external(e, fingerprint, metadata)

            result["external"] = report(
                e["logits"],
                e["labels"],
                T,
            )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()

    meta = dict(
        author="chitthara",
        date=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        seed=42,
        checkpoint=fingerprint,
        dataset="synthetic" if a.demo else "kermany_test",
        status="demo" if a.demo else "preliminary",
        n=len(yt),
        **metadata,
    )

    (out / "calibration.json").write_text(
        json.dumps(
            dict(meta=meta, results=result),
            indent=2,
        )
    )

    if not a.demo:
        dest = Path("trust_artifacts")
        dest.mkdir(exist_ok=True)

        (dest / "temperature.json").write_text(
            json.dumps(
                dict(
                    T=T,
                    fit_on="kermany_val",
                    seed=42,
                    date=meta["date"],
                    checkpoint_sha256=fingerprint,
                    **metadata,
                ),
                indent=2,
            )
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
