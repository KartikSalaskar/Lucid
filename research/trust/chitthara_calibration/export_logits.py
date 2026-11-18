
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.ml.common import SplitDS, build_model, eval_tf, PREPROC_VERSION


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "models" / "classifier.pt"
SPLITS = ROOT / "data" / "processed" / "splits.json"
OUT = ROOT / "results" / "trust" / "chitthara"

EXPECTED_CLASSES = ["NORMAL", "PNEUMONIA"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_splits(splits, checkpoint):
    if splits.get("classes") != EXPECTED_CLASSES:
        raise ValueError("Unexpected class order")

    if checkpoint.get("classes") != EXPECTED_CLASSES:
        raise ValueError("Checkpoint class order mismatch")

    if checkpoint.get("preprocessing_version") != PREPROC_VERSION:
        raise ValueError("Preprocessing version mismatch")

    for name in ("val", "test"):
        if not splits.get(name):
            raise ValueError(f"Missing or empty {name} split")

    return True


def export_split(
    model,
    items,
    split_name,
    transform,
    device,
    fingerprint,
    splits_fingerprint,
    classes,
    preprocessing_version,
):
    loader = DataLoader(
        SplitDS(items, transform),
        batch_size=32,
        shuffle=False,
    )

    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    if not all_logits:
        raise ValueError(f"No data exported for {split_name}")

    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{split_name}.npz"

    np.savez_compressed(
        destination,
        logits=logits,
        labels=labels,
        split=np.array(split_name),
        checkpoint_sha256=np.array(fingerprint),
        splits_sha256=np.array(splits_fingerprint),
        classes=np.array(classes),
        preprocessing_version=np.array(preprocessing_version),
    )

    print(f"Saved {destination}")
    print(
        f"  split: {split_name}, examples: {len(labels)}, "
        f"logits shape: {logits.shape}"
    )


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    if not SPLITS.is_file():
        raise FileNotFoundError(f"Missing split file: {SPLITS}")

    # Load and hash the exact same checkpoint file.
    fingerprint = sha256(CHECKPOINT)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    with open(SPLITS, "r", encoding="utf-8") as f:
        splits = json.load(f)

    validate_splits(splits, checkpoint)

    splits_fingerprint = sha256(SPLITS)

    device = torch.device("cpu")

    model = build_model(
        checkpoint["arch"],
        len(checkpoint["classes"]),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["state"])
    model.to(device)

    transform = eval_tf(checkpoint["img_size"])

    export_split(
        model,
        splits["val"],
        "val",
        transform,
        device,
        fingerprint,
        splits_fingerprint,
        checkpoint["classes"],
        checkpoint["preprocessing_version"],
    )

    export_split(
        model,
        splits["test"],
        "test",
        transform,
        device,
        fingerprint,
        splits_fingerprint,
        checkpoint["classes"],
        checkpoint["preprocessing_version"],
    )


if __name__ == "__main__":
    main()
