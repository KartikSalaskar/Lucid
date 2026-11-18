
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records")
    ap.add_argument("--source", default="kermany")
    ap.add_argument(
        "--split", choices=["train", "val", "test"], default="val"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="models/classifier.pt")
    a = ap.parse_args()

    from app.ml.predict import predict_logits, predict_proba, extract_features
    from app.ml.common import PREPROC_VERSION
    from app.services.imaging import quality

    checkpoint_path = Path(a.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    split_path = Path("data/processed/splits.json")
    S = json.loads(split_path.read_text())

    classes = S["classes"]
    if classes != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Unexpected class order; ask Nikhil")

    records = (
        json.loads(Path(a.records).read_text()) if a.records else S[a.split]
    )
    if not records:
        raise ValueError("Empty records")

    for r in records:
        if not Path(r["path"]).is_file():
            raise FileNotFoundError("Missing image path: " + r["path"])

    arrays = dict(logits=[], probs=[], features=[], quality=[])

    for start in range(0, len(records), 32):
        batch = records[start : start + 32]
        images = []

        for r in batch:
            with Image.open(r["path"]) as im:
                images.append(np.array(im.convert("L")))

        arrays["logits"].append(
            predict_logits(images, checkpoint_path=str(checkpoint_path))
        )
        arrays["probs"].append(
            predict_proba(images, checkpoint_path=str(checkpoint_path))
        )
        arrays["features"].append(
            extract_features(images, checkpoint_path=str(checkpoint_path))
        )
        arrays["quality"].extend(quality(im)["score"] for im in images)

        print(start + len(batch), "/", len(records))

    payload = {
        k: np.concatenate(v) for k, v in arrays.items() if k != "quality"
    }
    payload["quality"] = np.asarray(arrays["quality"], float)

    if payload["features"].shape != (len(records), 512):
        raise ValueError("Expected ResNet18 features [N,512]")

    if not np.allclose(payload["probs"].sum(1), 1):
        raise ValueError("Invalid probabilities")

    payload.update(
        labels=np.array([r["label"] for r in records], int),
        source=a.source,
        split=a.split,
        checkpoint_sha256=sha(checkpoint_path),
        splits_sha256=sha(split_path),
        classes=np.array(classes),
        preprocessing_version=PREPROC_VERSION,
    )

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **payload)

    print("Saved private input:", dest)


if __name__ == "__main__":
    main()
