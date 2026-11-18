import hashlib
import json
import time

from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from . import explainability_service as xai


_c: dict = {}

DEMO = {
    "mode": "DEMO",
    "label": None,
    "confidence": None,
    "note": "No trained model found (or torch missing). Train one with scripts/train_classifier.py; no real inference is shown."
}


def _load():
    if "m" in _c:
        return _c["m"]

    model_path = Path(settings.model_path)

    if not model_path.is_file():
        return None

    try:
        import torch
    except ImportError:
        return None

    from ..ml.common import build_model

    ck = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    if ck["classes"] != ["NORMAL", "PNEUMONIA"]:
        raise ValueError("Expected NORMAL and PNEUMONIA classes")

    if ck["arch"] != "resnet18":
        raise ValueError("Expected ResNet18 model")

    if ck["preprocessing_version"] != "1.0":
        raise ValueError("Preprocessing version mismatch")

    # Calculate checkpoint SHA256
    checkpoint_hash = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()

    # Load temperature calibration
    temperature_path = Path(settings.temperature_path)

    if not temperature_path.is_file():
        raise FileNotFoundError(
            f"Temperature calibration file not found: {temperature_path}"
        )

    calibration = json.loads(
        temperature_path.read_text(encoding="utf-8")
    )

    temperature = float(calibration["T"])

    if temperature <= 0:
        raise ValueError("Invalid temperature value")

    # Verify calibration belongs to this checkpoint
    if calibration.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(
            "Temperature calibration checkpoint SHA256 mismatch"
        )

    model = build_model(
        ck["arch"],
        len(ck["classes"]),
        False
    )

    model.load_state_dict(ck["state"])
    model.eval()

    _c["m"] = (
        model,
        ck,
        temperature,
        checkpoint_hash
    )

    return _c["m"]


def model_available() -> bool:
    return _load() is not None


def predict(gray) -> dict:
    ld = _load()

    if ld is None:
        return dict(DEMO)

    import torch

    from PIL import Image
    from ..ml.common import eval_tf

    model, ck, temperature, checkpoint_hash = ld

    t0 = time.perf_counter()

    x = eval_tf(ck["img_size"])(
        Image.fromarray(gray)
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)

        # Apply temperature scaling
        p = torch.softmax(
            logits / temperature,
            dim=1
        )[0]

    i = int(p.argmax())

    t1 = time.perf_counter()

    cam = xai.gradcam_overlay(
        model,
        x,
        i,
        gray
    )

    return {
        "mode": "MODEL (research output, not a diagnosis)",

        "label": ck["classes"][i],

        "confidence": round(float(p[i]), 4),

        "probs": {
            c: round(float(v), 4)
            for c, v in zip(ck["classes"], p)
        },

        "gradcam": cam,

        "model": {
            "name": ck["arch"],
            "version": ck["version"],
            "trained_at": ck["trained_at"],
            "preprocessing_version": ck["preprocessing_version"],
            "inference_at": datetime.now(timezone.utc).isoformat(),

            "calibration": {
                "method": "temperature_scaling",
                "temperature": temperature,
                "checkpoint_sha256": checkpoint_hash
            }
        },

        "infer_s": t1 - t0,

        "explain_s": time.perf_counter() - t1
    }