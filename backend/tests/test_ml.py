
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_split_has_no_patient_leakage():
    d = tempfile.mkdtemp()

    sh = lambda *c: subprocess.run(
        [sys.executable, *c],
        cwd=ROOT,
        check=True,
        capture_output=True
    )

    sh("scripts/generate_sample_data.py", f"{d}/raw", "20")
    sh(
        "scripts/prepare_dataset.py",
        "--raw", f"{d}/raw",
        "--out", f"{d}/out"
    )

    s = json.loads(Path(f"{d}/out/splits.json").read_text())

    pid = lambda k: {
        Path(i["path"]).stem.split("_")[0]
        for i in s[k]
    }

    assert not (pid("train") & pid("test"))
    assert not (pid("train") & pid("val"))
    assert not (pid("val") & pid("test"))


def test_gradcam():
    torch = pytest.importorskip("torch")

    from app.ml.common import build_model
    from app.services.explainability_service import gradcam_overlay

    m = build_model("resnet18", 2, False).eval()

    out = gradcam_overlay(
        m,
        torch.randn(1, 3, 224, 224),
        0,
        np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    )

    assert out.startswith("data:image/png;base64,")


def test_temperature_scaled_probabilities():
    torch = pytest.importorskip("torch")

    from PIL import Image
    from app.services import inference_service
    from app.ml.common import eval_tf

    loaded = inference_service._load()
    assert loaded is not None

    model, ck, temperature, checkpoint_hash = loaded

    rng = np.random.default_rng(42)
    gray = rng.integers(20, 220, (256, 256), dtype=np.uint8)

    x = eval_tf(ck["img_size"])(
        Image.fromarray(gray)
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        expected = torch.softmax(logits / temperature, dim=1)[0]

    result = inference_service.predict(gray)

    assert result["model"]["calibration"]["method"] == "temperature_scaling"

    assert (
        result["model"]["calibration"]["checkpoint_sha256"]
        == checkpoint_hash
    )

    for i, class_name in enumerate(ck["classes"]):
        assert abs(
            result["probs"][class_name] - float(expected[i])
        ) < 0.00006
