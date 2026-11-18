import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

c = TestClient(app)


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, "PNG")
    return b.getvalue()


def test_health():
    assert c.get("/api/health").json()["status"] == "ok"


def test_ok():
    a = (np.random.rand(256, 256) * 200 + 20).astype("uint8")

    r = c.post(
        "/api/analyze",
        files={"file": ("x.png", png(a))}
    )

    assert r.status_code == 200

    result = r.json()
    prediction = result["prediction"]

    assert 0 <= result["quality"]["score"] <= 100

    if prediction["mode"] == "DEMO":
        assert prediction["label"] is None
        assert prediction["confidence"] is None

    elif prediction["mode"].startswith("MODEL"):
        assert prediction["label"] in ["NORMAL", "PNEUMONIA"]
        assert 0 <= prediction["confidence"] <= 1

        calibration = prediction["model"]["calibration"]

        assert calibration["method"] == "temperature_scaling"
        assert calibration["temperature"] > 0

    else:
        raise AssertionError(f"Unexpected prediction mode: {prediction['mode']}")


def test_empty():
    assert c.post(
        "/api/analyze",
        files={"file": ("x.png", png(np.zeros((128, 128), "uint8")))}
    ).status_code == 422


def test_bad_ext():
    assert c.post(
        "/api/analyze",
        files={"file": ("x.exe", b"hi")}
    ).status_code == 422


def test_corrupt():
    assert c.post(
        "/api/analyze",
        files={"file": ("x.dcm", b"junk")}
    ).status_code == 422