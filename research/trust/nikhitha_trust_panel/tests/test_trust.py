import io

import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.trust import router
from research.trust.nikhitha_trust_panel.verdict import decide


CFG = dict(
    quality_low=40,
    quality_caution=60,
    uncertainty_caution=0.15,
)


def test_verdicts():
    all_ok = dict(temperature=True, ensemble=True, ood=True)
    s = dict(quality_score=80, uncertainty=0.02, ood_flag=False)

    assert decide(s, all_ok, CFG)[0] == "ok"
    assert decide(dict(s, quality_score=50), all_ok, CFG)[0] == "caution"
    assert decide(dict(s, ood_flag=True), all_ok, CFG)[0] == "low_reliability"
    assert decide(s, dict(all_ok, ood=False), CFG)[0] == "unavailable"


def test_upload_and_stub(monkeypatch):
    monkeypatch.setenv("LUCID_TRUST_STUB", "1")

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    rng = np.random.default_rng(42)
    img = Image.fromarray(
        rng.integers(30, 220, (128, 128), dtype="uint8")
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    r = client.post(
        "/api/trust/assess",
        files={"file": ("demo.png", buf.getvalue(), "image/png")},
    )

    assert r.status_code == 200
    assert r.json()["verdict"] == "unavailable"
    assert r.json()["signals"]["uncertainty"] is None

    r = client.post(
        "/api/trust/assess",
        files={"file": ("bad.txt", b"bad", "text/plain")},
    )

    assert r.status_code == 422