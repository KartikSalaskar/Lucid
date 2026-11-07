from . import explainability_service as xai
from . import trust_service
import time
from datetime import datetime, timezone
from pathlib import Path
from ..config import settings
from . import explainability_service as xai

_c: dict = {}
DEMO = {"mode": "DEMO", "label": None, "confidence": None,
        "note": "No trained model found (or torch missing). Train one with scripts/train_classifier.py; no real inference is shown."}

def _load():
    if "m" in _c: return _c["m"]
    if not Path(settings.model_path).exists(): return None
    try: import torch
    except ImportError: return None
    from ..ml.common import build_model
    ck = torch.load(settings.model_path, map_location="cpu")
    m = build_model(ck["arch"], len(ck["classes"]), False)
    m.load_state_dict(ck["state"]); m.eval()
    _c["m"] = (m, ck)
    return _c["m"]

def model_available() -> bool: return _load() is not None

def predict(gray) -> dict:
    ld = _load()
    if ld is None: return dict(DEMO)
    import torch
    from PIL import Image
    from ..ml.common import eval_tf
    m, ck = ld
    t0 = time.perf_counter()
    x = eval_tf(ck["img_size"])(Image.fromarray(gray)).unsqueeze(0)
    with torch.no_grad(): p = torch.softmax(m(x), 1)[0]
    i = int(p.argmax()); t1 = time.perf_counter()
    uncertainty = trust_service.calculate_uncertainty(
    {c: float(v) for c, v in zip(ck["classes"], p)}
)
    cam = xai.gradcam_overlay(m, x, i, gray)
    return {"mode": "MODEL (research output, not a diagnosis)", "label": ck["classes"][i],
            "confidence": round(float(p[i]), 4),
"uncertainty": uncertainty,
            "probs": {c: round(float(v), 4) for c, v in zip(ck["classes"], p)}, "gradcam": cam,
            "model": {"name": ck["arch"], "version": ck["version"], "trained_at": ck["trained_at"],
                      "preprocessing_version": ck["preprocessing_version"],
                      "inference_at": datetime.now(timezone.utc).isoformat()},
            "infer_s": t1 - t0, "explain_s": time.perf_counter() - t1}
