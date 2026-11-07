import hashlib, json, logging, time
from collections import Counter
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .database.db import Analysis, Session
from .services import imaging, inference_service
from .services.imaging import png64
from .routers.trust import router as trust_router

log = logging.getLogger("api")
app = FastAPI(title="Medical Imaging Analysis & Quality Platform",
              description="Research/educational only. Not a diagnostic system.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(trust_router)

def _json(path: str):
    p = Path(path)
    if not p.exists(): raise HTTPException(404, "not available yet - run the dataset/training scripts")
    return json.loads(p.read_text())

@app.get("/api/health")
def health():
    return {"status": "ok", "demo_mode": not inference_service.model_available()}

@app.get("/api/dataset/statistics")
def dataset_stats(): return _json(settings.dataset_stats_path)

@app.get("/api/model/metrics")
def model_metrics(): return _json(settings.metrics_path)

@app.get("/api/analytics/summary")
def summary():
    with Session() as s:
        rows = s.query(Analysis).order_by(Analysis.id.desc()).limit(200).all()
    return {"count": len(rows), "quality": [r.quality for r in rows],
            "confidence": [r.confidence for r in rows if r.confidence is not None],
            "predictions": dict(Counter(r.prediction for r in rows if r.prediction)),
            "total_ms": [r.total_ms for r in rows]}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), clahe: bool = True):
    data, t = await file.read(), {}
    if len(data) > settings.max_upload_mb * 2**20:
        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
    try:
        t0 = time.perf_counter()
        img, meta = imaging.load(data, file.filename or "")
        issues = imaging.validate(img)
        t["load_validate"] = time.perf_counter() - t0
        if issues: raise HTTPException(422, "Invalid: " + "; ".join(issues))
        t0 = time.perf_counter(); proc = imaging.preprocess(img, clahe); t["preprocess"] = time.perf_counter() - t0
        t0 = time.perf_counter(); q = imaging.quality(img); t["quality"] = time.perf_counter() - t0
        try:
            pred = inference_service.predict(img)  # model sees the unmodified image (matches training)
        except Exception:
            log.exception("inference failed")
            pred = {"mode": "ERROR", "label": None, "confidence": None, "note": "Inference failed; see server logs."}
        t["inference"], t["explain"] = pred.pop("infer_s", 0), pred.pop("explain_s", 0)
        out = {"sha256": hashlib.sha256(data).hexdigest()[:16], "shape": list(img.shape), "meta": meta,
               "quality": q, "stats": imaging.stats(img), "prediction": pred,
               "original": png64(img), "processed": png64(proc),
               "timings_ms": {k: round(v * 1000, 1) for k, v in t.items()}}
        try:
            with Session() as s:
                s.add(Analysis(sha256=out["sha256"], modality=meta.get("Modality", meta.get("Format")),
                               height=img.shape[0], width=img.shape[1], quality=q["score"],
                               model_version=(pred.get("model") or {}).get("version"), prediction=pred.get("label"),
                               confidence=pred.get("confidence"), total_ms=sum(t.values()) * 1000))
                s.commit()
        except Exception:
            log.exception("db write failed")
        return out
    except HTTPException: raise
    except ValueError as e: raise HTTPException(422, str(e))
    except Exception:
        log.exception("analysis failed")
        raise HTTPException(500, "Processing failed. See server logs.")
