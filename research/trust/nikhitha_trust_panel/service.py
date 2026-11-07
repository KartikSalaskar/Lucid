import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .verdict import decide, LABELS

log = logging.getLogger(__name__)


def assess_image(img, stub=False):
    from app.services.imaging import quality
    from app.config import settings

    signals = dict(
        quality_score=quality(img)["score"],
        calibrated_confidence=None,
        uncertainty=None,
        ood_score=None,
        ood_flag=None,
    )

    flags = dict(temperature=False, ensemble=False, ood=False)
    errors = []

    cfg_path = Path("research/trust/trust_config.json")
    cfg = (
        json.loads(cfg_path.read_text())
        if cfg_path.is_file() and not stub
        else None
    )

    model_path = Path(settings.model_path)
    fingerprint = (
        hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_path.is_file()
        else None
    )

    if not stub and fingerprint:
        try:
            from app.ml.predict import predict_logits, extract_features
            from research.trust.chitthara_calibration.calibration import (
                softmax,
            )
            from research.trust.disha_ensemble.ensemble import (
                predict_with_uncertainty,
            )
            from research.trust.niharika_ood.mahalanobis import (
                mahalanobis_score,
            )

            temp = Path("trust_artifacts/temperature.json")
            if temp.is_file():
                t = json.loads(temp.read_text())
                if (
                    t.get("checkpoint_sha256") == fingerprint
                    and t["T"] > 0
                ):
                    signals["calibrated_confidence"] = float(
                        softmax(predict_logits([img]), t["T"]).max()
                    )
                    flags["temperature"] = True

            paths = [
                Path(f"trust_artifacts/ensemble/seed{i}.pt")
                for i in (1, 2, 3)
            ]

            hashes = [
                (
                    hashlib.sha256(p.read_bytes()).hexdigest()
                    if p.is_file()
                    else ""
                )
                for p in paths
            ]

            if (
                cfg is not None
                and hashes == cfg.get("ensemble_hashes")
                and all(hashes)
            ):
                signals["uncertainty"] = float(
                    predict_with_uncertainty([img])["std"][0]
                )
                flags["ensemble"] = True

            ood = Path("trust_artifacts/ood.npz")
            if ood.is_file():
                with np.load(ood) as z:
                    if str(z.get("checkpoint_sha256", "")) == fingerprint:
                        params = dict(
                            means=z["means"],
                            precision=z["precision"],
                        )
                        score = float(
                            mahalanobis_score(
                                extract_features([img]), params
                            )[0]
                        )

                        signals.update(
                            ood_score=score,
                            ood_flag=score > float(z["threshold"]),
                        )
                        flags["ood"] = True

        except Exception:
            log.exception("Trust calculation failed")
            errors.append("A trust calculation failed; see server log")

    if (
        cfg is None
        or cfg.get("checkpoint_sha256") != fingerprint
        or errors
    ):
        verdict = "unavailable"
        label = LABELS[verdict]
        reasons = errors + [
            "Trust configuration or matching artifacts unavailable"
        ]
    else:
        verdict, label, reasons = decide(signals, flags, cfg)

    return dict(
        verdict=verdict,
        label=label,
        reasons=reasons,
        signals=signals,
        artifacts=flags,
        disclaimer="Research and educational software only. Not a diagnosis.",
    )