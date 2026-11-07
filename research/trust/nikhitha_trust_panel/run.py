import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .verdict import decide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if not a.demo:
        ap.error("Use --demo; real use is through the web app")

    out = Path("results/trust/nikhitha/demo")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    Image.fromarray(
        rng.integers(30, 220, (128, 128), dtype="uint8")
    ).save(out / "synthetic.png")

    signals = dict(
        quality_score=None,
        calibrated_confidence=None,
        uncertainty=None,
        ood_score=None,
        ood_flag=None,
    )

    flags = dict(temperature=False, ensemble=False, ood=False)

    # Example cutoffs only, NEVER saved as the real trust_config.
    config = dict(
        quality_low=40,
        quality_caution=60,
        uncertainty_caution=0.15,
    )

    verdict, label, reasons = decide(signals, flags, config)

    response = dict(
        verdict=verdict,
        label=label,
        reasons=reasons,
        signals=signals,
        artifacts=flags,
        disclaimer="Research and educational software only. Not a diagnosis.",
    )

    (out / "sample_response.json").write_text(
        json.dumps(response, indent=2)
    )

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()