LABELS = dict(
    ok="No reliability flags raised",
    caution="Caution: lower reliability",
    low_reliability="Low reliability: unlike training data or poor image",
    unavailable="Unavailable: trust artifacts not found",
)


def decide(signals, artifacts, config):
    reasons = []
    rank = 0

    q = signals.get("quality_score")
    if q is not None and q < config["quality_low"]:
        rank = 2
        reasons.append(f"Image quality score {q} out of 100 (very low)")
    elif q is not None and q < config["quality_caution"]:
        rank = 1
        reasons.append(f"Image quality score {q} out of 100 (low)")

    if signals.get("ood_flag") is True:
        rank = 2
        reasons.append("Image looks unlike the training data")

    unc = signals.get("uncertainty")
    if unc is not None and unc > config["uncertainty_caution"]:
        rank = max(rank, 1)
        reasons.append("The ensemble members disagree")

    missing = [k for k, v in artifacts.items() if not v]
    if missing:
        return (
            "unavailable",
            LABELS["unavailable"],
            reasons + ["Unavailable or incompatible: " + ", ".join(missing)],
        )

    key = ["ok", "caution", "low_reliability"][rank]
    return key, LABELS[key], reasons or ["No reliability flags raised"]
