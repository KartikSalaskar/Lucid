import math


def calculate_uncertainty(probabilities: dict) -> dict:
    """
    Calculate prediction uncertainty using normalized entropy.
    """

    values = list(probabilities.values())

    if not values or len(values) < 2:
        return {
            "entropy": None,
            "uncertainty": None,
            "level": "UNAVAILABLE"
        }

    total = sum(values)

    if total <= 0:
        return {
            "entropy": None,
            "uncertainty": None,
            "level": "UNAVAILABLE"
        }

    values = [p / total for p in values]

    entropy = -sum(
        p * math.log(p)
        for p in values if p > 0
    )

    max_entropy = math.log(len(values))
    uncertainty = entropy / max_entropy

    if uncertainty < 0.3:
        level = "LOW"
    elif uncertainty < 0.7:
        level = "MODERATE"
    else:
        level = "HIGH"

    return {
        "entropy": round(entropy, 4),
        "uncertainty": round(uncertainty, 4),
        "level": level
    }