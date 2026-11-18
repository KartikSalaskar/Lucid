import numpy as np
from scipy.special import logsumexp


def softmax(logits, T=1.0):
    if T <= 0:
        raise ValueError("T must be positive")
    z = np.asarray(logits, float) / T
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def ece(probs, labels, n_bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    total = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(probs, labels):
    return float(
        ((probs - np.eye(probs.shape[1])[labels]) ** 2).sum(1).mean()
    )


def nll(logits, labels, T=1.0):
    z = np.asarray(logits, float) / T
    return float(
        (logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]).mean()
    )


def fit_temperature(val_logits, val_labels, split="val"):
    if split != "val":
        raise ValueError("Fit accepts validation data only")
    grid = np.geomspace(0.05, 10, 400)
    return float(min(grid, key=lambda t: nll(val_logits, val_labels, t)))


def report(logits, y, T):
    p = softmax(logits, T)
    return dict(
        ece=ece(p, y),
        brier=brier(p, y),
        nll=nll(logits, y, T),
        accuracy=float((p.argmax(1) == y).mean()),
    )


def reliability_diagram(probs, labels, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    xs, ys = [], []
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())

    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(xs, ys, "o-")
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability; n={len(labels)}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()