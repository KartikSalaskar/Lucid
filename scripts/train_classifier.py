"""PYTHONPATH=backend python scripts/train_classifier.py [--arch resnet18 --epochs 10 ...]"""
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from app.ml.common import PREPROC_VERSION, SplitDS, build_model, eval_tf, train_tf

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="data/processed/splits.json"); ap.add_argument("--arch", default="resnet18")
ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--size", type=int, default=224); ap.add_argument("--out", default="models/classifier.pt")
ap.add_argument("--version", default="1.0.0"); ap.add_argument("--no-pretrained", action="store_true")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
import random
import numpy as np
random.seed(a.seed)
np.random.seed(a.seed)
torch.manual_seed(a.seed)
torch.use_deterministic_algorithms(True)
print("seed", a.seed)

S = json.loads(Path(a.splits).read_text()); K = len(S["classes"])
dl = lambda k, tf, sh: DataLoader(SplitDS(S[k], tf), a.bs, shuffle=sh)
tr, va = dl("train", train_tf(a.size), True), dl("val", eval_tf(a.size), False)
cnt = torch.bincount(torch.tensor([i["label"] for i in S["train"]]), minlength=K).float()
loss_fn = nn.CrossEntropyLoss(weight=cnt.sum() / (K * cnt.clamp(min=1)))  # imbalance handling
model = build_model(a.arch, K, not a.no_pretrained)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

def run(loader, train):
    model.train(train); tot, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            o = model(x); l = loss_fn(o, y)
            if train: opt.zero_grad(); l.backward(); opt.step()
            tot += l.item() * len(y); ys += y.tolist(); ps += o.argmax(1).tolist()
    return tot / len(ys), accuracy_score(ys, ps), f1_score(ys, ps, average="macro")

hist, best, bad = [], 1e9, 0
for ep in range(a.epochs):
    (tl, ta, _), (vl, vac, vf) = run(tr, True), run(va, False); sched.step()
    hist.append({"epoch": ep + 1, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": vac, "val_f1": vf})
    print(hist[-1])
    if vl < best:
        best, bad = vl, 0
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "arch": a.arch, "classes": S["classes"], "img_size": a.size,
                    "seed": a.seed, "version": a.version, "preprocessing_version": PREPROC_VERSION,
                    "trained_at": datetime.now(timezone.utc).isoformat()}, a.out)
    else:
        bad += 1
        if bad >= a.patience: print("early stop"); break
Path(a.out).with_suffix(".history.json").write_text(json.dumps(hist, indent=1))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for k in ("loss", "acc"):
        plt.figure(); [plt.plot([h[f"{p}_{k}"] for h in hist], label=f"{p}_{k}") for p in ("train", "val")]
        plt.legend(); plt.savefig(Path(a.out).with_name(f"history_{k}.png")); plt.close()
except ImportError: pass
