"""Train and evaluate WaferNet on MixedWM38.

Usage:
    python src/train.py --data data/Wafer_Map_Datasets.npz --epochs 20
    python src/train.py --synthetic --epochs 1      # quick smoke test, no dataset needed

Outputs (in results/):
    metrics.json         macro-F1, exact-match accuracy, per-class scores
    per_class.csv        precision / recall / F1 for each defect type
    per_pattern.csv      exact-match accuracy for each of the 38 patterns
    training_curve.png   train/val loss and val macro-F1 by epoch
and models/wafernet.pt (best checkpoint by validation macro-F1).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from data import CLASSES, WaferDataset, load_npz, pattern_name, stratified_split
from model import WaferNet, count_params

ROOT = Path(__file__).resolve().parent.parent


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


def synthetic(n=600, seed=0):
    """Fake wafer maps for testing the pipeline: a disc of passing dies, with
    failing dies painted in the center (Center) or along a line (Scratch)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:52, :52]
    disc = (yy - 25.5) ** 2 + (xx - 25.5) ** 2 < 25 ** 2
    maps = np.where(disc, 1, 0)[None].repeat(n, 0).astype(np.uint8)
    labels = np.zeros((n, 8), np.float32)
    for i in range(n):
        if rng.random() < 0.5:
            maps[i][((yy - 25.5) ** 2 + (xx - 25.5) ** 2 < 49) & disc] = 2; labels[i, 0] = 1
        if rng.random() < 0.5:
            r = rng.integers(10, 40); maps[i][r, 10:42] = 2; labels[i, 6] = 1
    return maps, labels


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu()); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "Wafer_Map_Datasets.npz"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    maps, labels = synthetic() if args.synthetic else load_npz(args.data)
    tr, va, te = stratified_split(labels, seed=args.seed)
    print(f"{len(labels):,} wafer maps -> train {len(tr):,} / val {len(va):,} / test {len(te):,} | device: {device}")

    dl = lambda idx, aug, shuf: DataLoader(WaferDataset(maps[idx], labels[idx], aug), batch_size=args.batch, shuffle=shuf)
    train_dl, val_dl, test_dl = dl(tr, True, True), dl(va, False, False), dl(te, False, False)

    model = WaferNet().to(device)
    print(f"WaferNet: {count_params(model):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    loss_fn = nn.BCEWithLogitsLoss()

    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "results").mkdir(exist_ok=True)
    ckpt = ROOT / "models" / "wafernet.pt"
    history, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(y)
        p, y = predict(model, val_dl, device)
        val_loss = float(nn.functional.binary_cross_entropy(torch.tensor(p), torch.tensor(y)))
        present = np.where(y.sum(0) > 0)[0]  # only score defect types that appear
        val_f1 = f1_score(y, p >= args.threshold, average="macro", labels=present, zero_division=0)
        history.append({"epoch": ep, "train_loss": total / len(tr), "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"epoch {ep:2d} | train loss {total / len(tr):.4f} | val loss {val_loss:.4f} | val macro-F1 {val_f1:.4f} | {time.time() - t0:.0f}s")
        if val_f1 > best:
            best = val_f1; torch.save(model.state_dict(), ckpt)

    # ---- Final evaluation on the held-out test set, using the best checkpoint ----
    model.load_state_dict(torch.load(ckpt, map_location=device))
    p, y = predict(model, test_dl, device)
    pred = (p >= args.threshold).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, zero_division=0)
    exact = float((pred == y).all(axis=1).mean())

    names = np.array([pattern_name(r) for r in y])
    rows = []
    for name in sorted(set(names)):
        m = names == name
        rows.append((name, int(m.sum()), float((pred[m] == y[m]).all(axis=1).mean())))

    metrics = {
        "test_maps": int(len(y)),
        "macro_f1": float(f1[sup > 0].mean()),
        "exact_match_accuracy": exact,
        "per_class": {c: {"precision": float(a), "recall": float(b), "f1": float(d), "support": int(s)}
                      for c, a, b, d, s in zip(CLASSES, prec, rec, f1, sup)},
        "epochs": args.epochs, "parameters": count_params(model), "device": device,
    }
    out = ROOT / "results"
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "per_class.csv").write_text("class,precision,recall,f1,support\n" + "".join(
        f"{c},{a:.4f},{b:.4f},{d:.4f},{s}\n" for c, a, b, d, s in zip(CLASSES, prec, rec, f1, sup)))
    (out / "per_pattern.csv").write_text("pattern,test_maps,exact_match_accuracy\n" + "".join(
        f"{n},{k},{a:.4f}\n" for n, k, a in rows))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train"); ax[0].plot(ep, [h["val_loss"] for h in history], label="validation")
    ax[0].set(title="Loss (BCE)", xlabel="epoch"); ax[0].legend()
    ax[1].plot(ep, [h["val_macro_f1"] for h in history], color="tab:green")
    ax[1].set(title="Validation macro-F1", xlabel="epoch", ylim=(0, 1))
    fig.tight_layout(); fig.savefig(out / "training_curve.png", dpi=150)

    print(f"\nTEST  macro-F1 {metrics['macro_f1']:.4f} | exact-match accuracy {exact:.4f}")
    for c, d in metrics["per_class"].items():
        print(f"  {c:10s} P {d['precision']:.3f}  R {d['recall']:.3f}  F1 {d['f1']:.3f}  (n={d['support']})")


if __name__ == "__main__":
    main()
