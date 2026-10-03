"""Grad-CAM: show which dies the model looked at for each predicted defect.

Usage:
    python src/gradcam.py --n 8          # saves results/gradcam.png from test-set maps

Engineers need to trust a model before using it on a production line. Grad-CAM
heatmaps let you check that "Edge_Ring" is predicted because of failing dies on the
edge, not because of something irrelevant.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import CLASSES, load_npz, one_hot_maps, pattern_name, stratified_split
from model import WaferNet
from train import ROOT, pick_device, synthetic


def gradcam(model, x, class_idx):
    """Heatmap (52x52, 0-1) of evidence for class_idx on a single map x (1,3,52,52)."""
    acts = {}
    h = model.features.register_forward_hook(lambda m, i, o: acts.update(a=o))
    model.zero_grad()
    logit = model(x)[0, class_idx]
    a = acts["a"]; h.remove()
    g = torch.autograd.grad(logit, a)[0]
    cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "Wafer_Map_Datasets.npz"))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    maps, labels = synthetic() if args.synthetic else load_npz(args.data)
    _, _, te = stratified_split(labels)
    rng = np.random.default_rng(1)
    pick = rng.choice(te[labels[te].sum(1) > 0], size=args.n, replace=False)  # defective maps only

    model = WaferNet().to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "wafernet.pt", map_location=device))
    model.eval()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    wafer_cmap = ListedColormap(["white", "#c9d6df", "#b2182b"])  # no die / pass / fail
    fig, ax = plt.subplots(2, args.n, figsize=(2.1 * args.n, 4.6))
    for j, i in enumerate(pick):
        x = torch.from_numpy(one_hot_maps(maps[i:i + 1])).to(device)
        prob = torch.sigmoid(model(x))[0].detach().cpu().numpy()
        top = int(prob.argmax())
        cam = gradcam(model, x, top)
        ax[0, j].imshow(maps[i], cmap=wafer_cmap, vmin=0, vmax=2)
        ax[0, j].set_title(f"true: {pattern_name(labels[i])}", fontsize=7)
        ax[1, j].imshow(maps[i], cmap=wafer_cmap, vmin=0, vmax=2)
        ax[1, j].imshow(cam, cmap="jet", alpha=0.45)
        ax[1, j].set_title(f"pred: {pattern_name(prob >= 0.5)}\nCAM for {CLASSES[top]}", fontsize=7)
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle("Wafer maps (red = failed die) and Grad-CAM evidence", fontsize=10)
    fig.tight_layout()
    out = ROOT / "results" / "gradcam.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
