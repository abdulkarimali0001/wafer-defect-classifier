"""A compact CNN for multi-label wafer defect classification.

Design choices (good to be able to explain in an interview):
- 3 input channels (no die / pass / fail) instead of one grey image, so the
  network never confuses "outside the wafer" with "failed die".
- 4 conv blocks with BatchNorm, then global average pooling: small, fast on a CPU,
  and the last conv layer keeps spatial maps that Grad-CAM can visualize.
- 8 independent outputs with sigmoid (BCE loss), because one wafer can have
  several defect types at the same time.
"""
import torch
from torch import nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class WaferNet(nn.Module):
    def __init__(self, n_classes: int = 8, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            block(3, width), nn.MaxPool2d(2),            # 52 -> 26
            block(width, width * 2), nn.MaxPool2d(2),    # 26 -> 13
            block(width * 2, width * 4), nn.MaxPool2d(2),  # 13 -> 6
            block(width * 4, width * 8),                 # 6x6 maps, used by Grad-CAM
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width * 8, n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))  # raw logits; apply sigmoid for probabilities


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = WaferNet()
    print(m(torch.zeros(2, 3, 52, 52)).shape, f"{count_params(m):,} parameters")
