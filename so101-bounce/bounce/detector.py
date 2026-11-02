"""Ball detector: a small fully convolutional network that outputs a heatmap of the ball centre (CenterNet-style)
at 1/4 resolution, plus a sub-pixel offset and the ball radius. ~0.2 M parameters, meant to run in a few ms on a
MacBook CPU for 320x240 frames.

Training targets: a Gaussian bump at the ball centre (only when at least 30 % of the ball is visible); images with
no visible ball have an all-zero heatmap, so the network learns to say "no ball" for hands, other objects and empty
scenes. Loss: penalty-reduced focal loss on the heatmap, L1 on offset and radius at the centre cell.
"""
import torch
from torch import nn
import torch.nn.functional as F

STRIDE = 4
MIN_VISIBLE = 0.3


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class BallNet(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        w = width
        self.s1 = block(3, w, 2)            # 1/2
        self.s2 = block(w, 2 * w, 2)        # 1/4
        self.s3 = block(2 * w, 4 * w, 2)    # 1/8
        self.s4 = block(4 * w, 4 * w, 2)    # 1/16: context (is this round orange thing the ball or a hand?)
        self.up4 = nn.Conv2d(4 * w, 4 * w, 1)
        self.up3 = nn.Conv2d(4 * w, 2 * w, 1)
        self.fuse = block(2 * w, 2 * w)
        self.head = nn.Conv2d(2 * w, 4, 1)  # heatmap logit, dx, dy, radius
        nn.init.constant_(self.head.bias[:1], -4.0)

    def forward(self, x):
        x2 = self.s2(self.s1(x))
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        y3 = x3 + F.interpolate(self.up4(x4), size=x3.shape[-2:], mode="nearest")
        y2 = x2 + F.interpolate(self.up3(y3), size=x2.shape[-2:], mode="nearest")
        return self.head(self.fuse(y2))


def targets(labels, h, w, device):
    """labels (B, 6): u, v, radius, visible, in_frame, camera  ->  heatmap (B,1,h,w), offset/radius targets, mask."""
    B = labels.shape[0]
    u, v, r = labels[:, 0] / STRIDE, labels[:, 1] / STRIDE, labels[:, 2]
    pos = (labels[:, 4] > 0) & (labels[:, 3] >= MIN_VISIBLE) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ys = torch.arange(h, device=device).view(1, h, 1)
    xs = torch.arange(w, device=device).view(1, 1, w)
    sig = (r / STRIDE / 2).clamp(min=0.7).view(B, 1, 1)
    hm = torch.exp(-((xs - u.view(B, 1, 1)) ** 2 + (ys - v.view(B, 1, 1)) ** 2) / (2 * sig ** 2))
    hm = hm * pos.view(B, 1, 1)
    ci, cj = v.floor().long().clamp(0, h - 1), u.floor().long().clamp(0, w - 1)
    return hm.unsqueeze(1), torch.stack([u - cj, v - ci, r / 10], -1), ci, cj, pos


def loss_fn(out, labels):
    B, _, h, w = out.shape
    hm_t, reg_t, ci, cj, pos = targets(labels, h, w, out.device)
    p = torch.sigmoid(out[:, :1]).clamp(1e-4, 1 - 1e-4)
    peak = hm_t > 0.99
    pos_l = -((1 - p) ** 2 * torch.log(p))[peak].sum()
    neg_l = -(((1 - hm_t) ** 4) * p ** 2 * torch.log(1 - p))[~peak].sum()
    focal = (pos_l + neg_l) / max(int(peak.sum()), 1)
    b = torch.arange(B, device=out.device)
    reg = out[b, 1:4, ci, cj]
    reg_l = (F.l1_loss(reg[pos], reg_t[pos]) if bool(pos.any()) else out.sum() * 0)
    return focal + reg_l, {"focal": float(focal), "reg": float(reg_l)}


@torch.no_grad()
def decode(out, threshold: float = 0.3):
    """Best detection per image: (u, v, radius, confidence) in input pixels; confidence < threshold = no ball."""
    B, _, h, w = out.shape
    p = torch.sigmoid(out[:, 0]).reshape(B, -1)
    conf, idx = p.max(-1)
    i, j = idx // w, idx % w
    b = torch.arange(B, device=out.device)
    dx, dy, r = out[b, 1, i, j], out[b, 2, i, j], out[b, 3, i, j]
    return torch.stack([(j + dx) * STRIDE, (i + dy) * STRIDE, r * 10, conf], -1), conf >= threshold
