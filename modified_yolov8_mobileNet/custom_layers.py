# custom_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    import timm
except Exception:
    timm = None

# -----------------------------
# Basic DW/Pointwise + Lite Blocks
# -----------------------------
class ConvDW(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, k // 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class SPPFLite(nn.Module):
    """ Non-lazy SPPF-lite: requires c1 in-channels explicitly. """
    def __init__(self, c1: int, c2: int = 128, k: int = 5):
        super().__init__()
        c_ = max(8, c1 // 2)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.cv2 = nn.Conv2d(c_ * 4, c2, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class C2fDW(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        c_ = max(8, c2 // 2)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, 0)
        self.m = nn.ModuleList(ConvDW(c_, c_, 3, 1) for _ in range(n))
        self.cv2 = nn.Conv2d(c_ * (n + 1), c2, 1, 1, 0)

    def forward(self, x):
        y = [self.cv1(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class FastSigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)

# -----------------------------
# BiFPN-Lite (no lazy; needs explicit in-ch list)
# -----------------------------
class BiFPNLite(nn.Module):
    """
    Expects [C2,C3,C4,C5] with known channels list `ch=[c2,c3,c4,c5]`.
    Returns [P2,P3,P4,P5] with `out_ch` channels each.
    """
    def __init__(self, ch: List[int], out_ch: int = 128):
        super().__init__()
        assert timm is not None or isinstance(ch, list)
        assert len(ch) == 4, "BiFPNLite: need C2..C5 channels list"
        self.out_ch = out_ch
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1, 1, 0) for c in ch])

        # positive init for stability
        self.wt_up = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(3)])
        self.wt_dn = nn.ParameterList([nn.Parameter(torch.ones(3)) for _ in range(3)])
        self.act = FastSigmoid()
        self.smooth = nn.ModuleList([C2fDW(out_ch, out_ch, n=1) for _ in range(6)])

    @torch.jit.ignore
    def _norm(self, w):
        s = self.act(w).clamp_min(1e-4)
        return s / (s.sum() + 1e-6)

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 4, "BiFPNLite expects [C2..C5]"
        c2, c3, c4, c5 = x
        p2, p3, p4, p5 = [lat(f) for lat, f in zip(self.lateral, [c2, c3, c4, c5])]

        # top-down
        w43 = self._norm(self.wt_up[0])
        p4 = self.smooth[0](w43[0] * p4 + w43[1] * F.interpolate(p5, scale_factor=2, mode='nearest'))
        w32 = self._norm(self.wt_up[1])
        p3 = self.smooth[1](w32[0] * p3 + w32[1] * F.interpolate(p4, scale_factor=2, mode='nearest'))
        w21 = self._norm(self.wt_up[2])
        p2 = self.smooth[2](w21[0] * p2 + w21[1] * F.interpolate(p3, scale_factor=2, mode='nearest'))

        # bottom-up
        w23 = self._norm(self.wt_dn[0])
        p3 = self.smooth[3](w23[0] * p3 + w23[1] * F.max_pool2d(p2, 2) + w23[2] * p3)
        w34 = self._norm(self.wt_dn[1])
        p4 = self.smooth[4](w34[0] * p4 + w34[1] * F.max_pool2d(p3, 2) + w34[2] * p4)
        w45 = self._norm(self.wt_dn[2])
        p5 = self.smooth[5](w45[0] * p5 + w45[1] * F.max_pool2d(p4, 2) + w45[2] * p5)

        return [p2, p3, p4, p5]

# -----------------------------
# TIMM backbone wrapper → [C2,C3,C4,C5]
# -----------------------------
class TimmBackboneC2C5(nn.Module):
    """Wrap a timm backbone exposing features_only; returns [C2,C3,C4,C5]"""
    def __init__(self, name: str = "mobilenetv3_large_100", pretrained: bool = True):
        super().__init__()
        assert timm is not None, "Please install timm"
        self.m = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        self.out_channels = self.m.feature_info.channels()

    def forward(self, x):
        feats = self.m(x)
        return [feats[0], feats[1], feats[2], feats[3]]

class TimmGhostBackboneC2C5(TimmBackboneC2C5):
    def __init__(self, name: str = "ghostnetv2_120", pretrained: bool = True):
        super().__init__(name=name, pretrained=pretrained)

# -----------------------------
# Split helpers
# -----------------------------
class Index0(nn.Module):
    def forward(self, x): return x[0]
class Index1(nn.Module):
    def forward(self, x): return x[1]
class Index2(nn.Module):
    def forward(self, x): return x[2]
class Index3(nn.Module):
    def forward(self, x): return x[3]
