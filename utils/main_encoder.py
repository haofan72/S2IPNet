import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class SPDConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c1 = c1 * 4
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        x = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        x = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            1,
        )
        return self.act(self.conv(x))


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act="relu"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act="relu"):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, k=3, s=1, p=1, act=act)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + identity)


class MainEncoder(nn.Module):
    def __init__(
        self, in_channels=3, filters=(32, 64, 128, 256), act="relu", strict_size=True
    ):
        super().__init__()
        assert len(filters) == 4
        self.strict_size = strict_size

        c0, c1, c2, c3 = filters

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c0, k=3, s=1, p=1, act=act),
            ConvBNAct(c0, c0, k=3, s=1, p=1, act=act),
        )

        self.down1 = SPDConv(c0, c1, k=3, s=1, act=True)
        self.down2 = SPDConv(c1, c2, k=3, s=1, act=True)
        self.down3 = SPDConv(c2, c3, k=3, s=1, act=True)

        self.stage1 = ResidualBlock(c1, c1, act=act)
        self.stage2 = ResidualBlock(c2, c2, act=act)
        self.stage3 = ResidualBlock(c3, c3, act=act)

    def _pad_to_divisible_by_8(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def forward(self, x):
        if self.strict_size:
            h, w = x.shape[-2:]
            assert (h % 8 == 0) and (
                w % 8 == 0
            ), f"Input H,W must be divisible by 8, got {h}x{w}"
        else:
            x = self._pad_to_divisible_by_8(x)

        out_0 = self.stem(x)
        out_1 = self.stage1(self.down1(out_0))
        out_2 = self.stage2(self.down2(out_1))
        out_3 = self.stage3(self.down3(out_2))
        return out_0, out_1, out_2, out_3


