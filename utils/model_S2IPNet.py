import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path

try:
    from .sam2.build_sam import build_sam2
    from .main_encoder import MainEncoder
    from .main_decoder import MainDecoder
except ImportError:
    from sam2.build_sam import build_sam2
    from main_encoder import MainEncoder
    from main_decoder import MainDecoder


class CrossAttentionFusion2D(nn.Module):
    def __init__(
        self, channels: int, heads: int = 4, dim_head: int = 32, sr_ratio: int = 4
    ):
        super().__init__()
        assert channels > 0
        self.channels = channels
        self.heads = heads
        self.dim_head = dim_head
        self.inner = heads * dim_head
        self.sr_ratio = sr_ratio

        self.q_proj = nn.Conv2d(channels, self.inner, kernel_size=1, bias=False)
        self.kv_proj = nn.Conv2d(channels, self.inner * 2, kernel_size=1, bias=False)

        self.q_norm = nn.GroupNorm(
            num_groups=min(32, self.inner), num_channels=self.inner
        )
        self.kv_norm = nn.GroupNorm(
            num_groups=min(32, self.inner * 2), num_channels=self.inner * 2
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.inner, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat_a.shape
        assert feat_b.shape[:2] == (
            b,
            c,
        ), f"feat_b should be (B,C,*,*), got {feat_b.shape}"

        q = self.q_norm(self.q_proj(feat_a))

        if self.sr_ratio > 1:
            kb = F.avg_pool2d(feat_b, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        else:
            kb = feat_b

        kv = self.kv_norm(self.kv_proj(kb))
        k, v = kv.chunk(2, dim=1)

        q = q.view(b, self.heads, self.dim_head, h * w).permute(0, 1, 3, 2).contiguous()

        hs, ws = k.shape[-2:]
        k = (
            k.view(b, self.heads, self.dim_head, hs * ws)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        v = (
            v.view(b, self.heads, self.dim_head, hs * ws)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(self.dim_head)
        attn = F.softmax(attn, dim=-1)

        ctx = torch.matmul(attn, v)
        ctx = ctx.permute(0, 1, 3, 2).contiguous().view(b, self.inner, h, w)

        ctx = self.out_proj(ctx)
        out = feat_a + self.gamma * ctx
        out = self.refine(out)
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(DoubleConv, self).__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, input):
        return self.conv(input)


class S2IPNet(nn.Module):
    def __init__(
        self,
        base_channels=32,
        imgsz: int = 256,
        in_channels: int = 1,
        SAM2_mode: str = "s",
        mode: str = "train",
    ):
        super(S2IPNet, self).__init__()
        self.filters = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]
        sam2_root = Path(__file__).resolve().parent / "sam2"
        if SAM2_mode == "s":
            checkpoint_path = sam2_root / "Checkpoints/sam2.1_hiera_small.pt"
            model_cfg = sam2_root / "sam2_configs/sam2.1_hiera_s.yaml"
        elif SAM2_mode == "b+":
            checkpoint_path = sam2_root / "Checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = sam2_root / "sam2_configs/sam2.1_hiera_b+.yaml"
        elif SAM2_mode == "l":
            checkpoint_path = sam2_root / "Checkpoints/sam2.1_hiera_large.pt"
            model_cfg = sam2_root / "sam2_configs/sam2.1_hiera_l.yaml"
        elif SAM2_mode == "t":
            checkpoint_path = sam2_root / "Checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = sam2_root / "sam2_configs/sam2.1_hiera_t.yaml"
        else:
            raise ValueError(
                f"Invalid SAM2_mode {SAM2_mode}, should be 's', 'b+', 'l', or 't'."
            )

        if mode == "test":
            checkpoint_path = None
        self.sam2encoder = build_sam2(model_cfg, checkpoint_path, imgsz=imgsz)
        for param in self.sam2encoder.parameters():
            param.requires_grad = False

        input_ch = 16
        self.Adapter0 = DoubleConv(
            input_ch, self.filters[0], mid_ch=(input_ch + self.filters[0]) // 2
        )
        self.Adapter1 = DoubleConv(
            input_ch, self.filters[1], mid_ch=(input_ch + self.filters[1]) // 2
        )
        self.Adapter2 = DoubleConv(
            input_ch, self.filters[2], mid_ch=(input_ch + self.filters[2]) // 2
        )
        self.Adapter3 = DoubleConv(
            input_ch, self.filters[3], mid_ch=(input_ch + self.filters[3]) // 2
        )

        self.encoder = MainEncoder(
            in_channels=in_channels, filters=self.filters, act="silu", strict_size=True
        )

        self.Fuse0 = CrossAttentionFusion2D(
            self.filters[0], heads=2, dim_head=16, sr_ratio=8
        )
        self.Fuse1 = CrossAttentionFusion2D(
            self.filters[1], heads=2, dim_head=16, sr_ratio=4
        )
        self.Fuse2 = CrossAttentionFusion2D(
            self.filters[2], heads=2, dim_head=16, sr_ratio=2
        )
        self.Fuse3 = CrossAttentionFusion2D(
            self.filters[3], heads=2, dim_head=16, sr_ratio=1
        )

        self.decoder = MainDecoder(filters=self.filters)

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam2encoder.eval()
        return self

    def forward(self, x):
        x = x if x.dim() != 3 else x.unsqueeze(0)
        if x.shape[1] == 1:
            x_sam = x.repeat(1, 3, 1, 1)
        else:
            x_sam = x
        self.sam2encoder.eval()
        with torch.no_grad():
            SAM_feats = self.sam2encoder(x_sam)
        feat_0 = self.Adapter0(SAM_feats[0])
        feat_1 = self.Adapter1(SAM_feats[1])
        feat_2 = self.Adapter2(SAM_feats[2])
        feat_3 = self.Adapter3(SAM_feats[3])
        out_0, out_1, out_2, out_3 = self.encoder(x)

        feats = [
            self.Fuse0(out_0, feat_0),
            self.Fuse1(out_1, feat_1),
            self.Fuse2(out_2, feat_2),
            self.Fuse3(out_3, feat_3),
        ]

        final_masks = self.decoder(feats[0], feats[1], feats[2], feats[3])

        return final_masks
