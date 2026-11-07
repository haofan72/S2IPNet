import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask


class MainDecoder(nn.Module):
    def __init__(self, filters=None):
        super(MainDecoder, self).__init__()
        self.sam_p4 = SupervisedAttentionModule(filters[3])
        self.sam_p3 = SupervisedAttentionModule(filters[2])
        self.sam_p2 = SupervisedAttentionModule(filters[1])
        self.sam_p1 = SupervisedAttentionModule(filters[0])

        self.conv_p3 = nn.Sequential(
            nn.Conv2d(
                filters[3] + filters[2], filters[2], kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(
                filters[2] + filters[1], filters[1], kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
        )
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(
                filters[1] + filters[0], filters[0], kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(sum(filters), sum(filters) // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(sum(filters) // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(sum(filters) // 4, 1, kernel_size=1),
        )

    def forward(self, d1, d2, d3, d4):
        p4, mask_p4 = self.sam_p4(d4)

        conv_p3_input = torch.cat(
            (d3, F.interpolate(p4, scale_factor=(2, 2), mode="bilinear")), dim=1
        )
        p3 = self.conv_p3(conv_p3_input)
        p3, mask_p3 = self.sam_p3(p3)

        conv_p2_input = torch.cat(
            (d2, F.interpolate(p3, scale_factor=(2, 2), mode="bilinear")), dim=1
        )
        p2 = self.conv_p2(conv_p2_input)
        p2, mask_p2 = self.sam_p2(p2)

        conv_p1_input = torch.cat(
            (d1, F.interpolate(p2, scale_factor=(2, 2), mode="bilinear")), dim=1
        )
        p1 = self.conv_p1(conv_p1_input)
        p1, mask_p1 = self.sam_p1(p1)

        p2_up = F.interpolate(p2, scale_factor=(2, 2), mode="bilinear")
        p3_up = F.interpolate(p3, scale_factor=(4, 4), mode="bilinear")
        p4_up = F.interpolate(p4, scale_factor=(8, 8), mode="bilinear")

        mask_p2_up = F.interpolate(mask_p2, scale_factor=(2, 2), mode="bilinear")
        mask_p3_up = F.interpolate(mask_p3, scale_factor=(4, 4), mode="bilinear")
        mask_p4_up = F.interpolate(mask_p4, scale_factor=(8, 8), mode="bilinear")
        fused_mask = self.fuse_conv(torch.cat([p1, p2_up, p3_up, p4_up], dim=1))

        return [fused_mask, mask_p1, mask_p2_up, mask_p3_up, mask_p4_up]
