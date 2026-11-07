# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from .sam.prompt_encoder import PromptEncoder
from .sam.mask_decoder import MaskDecoder
from .sam.transformer import TwoWayTransformer
from .sam2_utils import MLP

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        imgsz
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.imgsz = imgsz

    def forward(self, img_batch: torch.Tensor):
        _bb_feat_sizes = [
            (self.imgsz, self.imgsz),
            (self.imgsz // 2, self.imgsz // 2),
            (self.imgsz // 4, self.imgsz // 4),
            (self.imgsz // 8, self.imgsz // 8),
        ]
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"

        backbone_out = self.image_encoder(img_batch)
        _, vision_feats, _, _ = _prepare_backbone_features(backbone_out)
        feats = [
            feat.permute(1, 2, 0).reshape(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], _bb_feat_sizes[::-1])
        ][::-1]

        return feats


def _prepare_backbone_features(backbone_out, num_feature_levels: int = 4):
    """
    Processes backbone output to extract feature maps, positional embeddings, and their sizes for a specified number of feature levels.
    Args:
        backbone_out (Union[torch.Tensor, dict]): Output from the backbone network. If a dict, must contain "backbone_fpn" and "vision_pos_enc" keys.
        num_feature_levels (int, optional): Number of feature levels to extract. Defaults to 4.
    Returns:
        Tuple[
            dict,
            List[torch.Tensor],
            List[torch.Tensor],
            List[Tuple[int, int]]
        ]: 
            - Updated backbone output (copied if dict or cloned if tensor).
            - List of feature maps, each flattened and permuted to shape (HW, N, C).
            - List of positional embeddings, each flattened and permuted to shape (HW, N, C).
            - List of feature map sizes as (height, width) tuples.
    """
    if isinstance(backbone_out, torch.Tensor):
        backbone_out = backbone_out.clone()
    elif isinstance(backbone_out, dict):
        backbone_out = backbone_out.copy()

    assert len(backbone_out["backbone_fpn"]) == len(
        backbone_out["vision_pos_enc"])
    assert len(backbone_out["backbone_fpn"]) >= num_feature_levels

    feature_maps = backbone_out["backbone_fpn"][-num_feature_levels:]
    vision_pos_embeds = backbone_out["vision_pos_enc"][-num_feature_levels:]

    feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
    # flatten NxCxHxW to HWxNxC
    vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
    vision_pos_embeds = [x.flatten(2).permute(2, 0, 1)
                         for x in vision_pos_embeds]

    return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
