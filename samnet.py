import torch
import torch.nn as nn
import torch.nn.functional as F
from imageenc import imageenc
from efficient_kan import KAN
from collections import OrderedDict
from sam2.build_sam import build_sam2
import os
from sam2.utils.transforms import SAM2Transforms
class KANAdapter(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.kan = KAN([
            in_features,
            int(in_features),
            in_features,
        ])
    
    def forward(self, t):
        return self.kan(t)    
        
class AdapterBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.features=block.attn.qkv.in_features
        self.kan_adapter = KANAdapter(self.features)
        self.conv = nn.Conv2d(
            in_channels=int(self.features/2),
            out_channels=self.features,
            kernel_size=3,
            stride=2,
            padding=1
        ).cuda()

    def forward(self, rgb_feature, t):
        if t.shape[3] != self.features:
            t = t.permute(0, 3, 1, 2)
            t = self.conv(t)
            t = t.permute(0, 2, 3, 1)
        t_out = self.kan_adapter(t+rgb_feature)
        fused_feature = rgb_feature + t_out
        return self.block(fused_feature), t_out
  
class ABlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, rgb_feature, t):
        return self.block(rgb_feature), t

class SAMNET(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super().__init__()
        sam_cfg = "//home/magus/lxy-workspace/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_path = "//home/magus/lxy-workspace/sam2/sam2.1_hiera_large.pt"
        sam = build_sam2(sam_cfg, sam_path)
        self.image_encoder = sam.image_encoder
        # self.no_mask_embed=nn.Embedding(1, sam.image_encoder.neck.d_model)
        self.mask_decoder = sam.sam_mask_decoder
        self.prompt_encoder = sam.sam_prompt_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        self.trunk = self.image_encoder.trunk
        self.patch_embed = self.trunk.patch_embed
        self.neck = self.image_encoder.neck
        self.pos_embed_window=self.trunk.pos_embed_window
        self.pos_embed=self.trunk.pos_embed
        stages =  (2, 6, 36, 4)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.blocks = nn.ModuleList([
            AdapterBlock(block) if (i-1) in self.stage_ends or i == 0 else ABlock(block)
            for i, block in enumerate(self.trunk.blocks)
        ])
        # self.blocks = nn.ModuleList([
        #     ABlock(block) for i, block in enumerate(self.trunk.blocks)
        # ])
        self.linear = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.use_high_res_features_in_sam=True
        self.num_feature_levels = 3 if self.use_high_res_features_in_sam else 1
        self._bb_feat_sizes = [
            (128, 128),
            (64, 64),
            (32, 32),
        ]
        self.return_logits=True
        self._transforms = SAM2Transforms(
            resolution=512,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.hidden_dim = self.image_encoder.neck.d_model
        self.image_embedding_size = 32
        self.no_mask_embed = nn.Embedding(1, self.hidden_dim )
        if(checkpoint_path):
            self.load_pretrained(checkpoint_path)
    
    def _get_pos_embed(self, hw) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward_image(self, img_batch: torch.Tensor, t_batch: torch.Tensor):
        rgb_feature = self.patch_embed(img_batch)
        t_feature = self.patch_embed(t_batch)
        rgb_feature = rgb_feature + self._get_pos_embed(rgb_feature.shape[1:3])
        outputs = []
        for i, block in enumerate(self.blocks):
            #print(rgb_feature.shape,t_feature.shape,i)
            rgb_feature,t_feature = block(rgb_feature, t_feature)
            if i in self.stage_ends:
                feats = rgb_feature.permute(0, 3, 1, 2)
                outputs.append(feats)
        features, pos = self.neck(outputs)
        features, pos = features[: -1], pos[: -1]
        src = features[-1]
        backbone_out = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        # for i in backbone_out["backbone_fpn"]:
        #     print(i.shape)
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out
    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes
    def forward(self, vis_image, inf_image):
        batch_size = vis_image.shape[0]
        backbone_out = self.forward_image(vis_image, inf_image)
        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        feats = [
            feat.permute(1, 2, 0).reshape(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        num_images = len(self._features["image_embed"])
        sparse_embeddings = torch.empty((1, 0, self.hidden_dim), device="cuda")
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            1, -1, self.image_embedding_size, self.image_embedding_size
        )
        all_masks = []
        # all_ious = []
        # all_low_res_masks = []
        for img_idx in range(num_images):
            high_res_features = [
                feat_level[img_idx].unsqueeze(0)
                for feat_level in self._features["high_res_feats"]
            ]
            low_res_masks, iou_predictions, _, _ = self.mask_decoder(
                image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )
            masks = self._transforms.postprocess_masks(
                low_res_masks, (512,512)
            )
            # low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
            # if not self.return_logits:
            #     masks = masks > 0.0
            # masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            # iou_predictions_np = (
            #     iou_predictions.squeeze(0).float().detach().cpu().numpy()
            # )
            # low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks)
            # all_ious.append(iou_predictions_np)
            # all_low_res_masks.append(low_res_masks_np)
        return torch.cat(all_masks,dim=0)

    def load_pretrained(self, pretrained_path):
        """
        Load pretrained weights into SAMNET model.

        Args:
            pretrained_path (str): Path to the pretrained model weights.
        """
        print(f"Loading pretrained model from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path)
        # If the model was trained using DataParallel, adjust key names
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."):  # Remove 'module.' prefix
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v

        self.load_state_dict(new_state_dict)
        print("Pretrained model loaded successfully.")
