import torch.nn as nn
from efficient_kan import KAN
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
class MLPAdapter(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1_rgb = nn.Linear(int(in_features), int(in_features*2))
        self.fc1_t = nn.Linear(int(in_features), int(in_features*2))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(in_features*4), int(in_features))
        self.drop = nn.Dropout(0.1)
        self.conv = nn.Conv2d(
            in_channels=int(in_features/2),
            out_channels=in_features,
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, rgb_feature, t_feature):
        if t_feature.shape != rgb_feature.shape:
            t_feature = t_feature.permute(0, 3, 1, 2) 
            t_feature=self.conv(t_feature)
            t_feature = t_feature.permute(0, 2, 3, 1)
        rgb_out = self.act(self.fc1_rgb(rgb_feature)) 
        t_out = self.act(self.fc1_t(t_feature)) 
        rgb_out=self.drop(rgb_out)
        t_out=self.drop(t_out)
        fused = torch.cat([rgb_out, t_out], dim=-1)
        output = self.fc2(fused) 
        output=self.drop(output)
        return output
    
class AdapterBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.features=block.attn.qkv.in_features
        #print(self.features)
        self.mlp_adapter = MLPAdapter(self.features)

    def forward(self, rgb_feature, t):
        t_out = self.mlp_adapter(rgb_feature, t)
        fused_feature = rgb_feature + t_out
        return self.block(fused_feature), t_out
    
class ABlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, rgb_feature, t):
        return self.block(rgb_feature), t

class imageenc(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()    
        self.encoder = model.image_encoder.trunk
        for param in self.encoder.parameters():
            param.requires_grad = False
        stages =  (2, 6, 36, 4)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        # self.blocks = nn.ModuleList([
        #     AdapterBlock(block) if (i-1) in self.stage_ends else ABlock(block)
        #     for i, block in enumerate(self.encoder.blocks)
        # ])
        self.blocks = nn.ModuleList([
            AdapterBlock(block) for block in self.encoder.blocks
        ])
        self.patch_embed = self.encoder.patch_embed


    def forward(self, rgb,t):
        rgb_feature = self.patch_embed(rgb)
        t_feature = self.patch_embed(t)
        outputs = []
        for i, block in enumerate(self.blocks):
            #print(rgb_feature.shape,t_feature.shape,i)
            rgb_feature,t_feature = block(rgb_feature, t_feature)
            if i in self.stage_ends:
                feats = rgb_feature.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs
