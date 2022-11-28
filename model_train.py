import torch
from torch import nn
from swin_transformer_v2 import SwinTransformerBlock
from swin_transformer_v2 import PatchEmbed
from swin_transformer_v2 import PatchMerging

class RES_block(nn.Module):
    def __init__(self):
        super(RES_block, self).__init__()
        self.layer1 = nn.Conv2d()
        self.layer2 = nn.Conv2d()

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x + x2


def MRU_block(x, y, sample_martix):
    B, C, H, W = x.shape
    temp_x = torch.zeros(x.shape)
    for batch in range(B):
        for channel in range(C):
            temp_x[batch][channel][:][:] = x * sample_martix


class layer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):
        super(layer, self).__init__()
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SWNet(nn.Module):
    """
    Parameters:
        img_size (int | tuple(int)): Input image size. Default 224


    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super(SWNet, self).__init__()
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        patches_resolution = self.patch_embed.patches_resolution


        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            temp_layer = layer(dim=int(embed_dim * 2 ** i_layer),
                          input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          num_heads=num_heads[i_layer],
                          window_size=window_size,
                          mlp_ratio=4.,
                          qkv_bias=True,
                          drop=0.,
                          attn_drop=0.,
                          drop_path=0.,
                          norm_layer=nn.LayerNorm,
                          downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                          use_checkpoint=False,
                          pretrained_window_size=0)
            self.layers.append(temp_layer)



    def forward(self, x):
        PatchEmbed_data = self.patch_embed(x)
        for layer in self.layers:
            PatchEmbed_data = layer(PatchEmbed_data)
        return PatchEmbed_data



