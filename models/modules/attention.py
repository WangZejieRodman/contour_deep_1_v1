"""
Attention Modules
注意力模块：空间注意力 + 跨层注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """空间注意力：突出重要区域"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] (加权后的特征)
        """
        # 沿通道维度计算统计量
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W], 均值：告诉网络"这个位置普遍重要"
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W], 最大值：告诉网络"这个位置有显著特征"

        # 拼接
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]

        # 卷积生成注意力图
        attention = self.sigmoid(self.conv(concat))  # [B, 1, H, W]  值域: [0, 1]  (sigmoid激活)
        #Conv: 学习"什么样的统计模式对应重要区域", 例如：高均值+高最大值 = 墙壁边缘, 低均值+高最大值 = 孤立噪声
        #Sigmoid: 将输出压缩到[0,1]，作为权重, 1 = 很重要，0 = 不重要

        # 加权
        out = x * attention

        return out


class CrossLayerAttention(nn.Module):
    """跨层注意力：学习8层BEV的权重"""

    def __init__(self, num_layers=8, feature_dim=128):
        """
        Args:
            num_layers: BEV层数 (8层 + 1层VCD = 9, 但只对8层BEV做注意力)
            feature_dim: 特征维度
        """
        super(CrossLayerAttention, self).__init__()

        self.num_layers = num_layers

        # 为每层学习一个权重
        self.layer_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, H, W] -> [B, C, 1, 1]
            nn.Conv2d(feature_dim, num_layers, kernel_size=1),
            nn.Softmax(dim=1)  # 沿层维度归一化
        )

    def forward(self, x, bev_features):
        """
        Args:
            x: [B, C, H, W] 当前特征
            bev_features: List of [B, C, H, W], 长度为8 (每层BEV的特征)
        Returns:
            out: [B, C, H, W] 加权融合后的特征
        """
        # 计算层权重
        weights = self.layer_weights(x)  # [B, 8, 1, 1]

        # === 用学到的权重重新组合单层特征 ==
        # 堆叠BEV特征
        stacked = torch.stack(bev_features, dim=1)  # [B, 8, C, H, W]
        # 加权求和
        weights = weights.unsqueeze(2)  # [B, 8, 1, 1, 1]
        weighted = stacked * weights  # [B, 8, C, H, W]
        out = weighted.sum(dim=1)  # [B, C, H, W]

        return out


# 测试代码
if __name__ == "__main__":
    print("Testing Attention Modules...")

    # 测试空间注意力
    print("\n=== SpatialAttention ===")
    spatial_attn = SpatialAttention()
    x = torch.randn(2, 128, 200, 200)
    out = spatial_attn(x)

    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape
    print("✓ SpatialAttention test passed!")

    # 测试跨层注意力
    print("\n=== CrossLayerAttention ===")
    cross_attn = CrossLayerAttention(num_layers=8, feature_dim=128)
    x = torch.randn(2, 128, 200, 200)
    bev_features = [torch.randn(2, 128, 200, 200) for _ in range(8)]

    out = cross_attn(x, bev_features)

    print(f"Input: {x.shape}")
    print(f"BEV features: {len(bev_features)} × {bev_features[0].shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape
    print("✓ CrossLayerAttention test passed!")

    print("\n✓ All attention tests passed!")