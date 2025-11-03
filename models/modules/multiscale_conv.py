"""
Multi-Scale Convolution Module
多尺度卷积模块：捕捉不同感受野的特征
"""

import torch
import torch.nn as nn


class MultiScaleConv(nn.Module):
    """多尺度卷积模块"""

    def __init__(self, in_channels=9, out_channels=128):
        """
        Args:
            in_channels: 输入通道数 (9: 8层BEV + 1层VCD)
            out_channels: 输出通道数
        """
        super(MultiScaleConv, self).__init__()

        # 每个分支输出64通道
        branch_channels = 64

        # 分支1: 3×3卷积 (局部特征)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 分支2: 7×7卷积 (中等尺度特征)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 分支3: 15×15卷积 (全局特征)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        # 融合层: 1×1卷积 (3×64 -> 128)
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 9, H, W]
        Returns:
            out: [B, 128, H, W]
        """
        # 三个分支并行
        b1 = self.branch1(x)  # [B, 64, H, W]
        b2 = self.branch2(x)  # [B, 64, H, W]
        b3 = self.branch3(x)  # [B, 64, H, W]

        # 拼接
        concat = torch.cat([b1, b2, b3], dim=1)  # [B, 192, H, W]

        # 融合
        out = self.fusion(concat)  # [B, 128, H, W]

        return out


# 测试代码
if __name__ == "__main__":
    print("Testing MultiScaleConv...")

    # 创建模块
    model = MultiScaleConv(in_channels=9, out_channels=128)

    # 测试输入
    x = torch.randn(2, 9, 200, 200)  # [B, C, H, W]

    # 前向传播
    out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected: [2, 128, 200, 200]")

    assert out.shape == (2, 128, 200, 200), "Shape mismatch!"

    # 参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    print("✓ MultiScaleConv test passed!")