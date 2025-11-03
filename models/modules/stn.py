"""
STN2d: 2D空间变换网络
用于学习BEV图像的旋转对齐，实现旋转不变性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationNetwork(nn.Module):
    """
    定位网络：学习旋转角度θ

    输入: [B, C, H, W] 特征图
    输出: [B, 1] 旋转角度（弧度）
    """

    def __init__(self, in_channels=32):
        super(LocalizationNetwork, self).__init__()

        # 全局特征提取
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        # 全局池化
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 预测旋转角度
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        # 初始化：让STN初始状态不做旋转
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.zero_()

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            theta: [B, 1] 旋转角度（弧度）
        """
        batch_size = x.size(0)

        # 提取全局特征
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # [B, 128, 1, 1]
        x = x.view(batch_size, -1)  # [B, 128]

        # 预测角度
        x = F.relu(self.fc1(x))
        theta = self.fc2(x)  # [B, 1]

        return theta


class STN2d(nn.Module):
    """
    2D空间变换网络

    功能：学习旋转角度θ，对输入特征图进行旋转对齐
    原理：模仿PointNetVlad的STN3d，但简化为2D旋转
    """

    def __init__(self, in_channels=32):
        """
        Args:
            in_channels: 输入特征图通道数
        """
        super(STN2d, self).__init__()

        # 定位网络：学习旋转角度
        self.localization = LocalizationNetwork(in_channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征图
        Returns:
            x_transformed: [B, C, H, W] 旋转对齐后的特征图
        """
        batch_size = x.size(0)

        # 1. 学习旋转角度
        theta = self.localization(x)  # [B, 1]

        # 2. 构造2×3旋转矩阵
        affine_matrix = self._build_rotation_matrix(theta)  # [B, 2, 3]

        # 3. 生成采样网格
        grid = F.affine_grid(affine_matrix, x.size(), align_corners=False)

        # 4. 双线性插值变换
        x_transformed = F.grid_sample(x, grid, align_corners=False)

        return x_transformed

    def _build_rotation_matrix(self, theta):
        """
        构造2D旋转的2×3仿射矩阵

        矩阵形式：
        [[cos(θ)  -sin(θ)  0]
         [sin(θ)   cos(θ)  0]]

        Args:
            theta: [B, 1] 旋转角度（弧度）
        Returns:
            affine_matrix: [B, 2, 3] 仿射变换矩阵
        """
        cos_theta = torch.cos(theta)  # [B, 1]
        sin_theta = torch.sin(theta)  # [B, 1]
        zeros = torch.zeros_like(theta)  # [B, 1]

        # 第一行：[cos -sin 0]
        row1 = torch.cat([cos_theta, -sin_theta, zeros], dim=1)  # [B, 3]

        # 第二行：[sin cos 0]
        row2 = torch.cat([sin_theta, cos_theta, zeros], dim=1)  # [B, 3]

        # 堆叠成2×3矩阵
        affine_matrix = torch.stack([row1, row2], dim=1)  # [B, 2, 3]

        return affine_matrix

    def get_rotation_angle(self, x):
        """
        辅助函数：获取学到的旋转角度（用于调试）

        Args:
            x: [B, C, H, W]
        Returns:
            theta_deg: [B, 1] 旋转角度（度）
        """
        with torch.no_grad():
            theta = self.localization(x)
            theta_deg = torch.rad2deg(theta)
        return theta_deg


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("测试STN2d模块...")

    # 创建模块
    stn = STN2d(in_channels=32)

    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 32, 200, 200)

    print(f"\n输入shape: {x.shape}")

    # 前向传播
    x_aligned = stn(x)

    print(f"输出shape: {x_aligned.shape}")
    print(f"预期shape: {x.shape}")

    assert x_aligned.shape == x.shape, "Shape不匹配!"

    # 获取学到的角度
    angles = stn.get_rotation_angle(x)
    print(f"\n学到的旋转角度: {angles.squeeze().tolist()} 度")

    # 参数量统计
    total_params = sum(p.numel() for p in stn.parameters())
    print(f"\nSTN2d参数量: {total_params:,} ({total_params/1e3:.1f}K)")

    # 梯度测试
    x.requires_grad = True
    output = stn(x)
    loss = output.sum()
    loss.backward()

    print(f"\n梯度检查: {'✓ 通过' if x.grad is not None else '✗ 失败'}")

    print("\n✓ STN2d测试完成!")