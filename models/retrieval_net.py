"""
RetrievalNet: 方向1检索特征提取网络 (集成STN2d版本)
输入: [B, 9, 200, 200] (8层BEV + 1层VCD)
输出: [B, 128] 全局特征向量

架构改进：
1. 增加pre_conv提取基础特征 (9→32)
2. 增加STN2d学习旋转对齐
3. 后续网络输入改为32通道
"""

import torch
import torch.nn as nn
from models.modules import MultiScaleConv, SpatialAttention, STN2d


class ResBlock(nn.Module):
    """残差块，用于下采样"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class RetrievalNet(nn.Module):
    """
    完整版RetrievalNet（集成STN2d）

    流程：
    Input [B, 9, 200, 200]
      ↓
    → pre_conv (9→32) → 提取基础特征
      ↓
    → STN2d → 学习旋转对齐
      ↓
    → MultiScaleConv (32→128)
      ↓
    → SpatialAttention
      ↓
    → ResBlocks下采样
      ↓
    → GlobalPooling (GAP + GMP)
      ↓
    → FC
      ↓
    → [B, 128] L2归一化特征
    """

    def __init__(self, output_dim=128, use_stn=True):
        """
        Args:
            output_dim: 输出特征维度
            use_stn: 是否使用STN2d（方便对比实验）
        """
        super(RetrievalNet, self).__init__()

        self.use_stn = use_stn

        # === 阶段0：轻量级特征提取 ===
        self.pre_conv = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # === 新增：STN2d模块 ===
        if self.use_stn:
            self.stn = STN2d(in_channels=32)

        # === 阶段1：多尺度卷积（输入改为32通道）===
        self.multiscale_conv = MultiScaleConv(in_channels=32, out_channels=128)

        # === 阶段2：空间注意力 ===
        self.spatial_attention = SpatialAttention()

        # === 阶段3：残差下采样 ===
        self.res_block1 = ResBlock(128, 128, stride=2)  # 200→100
        self.res_block2 = ResBlock(128, 128, stride=2)  # 100→50
        self.res_block3 = ResBlock(128, 128, stride=2)  # 50→25

        # === 阶段4：全局池化 ===
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # === 阶段5：全连接 ===
        # GAP(128) + GMP(128) = 256维输入
        self.fc = nn.Sequential(
            nn.Linear(256, 256),  # 修改：32→256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 9, 200, 200]
               - 前8个通道：8层BEV
               - 第9个通道：VCD

        Returns:
            features: [B, 128]
        """
        batch_size = x.size(0)

        # === 阶段0：轻量级特征提取 ===
        x = self.pre_conv(x)  # [B, 32, 200, 200]

        # === 新增：STN2d对齐 ===
        if self.use_stn:
            x = self.stn(x)  # [B, 32, 200, 200]（已旋转对齐）

        # === 阶段1：多尺度卷积 ===
        x = self.multiscale_conv(x)  # [B, 128, 200, 200]

        # === 阶段2：空间注意力 ===
        x = self.spatial_attention(x)  # [B, 128, 200, 200]

        # === 阶段3：残差下采样 ===
        x = self.res_block1(x)  # [B, 128, 100, 100]
        x = self.res_block2(x)  # [B, 128, 50, 50]
        x = self.res_block3(x)  # [B, 128, 25, 25]

        # === 阶段4：全局池化 ===
        gap_features = self.gap(x)  # [B, 128, 1, 1]
        gmp_features = self.gmp(x)  # [B, 128, 1, 1]

        gap_features = gap_features.view(batch_size, -1)  # [B, 128]
        gmp_features = gmp_features.view(batch_size, -1)  # [B, 128]

        global_features = torch.cat([gap_features, gmp_features], dim=1)  # [B, 256]

        # === 阶段5：全连接 ===
        output = self.fc(global_features)  # [B, 128]

        # L2归一化
        output = torch.nn.functional.normalize(output, p=2, dim=1)

        return output

    def get_embedding_dim(self):
        return 128

    def get_stn_angle(self, x):
        """
        获取STN学到的旋转角度（用于调试）

        Args:
            x: [B, 9, 200, 200]
        Returns:
            angles: [B, 1] 旋转角度（度）
        """
        if not self.use_stn:
            return None

        with torch.no_grad():
            x = self.pre_conv(x)
            angles = self.stn.get_rotation_angle(x)
        return angles


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("测试RetrievalNet (集成STN2d版本)...")

    # 测试：有STN vs 无STN
    for use_stn in [False, True]:
        print(f"\n{'='*60}")
        print(f"测试模式: {'使用STN' if use_stn else '不使用STN'}")
        print(f"{'='*60}")

        # 创建网络
        model = RetrievalNet(output_dim=128, use_stn=use_stn)

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n网络参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  参数量: {total_params / 1e6:.2f}M")
        print(f"  目标: <10M ({'✓' if total_params < 10e6 else '✗'})")

        # 测试前向传播
        print("\n测试前向传播:")
        batch_size = 4
        x = torch.randn(batch_size, 9, 200, 200)

        print(f"  输入: {x.shape}")

        with torch.no_grad():
            output = model(x)

        print(f"  输出: {output.shape}")
        print(f"  预期: [{batch_size}, 128]")

        assert output.shape == (batch_size, 128), "输出维度错误!"

        # 验证L2归一化
        norms = torch.norm(output, p=2, dim=1)
        print(f"\n  特征向量L2范数均值: {norms.mean().item():.6f} (应接近1.0)")

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2归一化失败!"

        # 如果使用STN，测试获取旋转角度
        if use_stn:
            angles = model.get_stn_angle(x)
            print(f"  STN学到的旋转角度: {angles.squeeze().tolist()} 度")

    # 测试梯度
    print("\n" + "="*60)
    print("测试梯度反向传播:")
    print("="*60)

    model = RetrievalNet(output_dim=128, use_stn=True)
    x = torch.randn(batch_size, 9, 200, 200)
    x.requires_grad = True

    output = model(x)
    loss = output.sum()
    loss.backward()

    print(f"  梯度检查: {'✓ 通过' if x.grad is not None else '✗ 失败'}")

    # 推理速度测试
    print("\n" + "="*60)
    print("推理速度测试:")
    print("="*60)

    model.eval()
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    model = model.to(device)
    x_test = torch.randn(batch_size, 9, 200, 200).to(device)

    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(x_test)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 计时
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            _ = model(x_test)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        end = time.time()

    avg_time = (end - start) / num_iterations * 1000
    fps = 1000 / avg_time * batch_size

    print(f"  平均推理时间: {avg_time:.2f}ms")
    print(f"  吞吐量: {fps:.1f} samples/sec")
    print(f"  目标: >30 FPS ({'✓ 达标' if fps > 30 else '✗ 未达标'})")

    print("\n✓ RetrievalNet (集成STN2d) 所有测试通过!")
    print("\n架构说明:")
    print("  - 新增pre_conv (9→32通道)")
    print("  - 新增STN2d学习旋转对齐")
    print("  - 可通过use_stn参数控制是否使用STN")