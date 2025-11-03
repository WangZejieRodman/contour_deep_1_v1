"""
Loss Functions for Place Recognition
包括：Triplet Loss with Hard Mining, InfoNCE Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss with Hard Mining

    策略：
    - Hard Negative Mining: 选择距离anchor最近的负样本（最难的）
    - Semi-Hard Positive Mining: 选择距离适中的正样本
    """

    def __init__(self, margin=0.5, mining='hard'):
        """
        Args:
            margin: 边界值，正样本距离应比负样本距离小至少margin
            mining: 'hard' (最难负样本), 'semi-hard', 'all' (所有三元组)
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: [B, D] anchor特征
            positive: [B, D] positive特征
            negatives: [B, N, D] 多个negative特征

        Returns:
            loss: 标量
            stats: 统计字典
        """
        batch_size = anchor.size(0)
        num_negatives = negatives.size(1)

        # 计算距离（欧式距离平方）
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)  # [B]

        # 计算所有负样本距离
        anchor_expanded = anchor.unsqueeze(1)  # [B, 1, D]
        neg_dists = torch.sum((anchor_expanded - negatives) ** 2, dim=2)  # [B, N]

        # Hard Mining: 选择最难的负样本（距离最小的）
        if self.mining == 'hard':
            neg_dist, _ = torch.min(neg_dists, dim=1)  # [B]

        # Semi-Hard Mining: 选择满足 pos_dist < neg_dist < pos_dist + margin 的负样本
        elif self.mining == 'semi-hard':
            # 找到所有semi-hard负样本
            mask = (neg_dists > pos_dist.unsqueeze(1)) & \
                   (neg_dists < pos_dist.unsqueeze(1) + self.margin)

            # 如果存在semi-hard负样本，选择其中最近的；否则退化到hard mining
            if mask.any():
                neg_dists_masked = neg_dists.clone()
                neg_dists_masked[~mask] = float('inf')
                neg_dist, _ = torch.min(neg_dists_masked, dim=1)
            else:
                neg_dist, _ = torch.min(neg_dists, dim=1)

        # All: 对所有负样本计算损失
        else:
            neg_dist = neg_dists  # [B, N]
            pos_dist = pos_dist.unsqueeze(1).expand(-1, num_negatives)  # [B, N]

        # 计算Triplet Loss
        if self.mining in ['hard', 'semi-hard']:
            loss = F.relu(pos_dist - neg_dist + self.margin)  # [B]
            loss = loss.mean()
        else:
            loss = F.relu(pos_dist - neg_dist + self.margin)  # [B, N]
            loss = loss.mean()

        # 统计信息
        with torch.no_grad():
            # 计算有效三元组比例（违反margin的三元组）
            if self.mining in ['hard', 'semi-hard']:
                active_triplets = (pos_dist > neg_dist - self.margin).float().mean()
            else:
                # mining='all'时，pos_dist需要扩展维度
                active_triplets = (pos_dist.unsqueeze(1) > neg_dist - self.margin).float().mean()

            stats = {
                'loss': loss.item(),
                'pos_dist_mean': pos_dist.mean().item() if pos_dist.dim() == 1 else pos_dist.mean().item(),
                'neg_dist_mean': neg_dist.mean().item(),
                'active_triplets': active_triplets.item(),
                'margin': self.margin
            }

        return loss, stats


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (对比学习)

    将问题转化为多分类：给定anchor，从1个正样本+N个负样本中识别出正样本
    """

    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: 温度参数，控制分布的平滑度
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: [B, D]
            positive: [B, D]
            negatives: [B, N, D]

        Returns:
            loss: 标量
            stats: 统计字典
        """
        batch_size = anchor.size(0)
        num_negatives = negatives.size(1)

        # 归一化特征（InfoNCE通常用余弦相似度）
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)

        # 计算相似度
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # [B]

        anchor_expanded = anchor.unsqueeze(1)  # [B, 1, D]
        neg_sim = torch.sum(anchor_expanded * negatives, dim=2) / self.temperature  # [B, N]

        # 构建logits: [正样本相似度, 负样本1, 负样本2, ...]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]

        # 标签：正样本在第0位
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)

        # 统计信息
        with torch.no_grad():
            # 预测准确率
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == labels).float().mean()

            stats = {
                'loss': loss.item(),
                'accuracy': accuracy.item(),
                'pos_sim_mean': pos_sim.mean().item(),
                'neg_sim_mean': neg_sim.mean().item(),
                'temperature': self.temperature
            }

        return loss, stats


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Testing Loss Functions...")

    # 模拟数据
    batch_size = 4
    feature_dim = 128
    num_negatives = 10

    anchor = torch.randn(batch_size, feature_dim)
    positive = anchor + 0.1 * torch.randn(batch_size, feature_dim)  # 相似
    negatives = torch.randn(batch_size, num_negatives, feature_dim)  # 不相似

    # 测试Triplet Loss
    print("\n=== Triplet Loss ===")
    triplet_loss = TripletLoss(margin=0.5, mining='hard')
    loss, stats = triplet_loss(anchor, positive, negatives)

    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")

    # 测试InfoNCE Loss
    print("\n=== InfoNCE Loss ===")
    infonce_loss = InfoNCELoss(temperature=0.07)
    loss, stats = infonce_loss(anchor, positive, negatives)

    print(f"Loss: {loss.item():.4f}")
    print(f"Stats: {stats}")

    # 梯度测试
    print("\n=== 梯度测试 ===")
    anchor.requires_grad = True
    loss, _ = triplet_loss(anchor, positive, negatives)
    loss.backward()
    print(f"Anchor gradient norm: {anchor.grad.norm().item():.6f}")

    print("\n✓ Loss Functions测试完成!")