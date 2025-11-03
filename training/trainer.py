"""
Base Trainer Class
通用训练器框架
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np
import gc  # 添加


def clear_cuda_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class BaseTrainer:
    """通用训练器基类"""

    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 optimizer,
                 scheduler,
                 criterion,
                 device: str = 'cuda',
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints',
                 experiment_name: str = 'experiment'):
        """
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            criterion: 损失函数
            device: 设备
            log_dir: 日志目录
            checkpoint_dir: checkpoint目录
            experiment_name: 实验名称
        """
        # 1. 将模型移到GPU
        self.model = model.to(device)

        # 2. 保存所有组件
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        # 3. 创建日志和checkpoint目录
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 4. 初始化TensorBoard
        self.writer = SummaryWriter(self.log_dir)

        # 5. 初始化训练状态
        self.current_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0

        print(f"Trainer initialized:")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Device: {device}")

        # 6. 清理CUDA缓存
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory: {total_memory:.2f} GB")
            clear_cuda_cache()  # 初始清理

    def train_epoch(self) -> Dict:
        """训练一个epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_stats = {}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            anchor = batch['anchor'].to(self.device)
            positive = batch['positive'].to(self.device)
            negatives = batch['negatives'].to(self.device)

            # 前向传播: 提取特征
            anchor_feat = self.model(anchor)
            positive_feat = self.model(positive)

            # 处理negatives: [B, N, 9, H, W] -> [B*N, 9, H, W]
            # ===== 关键修改：逐个处理negatives，避免显存峰值 =====
            batch_size, num_neg = negatives.size(0), negatives.size(1)

            # 如果num_neg较小（<=10），可以一次性处理
            if num_neg <= 10:
                negatives_flat = negatives.view(-1, *negatives.size()[2:])
                negative_feat = self.model(negatives_flat)
                negative_feat = negative_feat.view(batch_size, num_neg, -1)
            else:
                # 分批处理negatives（更节省显存）
                negative_feat_list = []
                for i in range(num_neg):
                    neg_i = negatives[:, i, :, :, :]  # [B, 9, H, W]
                    neg_feat_i = self.model(neg_i)  # [B, D]
                    negative_feat_list.append(neg_feat_i)
                negative_feat = torch.stack(negative_feat_list, dim=1)  # [B, N, D]
            # ===== 修改结束 =====

            # 计算损失
            loss, stats = self.criterion(anchor_feat, positive_feat, negative_feat)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # 累积统计
            epoch_loss += loss.item()
            for key, value in stats.items():
                if key not in epoch_stats:
                    epoch_stats[key] = 0.0
                epoch_stats[key] += value

            num_batches += 1
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # 记录到TensorBoard（每10个batch）
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        # 计算平均值
        epoch_loss /= num_batches
        for key in epoch_stats:
            epoch_stats[key] /= num_batches

        return {'loss': epoch_loss, **epoch_stats}

    def validate(self) -> Dict:
        """验证"""
        self.model.eval()

        val_loss = 0.0
        val_stats = {}
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")

            for batch in pbar:
                anchor = batch['anchor'].to(self.device)
                positive = batch['positive'].to(self.device)
                negatives = batch['negatives'].to(self.device)

                # 前向传播
                anchor_feat = self.model(anchor)
                positive_feat = self.model(positive)

                batch_size, num_neg = negatives.size(0), negatives.size(1)

                if num_neg <= 10:
                    negatives_flat = negatives.view(-1, *negatives.size()[2:])
                    negative_feat = self.model(negatives_flat)
                    negative_feat = negative_feat.view(batch_size, num_neg, -1)
                else:
                    negative_feat_list = []
                    for i in range(num_neg):
                        neg_i = negatives[:, i, :, :, :]
                        neg_feat_i = self.model(neg_i)
                        negative_feat_list.append(neg_feat_i)
                    negative_feat = torch.stack(negative_feat_list, dim=1)

                # 计算损失
                loss, stats = self.criterion(anchor_feat, positive_feat, negative_feat)

                val_loss += loss.item()
                for key, value in stats.items():
                    if key not in val_stats:
                        val_stats[key] = 0.0
                    val_stats[key] += value

                num_batches += 1

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 计算平均值
        val_loss /= num_batches
        for key in val_stats:
            val_stats[key] /= num_batches

        # === 新增：统计STN角度分布 ===
        self._log_stn_angles()

        return {'loss': val_loss, **val_stats}

    def save_checkpoint(self, metric: float, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'metric': metric
        }

        # 保存最新的checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 如果是最佳，额外保存
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (metric={metric:.4f})")

        # 每5个epoch保存一次
        if (self.current_epoch + 1) % 5 == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'epoch_{self.current_epoch + 1}.pth')
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        return True

    def train(self, num_epochs: int, save_freq: int = 5):
        """完整训练循环"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 60)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # 训练
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 学习率调度
            if self.scheduler:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start_time

            # 打印结果
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")

            # TensorBoard记录
            self.writer.add_scalar('Epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/val_loss', val_metrics['loss'], epoch)

            # 保存checkpoint
            val_loss = val_metrics['loss']
            is_best = val_loss < self.best_metric if epoch > 0 else True

            if is_best:
                self.best_metric = val_loss

            self.save_checkpoint(val_loss, is_best)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_metric:.4f}")

        self.writer.close()

    def _log_stn_angles(self):
        """统计并打印STN学到的旋转角度分布"""
        # 检查模型是否使用STN
        if not hasattr(self.model, 'use_stn') or not self.model.use_stn:
            return

        # 从验证集采样100个样本
        import torch
        import numpy as np

        angles_list = []
        sample_count = 0
        max_samples = 100

        with torch.no_grad():
            for batch in self.val_loader:
                if sample_count >= max_samples:
                    break

                anchor = batch['anchor'].to(self.device)

                # 获取STN角度
                angles = self.model.get_stn_angle(anchor)  # [B, 1]
                angles_list.append(angles.cpu().numpy())

                sample_count += anchor.size(0)

        if len(angles_list) == 0:
            return

        # 合并所有角度
        all_angles = np.concatenate(angles_list, axis=0).flatten()  # [N]

        # 统计
        angle_min = float(np.min(all_angles))
        angle_max = float(np.max(all_angles))
        angle_mean = float(np.mean(all_angles))
        angle_std = float(np.std(all_angles))

        # 打印
        print(f"\n  === STN角度分布 ===")
        print(f"  样本数: {len(all_angles)}")
        print(f"  范围: [{angle_min:.1f}°, {angle_max:.1f}°]")
        print(f"  均值: {angle_mean:.1f}°")
        print(f"  标准差: {angle_std:.1f}°")

        # TensorBoard记录（如果可用）
        if self.writer:
            self.writer.add_scalar('STN/angle_min', angle_min, self.current_epoch)
            self.writer.add_scalar('STN/angle_max', angle_max, self.current_epoch)
            self.writer.add_scalar('STN/angle_mean', angle_mean, self.current_epoch)
            self.writer.add_scalar('STN/angle_std', angle_std, self.current_epoch)


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("Testing Trainer...")

    # 模拟模型和数据
    from torch.utils.data import DataLoader, TensorDataset

    # 简单模型
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32)
    )

    # 模拟数据
    train_data = TensorDataset(
        torch.randn(100, 128),
        torch.randn(100, 128),
        torch.randn(100, 10, 128)
    )
    train_loader = DataLoader(train_data, batch_size=8)

    # 优化器和损失
    from training.losses import TripletLoss

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = TripletLoss(margin=0.5)

    # 创建训练器（使用简化的接口进行测试）
    print("\n✓ Trainer基类测试完成!")