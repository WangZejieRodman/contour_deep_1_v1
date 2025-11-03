# training/train_retrieval_adaptive.py
"""
Training Script for Direction 1: Retrieval Feature Network (Adaptive Version)
自适应分层版本的训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.retrieval_net import RetrievalNet
from data.dataset_retrieval_adaptive import AdaptiveRetrievalDataset, create_dataloader  # 关键修改：导入自适应数据集
from training.losses import TripletLoss, InfoNCELoss
from training.trainer import BaseTrainer


def create_optimizer(model, config):
    """创建优化器"""
    optimizer_name = config.get('optimizer', 'adamw')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-4)

    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, config, num_epochs):
    """创建学习率调度器"""
    scheduler_name = config.get('scheduler', 'cosine')
    warmup_epochs = config.get('warmup_epochs', 5)

    if scheduler_name.lower() == 'cosine':
        # 带Warmup的CosineAnnealing
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

        # Warmup调度器（手动实现）
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # 组合调度器
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs]
        )
    elif scheduler_name.lower() == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = None

    return scheduler


def create_criterion(config):
    """创建损失函数"""
    loss_name = config.get('loss', 'triplet')

    if loss_name.lower() == 'triplet':
        margin = config.get('margin', 0.5)
        mining = config.get('mining', 'hard')
        criterion = TripletLoss(margin=margin, mining=mining)
    elif loss_name.lower() == 'infonce':
        temperature = config.get('temperature', 0.07)
        criterion = InfoNCELoss(temperature=temperature)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    return criterion


def main(args):
    # 加载配置
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)

    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 创建数据集（使用自适应版本）
    print("\nCreating adaptive datasets...")
    train_dataset = AdaptiveRetrievalDataset(  # 关键修改：使用AdaptiveRetrievalDataset
        queries_pickle=args.train_pickle,
        cache_root=base_config['preprocessing']['cache_dir'],
        split='train',
        num_negatives=train_config['data']['num_negatives'],
        augmentation_config=base_config['augmentation'],  # 训练时应用增强
        resolution=base_config['bev']['resolution'],
        use_cache=True,
        max_cache_size=train_config['data']['max_cache_size']
    )

    val_dataset = AdaptiveRetrievalDataset(  # 关键修改：使用AdaptiveRetrievalDataset
        queries_pickle=args.test_pickle,
        cache_root=base_config['preprocessing']['cache_dir'],
        split='test',
        num_negatives=train_config['data']['num_negatives'],
        augmentation_config=base_config['augmentation'],  # 验证集不增强
        resolution=base_config['bev']['resolution'],
        use_cache=True
    )

    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config['training']['batch_size'],
        num_workers=train_config['training']['num_workers'],
        shuffle=True
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config['training']['batch_size'],
        num_workers=train_config['training']['num_workers'],
        shuffle=False
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # 创建模型
    print("\nCreating model...")
    model = RetrievalNet(output_dim=train_config['model']['output_dim'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 创建优化器、调度器、损失函数
    optimizer = create_optimizer(model, train_config['optimizer'])
    scheduler = create_scheduler(optimizer, train_config['scheduler'], train_config['training']['num_epochs'])
    criterion = create_criterion(train_config['loss'])

    print(f"  Optimizer: {optimizer.__class__.__name__}")
    print(f"  Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    print(f"  Loss: {criterion.__class__.__name__}")

    # 创建训练器
    print("\nInitializing trainer...")
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        log_dir=train_config['logging']['log_dir'],
        checkpoint_dir=train_config['logging']['checkpoint_dir'],
        experiment_name=args.experiment_name
    )

    # 加载checkpoint（如果指定）
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # 开始训练
    print("\n" + "=" * 60)
    print("Starting training (Adaptive BEV + Dynamic Augmentation)...")
    print("=" * 60)

    trainer.train(
        num_epochs=train_config['training']['num_epochs'],
        save_freq=train_config['training']['save_freq']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Retrieval Network (Adaptive Version)')

    parser.add_argument('--config', type=str,
                        default='/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml',  # 修改路径
                        help='Base config file')
    parser.add_argument('--train_config', type=str,
                        default='/home/wzj/pan1/contour_deep_1++/configs/config_retrieval.yaml',  # 修改路径
                        help='Training config file')
    parser.add_argument('--train_pickle', type=str,
                        default='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/training_queries_chilean_period.pickle',  # 修改路径
                        help='Training queries pickle')
    parser.add_argument('--test_pickle', type=str,
                        default='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/test_queries_chilean_period.pickle',  # 修改路径
                        help='Test queries pickle')
    parser.add_argument('--experiment_name', type=str,
                        default='retrieval_adaptive',  # 修改实验名称
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    main(args)