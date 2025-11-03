"""
自适应RetrievalDataset: 加载自适应BEV缓存 + 训练时动态增强
功能：
1. 加载预处理好的自适应BEV缓存（train_adaptive/, test_adaptive/）
2. 训练时动态应用BEV级别的旋转/平移增强
3. 所有BEV层使用相同的变换参数
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import Dict, List, Tuple, Optional
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils import (
    load_pickle,
    normalize_bev,
    stack_bev_with_vcd,
    apply_augmentation  # 复用原有增强函数
)


class AdaptiveRetrievalDataset(Dataset):
    """自适应检索数据集（加载自适应BEV + 训练时动态增强）"""

    def __init__(self,
                 queries_pickle: str,
                 cache_root: str,
                 split: str = 'train',
                 num_negatives: int = 10,
                 augmentation_config: Optional[Dict] = None,
                 resolution: float = 0.2,
                 use_cache: bool = True,
                 max_cache_size: int = 1000):
        """
        初始化数据集

        Args:
            queries_pickle: 查询pickle文件路径
            cache_root: BEV缓存根目录
            split: 'train' 或 'test'
            num_negatives: 每个anchor采样的负样本数量
            augmentation_config: 数据增强配置（来自config_base.yaml）
            resolution: BEV分辨率（用于增强）
            use_cache: 是否使用内存缓存
            max_cache_size: 最大缓存数量
        """
        self.split = split
        # 加载自适应BEV缓存目录
        self.cache_dir = os.path.join(cache_root, f"{split}_adaptive")
        self.num_negatives = num_negatives
        self.aug_config = augmentation_config
        self.resolution = resolution
        self.use_cache = use_cache

        # 加载查询数据
        self.queries = load_pickle(queries_pickle)
        self.query_keys = sorted(self.queries.keys())

        print(f"[{split}] Loaded {len(self.query_keys)} queries (adaptive BEV)")

        # 内存缓存
        if self.use_cache:
            from collections import OrderedDict
            self.memory_cache = OrderedDict()
            self.max_cache_size = max_cache_size

        # 统计
        self._compute_statistics()

    def _compute_statistics(self):
        """计算统计信息"""
        total_positives = 0
        total_negatives = 0

        for query_data in self.queries.values():
            total_positives += len(query_data['positives'])
            total_negatives += len(query_data['negatives'])

        self.stats = {
            'total_queries': len(self.queries),
            'avg_positives': total_positives / len(self.queries),
            'avg_negatives': total_negatives / len(self.queries),
        }

        print(f"[{self.split}] Statistics:")
        print(f"  Avg positives: {self.stats['avg_positives']:.1f}")
        print(f"  Avg negatives: {self.stats['avg_negatives']:.1f}")

    def __len__(self) -> int:
        return len(self.query_keys)

    def _load_bev_from_cache(self, query_idx: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        从自适应BEV缓存加载

        Args:
            query_idx: 查询索引

        Returns:
            (bev_layers, vcd) 或 None
        """
        # 检查内存缓存
        if self.use_cache and query_idx in self.memory_cache:
            self.memory_cache.move_to_end(query_idx)
            return self.memory_cache[query_idx]

        # 从磁盘加载
        cache_filename = f"{query_idx:06d}.npz"
        cache_path = os.path.join(self.cache_dir, cache_filename)

        if not os.path.exists(cache_path):
            print(f"Warning: Cache not found for query {query_idx}")
            return None

        try:
            data = np.load(cache_path, mmap_mode='r' if not self.use_cache else None)
            bev_layers = data['bev_layers']
            vcd = data['vcd']

            # 更新内存缓存
            if self.use_cache:
                if len(self.memory_cache) >= self.max_cache_size:
                    self.memory_cache.popitem(last=False)
                self.memory_cache[query_idx] = (bev_layers.copy(), vcd.copy())

            return bev_layers, vcd

        except Exception as e:
            print(f"Error loading cache {query_idx}: {e}")
            return None

    def _sample_triplet(self, query_key: int) -> Tuple[int, int, List[int]]:
        """
        采样三元组

        Args:
            query_key: anchor查询键

        Returns:
            (anchor_idx, positive_idx, negative_indices)
        """
        query_data = self.queries[query_key]

        anchor_idx = query_key

        # Positive
        positives = query_data['positives']
        if len(positives) == 0:
            positive_idx = anchor_idx
        else:
            positive_idx = random.choice(positives)

        # Negatives
        negatives = query_data['negatives']
        if len(negatives) >= self.num_negatives:
            negative_indices = random.sample(negatives, self.num_negatives)
        else:
            negative_indices = random.choices(negatives, k=self.num_negatives)

        return anchor_idx, positive_idx, negative_indices

    def _preprocess_bev(self, bev_layers: np.ndarray, vcd: np.ndarray,
                        apply_aug: bool = False) -> torch.Tensor:
        """
        预处理BEV: 归一化、堆叠、增强（训练时）、转tensor

        Args:
            bev_layers: [8, H, W]
            vcd: [H, W]
            apply_aug: 是否应用数据增强

        Returns:
            tensor: [9, H, W]
        """
        # 1. 归一化
        bev_norm, vcd_norm = normalize_bev(bev_layers, vcd)

        # 2. 堆叠（所有层+VCD）
        stacked = stack_bev_with_vcd(bev_norm, vcd_norm)  # [9, 200, 200]

        # 3. 数据增强（仅训练集，动态应用）
        if apply_aug and self.aug_config is not None:
            # 复用原有的apply_augmentation函数
            # 会对所有9个通道应用相同的旋转/平移
            stacked = apply_augmentation(stacked, self.aug_config, self.resolution)

        # 4. 转tensor
        tensor = torch.from_numpy(stacked).float()

        return tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            {
                'anchor': [9, H, W],
                'positive': [9, H, W],
                'negatives': [num_neg, 9, H, W],
                'anchor_idx': int,
                'positive_idx': int,
                'negative_indices': [num_neg]
            }
        """
        query_key = self.query_keys[idx]

        # 1. 采样三元组
        anchor_idx, positive_idx, negative_indices = self._sample_triplet(query_key)

        # 2. 加载BEV
        anchor_bev = self._load_bev_from_cache(anchor_idx)
        positive_bev = self._load_bev_from_cache(positive_idx)

        if anchor_bev is None or positive_bev is None:
            dummy = torch.zeros(9, 200, 200)
            return {
                'anchor': dummy,
                'positive': dummy,
                'negatives': torch.zeros(self.num_negatives, 9, 200, 200),
                'anchor_idx': anchor_idx,
                'positive_idx': positive_idx,
                'negative_indices': negative_indices
            }

        # 3. 预处理（训练时应用增强）
        apply_aug = (self.split == 'train')

        anchor_tensor = self._preprocess_bev(*anchor_bev, apply_aug=apply_aug)
        positive_tensor = self._preprocess_bev(*positive_bev, apply_aug=apply_aug)

        # 4. 加载负样本
        negative_tensors = []
        valid_negative_indices = []

        for neg_idx in negative_indices:
            neg_bev = self._load_bev_from_cache(neg_idx)
            if neg_bev is not None:
                # 负样本也应用增强（训练时）
                neg_tensor = self._preprocess_bev(*neg_bev, apply_aug=apply_aug)
                negative_tensors.append(neg_tensor)
                valid_negative_indices.append(neg_idx)

        # 填充
        while len(negative_tensors) < self.num_negatives:
            negative_tensors.append(torch.zeros_like(anchor_tensor))
            valid_negative_indices.append(-1)

        negatives_tensor = torch.stack(negative_tensors)

        return {
            'anchor': anchor_tensor,
            'positive': positive_tensor,
            'negatives': negatives_tensor,
            'anchor_idx': anchor_idx,
            'positive_idx': positive_idx,
            'negative_indices': valid_negative_indices
        }


def create_dataloader(dataset, batch_size=8, num_workers=4, shuffle=True):
    """创建DataLoader"""

    def collate_fn(batch):
        anchors = torch.stack([item['anchor'] for item in batch])
        positives = torch.stack([item['positive'] for item in batch])
        negatives = torch.stack([item['negatives'] for item in batch])

        return {
            'anchor': anchors,
            'positive': positives,
            'negatives': negatives,
            'anchor_idx': [item['anchor_idx'] for item in batch],
            'positive_idx': [item['positive_idx'] for item in batch],
            'negative_indices': [item['negative_indices'] for item in batch]
        }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )


# ========== 测试代码 ==========
if __name__ == "__main__":
    """
    测试AdaptiveRetrievalDataset
    用法: python data/dataset_retrieval_adaptive.py
    """
    import yaml
    import time

    print("测试AdaptiveRetrievalDataset...")

    # 1. 加载配置
    config_path = "/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. 创建训练数据集
    print("\n=== 创建训练数据集（带增强） ===")
    train_dataset = AdaptiveRetrievalDataset(
        queries_pickle="/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/training_queries_chilean_period.pickle",
        cache_root="/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/",
        split='train',
        num_negatives=10,
        augmentation_config=config['augmentation'],  # 传入增强配置
        resolution=config['bev']['resolution'],
        use_cache=True,
        max_cache_size=500
    )

    print(f"数据集大小: {len(train_dataset)}")

    # 3. 测试单个样本
    print("\n=== 测试单个样本 ===")
    sample = train_dataset[0]
    print(f"Anchor shape: {sample['anchor'].shape}")
    print(f"Positive shape: {sample['positive'].shape}")
    print(f"Negatives shape: {sample['negatives'].shape}")

    # 4. 创建DataLoader并测试
    print("\n=== 测试DataLoader ===")
    train_loader = create_dataloader(
        train_dataset,
        batch_size=4,
        num_workers=2,
        shuffle=True
    )

    print(f"Batches per epoch: {len(train_loader)}")

    # 测试加载速度
    print("\n=== 测试加载速度 ===")
    start_time = time.time()
    num_batches_to_test = 10

    for i, batch in enumerate(train_loader):
        if i >= num_batches_to_test:
            break
        print(f"Batch {i + 1}: Anchor {batch['anchor'].shape}")

    elapsed = time.time() - start_time
    samples_per_sec = (num_batches_to_test * 4) / elapsed

    print(f"\n加载速度: {samples_per_sec:.1f} samples/sec")

    # 5. 测试增强效果（同一样本两次加载应该不同）
    print("\n=== 测试增强效果 ===")
    sample1 = train_dataset[0]
    sample2 = train_dataset[0]

    diff = torch.abs(sample1['anchor'] - sample2['anchor']).sum().item()
    print(f"同一样本两次加载的差异: {diff:.2f}")

    if diff > 0:
        print("✓ 增强正常工作（每次加载结果不同）")
    else:
        print("✗ 增强未生效（每次加载结果相同）")

    print("\n✓ AdaptiveRetrievalDataset测试完成!")