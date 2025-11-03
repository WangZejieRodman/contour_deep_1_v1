"""
自适应BEV预处理脚本（预处理阶段不增强）
用法: python scripts/preprocess_bev_adaptive.py --split train
      python scripts/preprocess_bev_adaptive.py --split test
      python scripts/preprocess_bev_adaptive.py --split all
"""

import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.bev_generator_adaptive import (
    AdaptiveBEVGenerator,
    load_chilean_pointcloud,
    load_config_from_yaml
)
from data.data_utils import load_pickle


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocess_adaptive_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def process_single_pointcloud(pointcloud_path: str,
                              query_id: int,
                              generator: AdaptiveBEVGenerator,
                              base_path: str):
    """
    处理单个点云（生成自适应BEV，不增强）

    Args:
        pointcloud_path: 点云相对路径
        query_id: 查询ID
        generator: 自适应BEV生成器
        base_path: 数据集根目录

    Returns:
        (bev_layers, vcd, metadata, success)
    """
    full_path = os.path.join(base_path, pointcloud_path)

    if not os.path.exists(full_path):
        logging.warning(f"点云文件不存在: {full_path}")
        return None, None, None, False

    try:
        # 1. 加载点云
        pointcloud = load_chilean_pointcloud(full_path)

        if len(pointcloud) < 100:
            logging.warning(f"点云太小 ({len(pointcloud)} points): {pointcloud_path}")
            return None, None, None, False

        # 2. 生成自适应BEV（不应用随机变换）
        bev_layers, vcd, metadata = generator.make_bev(pointcloud, str_id=str(query_id))

        return bev_layers, vcd, metadata, True

    except Exception as e:
        logging.error(f"处理失败 {pointcloud_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, False


def preprocess_split(split: str,
                     config_path: str,
                     base_path: str,
                     cache_root: str,
                     overwrite: bool = False):
    """
    预处理数据集（自适应分层，不增强）

    Args:
        split: 'train' 或 'test'
        config_path: 配置文件路径
        base_path: 数据集根目录
        cache_root: 缓存根目录
        overwrite: 是否覆盖已存在的缓存
    """
    logging.info(f"{'='*60}")
    logging.info(f"处理 {split} 集 (自适应分层, 不增强)")
    logging.info(f"{'='*60}")

    # 1. 加载配置
    config = load_config_from_yaml(config_path)
    generator = AdaptiveBEVGenerator(config, layer_thickness=0.625)
    logging.info(f"BEV配置: 固定层厚{generator.layer_thickness}m, {generator.num_layers}层")

    # 2. 加载查询数据
    if split == 'train':
        pickle_file = os.path.join(os.path.dirname(cache_root),
                                   "training_queries_chilean_period.pickle")
    else:
        pickle_file = os.path.join(os.path.dirname(cache_root),
                                   "test_queries_chilean_period.pickle")

    if not os.path.exists(pickle_file):
        logging.error(f"Pickle文件不存在: {pickle_file}")
        return 0, 0, 0

    queries = load_pickle(pickle_file)
    logging.info(f"加载 {len(queries)} 个查询")

    # 3. 创建缓存目录（添加_adaptive后缀）
    cache_dir = os.path.join(cache_root, f"{split}_adaptive")
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"缓存目录: {cache_dir}")

    # 4. 批量处理
    success_count = 0
    skip_count = 0
    fail_count = 0

    # 统计信息
    total_discard_ratio = []

    for query_id, query_data in tqdm(queries.items(), desc=f"Processing {split}"):
        cache_filename = f"{query_id:06d}.npz"
        cache_path = os.path.join(cache_dir, cache_filename)

        # 检查是否已存在
        if os.path.exists(cache_path) and not overwrite:
            skip_count += 1
            continue

        # 处理点云
        pointcloud_path = query_data['query']
        bev_layers, vcd, metadata, success = process_single_pointcloud(
            pointcloud_path, query_id, generator, base_path
        )

        if success:
            # 保存缓存（包含元数据）
            np.savez_compressed(
                cache_path,
                bev_layers=bev_layers,
                vcd=vcd,
                query_id=query_id,
                pointcloud_path=pointcloud_path,
                # 保存自适应分层信息
                lv_grads=metadata['lv_grads'],
                z_center=metadata['z_center'],
                z_min=metadata['z_min'],
                z_max=metadata['z_max'],
                layer_thickness=metadata['layer_thickness'],
                num_layers=metadata['num_layers'],
                points_total=metadata['points_total'],
                points_used=metadata['points_used'],
                points_discarded=metadata['points_discarded'],
                discard_ratio=metadata['discard_ratio']
            )
            success_count += 1
            total_discard_ratio.append(metadata['discard_ratio'])
        else:
            fail_count += 1

    # 5. 统计报告
    logging.info(f"\n{'='*60}")
    logging.info(f"{split.upper()} 集处理完成!")
    logging.info(f"  成功: {success_count}")
    logging.info(f"  跳过: {skip_count}")
    logging.info(f"  失败: {fail_count}")
    logging.info(f"  总计: {len(queries)}")

    if total_discard_ratio:
        avg_discard = np.mean(total_discard_ratio) * 100
        max_discard = np.max(total_discard_ratio) * 100
        logging.info(f"  平均丢弃点比例: {avg_discard:.2f}%")
        logging.info(f"  最大丢弃点比例: {max_discard:.2f}%")

    logging.info(f"{'='*60}\n")

    return success_count, skip_count, fail_count


def verify_cache(cache_dir: str, expected_count: int):
    """
    验证缓存完整性

    Args:
        cache_dir: 缓存目录
        expected_count: 预期文件数量
    """
    logging.info(f"\n验证缓存: {cache_dir}")

    # 1. 检查文件数量
    cache_files = list(Path(cache_dir).glob("*.npz"))
    actual_count = len(cache_files)

    logging.info(f"  文件数量: {actual_count}/{expected_count}")

    if actual_count != expected_count:
        logging.warning(f"  ⚠️  文件数量不匹配!")
        return False

    # 2. 随机抽取10个文件验证
    import random
    sample_files = random.sample(cache_files, min(10, len(cache_files)))

    for cache_file in sample_files:
        try:
            data = np.load(cache_file)
            assert 'bev_layers' in data
            assert 'vcd' in data
            assert 'lv_grads' in data
            assert 'z_center' in data
            assert data['bev_layers'].shape == (8, 200, 200)
            assert data['vcd'].shape == (200, 200)
            assert len(data['lv_grads']) == 9  # 8层+边界
        except Exception as e:
            logging.error(f"  ✗ 文件损坏: {cache_file.name} - {e}")
            return False

    logging.info(f"  ✓ 随机验证 {len(sample_files)} 个文件通过")

    # 3. 计算磁盘占用
    total_size = sum(f.stat().st_size for f in cache_files)
    size_gb = total_size / (1024 ** 3)
    avg_size_kb = (total_size / len(cache_files)) / 1024

    logging.info(f"  磁盘占用: {size_gb:.2f} GB")
    logging.info(f"  平均文件大小: {avg_size_kb:.1f} KB")

    return True


def main():
    parser = argparse.ArgumentParser(description='自适应BEV预处理（不增强）')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'test', 'all'],
                        help='处理哪个数据集')
    parser.add_argument('--config', type=str,
                        default='/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml',
                        help='配置文件路径')
    parser.add_argument('--base_path', type=str,
                        default='/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/',
                        help='数据集根目录')
    parser.add_argument('--cache_root', type=str,
                        default='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/',
                        help='缓存根目录')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的缓存')

    args = parser.parse_args()

    # 设置日志
    log_file = setup_logging('logs')
    logging.info(f"日志文件: {log_file}")
    logging.info(f"配置: {args}")

    # 处理数据集
    if args.split in ['train', 'all']:
        train_stats = preprocess_split(
            'train', args.config, args.base_path, args.cache_root, args.overwrite
        )

    if args.split in ['test', 'all']:
        test_stats = preprocess_split(
            'test', args.config, args.base_path, args.cache_root, args.overwrite
        )

    # 验证缓存
    if args.split in ['train', 'all']:
        verify_cache(os.path.join(args.cache_root, 'train_adaptive'), 1850)

    if args.split in ['test', 'all']:
        verify_cache(os.path.join(args.cache_root, 'test_adaptive'), 1750)

    logging.info("\n✓ 自适应预处理完成!")


if __name__ == "__main__":
    main()