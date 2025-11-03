"""
Data Utilities: 数据处理通用工具
功能：
1. 读取Chilean点云文件路径
2. 读取训练/测试pickle文件
3. 数据归一化
4. 数据增强
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd


def load_pickle(filepath: str) -> Dict:
    """
    加载pickle文件

    Args:
        filepath: pickle文件路径

    Returns:
        data: 字典数据
    """
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data


def save_pickle(data: Dict, filepath: str):
    """
    保存pickle文件

    Args:
        data: 要保存的数据
        filepath: 保存路径
    """
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to: {filepath}")


def load_training_queries(pickle_path: str) -> Dict:
    """
    加载训练查询数据

    Args:
        pickle_path: training_queries_chilean_period.pickle路径

    Returns:
        queries: 查询字典
            key: 查询索引
            value: {
                'query': 点云文件路径,
                'positives': 正样本索引列表,
                'negatives': 负样本索引列表
            }
    """
    queries = load_pickle(pickle_path)
    print(f"Loaded {len(queries)} training queries from {pickle_path}")
    return queries


def load_test_queries(pickle_path: str) -> Dict:
    """
    加载测试查询数据（格式同训练集）

    Args:
        pickle_path: test_queries_chilean_period.pickle路径

    Returns:
        queries: 查询字典
    """
    queries = load_pickle(pickle_path)
    print(f"Loaded {len(queries)} test queries from {pickle_path}")
    return queries


def construct_pointcloud_path(base_path: str, relative_path: str) -> str:
    """
    构建完整的点云文件路径

    Args:
        base_path: 数据集根目录
        relative_path: 相对路径（从pickle中读取）

    Returns:
        full_path: 完整路径

    Example:
        base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
        relative_path = "chilean/100/pointcloud_20m_10overlap/1583230881342732339.bin"
        返回: "/home/wzj/pan2/.../chilean/100/pointcloud_20m_10overlap/1583230881342732339.bin"
    """
    full_path = os.path.join(base_path, relative_path)
    return full_path


def verify_pointcloud_exists(filepath: str) -> bool:
    """
    验证点云文件是否存在

    Args:
        filepath: 点云文件路径

    Returns:
        exists: 是否存在
    """
    return os.path.exists(filepath)


def normalize_bev(bev_layers: np.ndarray, vcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    归一化BEV数据

    Args:
        bev_layers: [num_layers, H, W]，值域[0, 255]
        vcd: [H, W]，值域[0, num_layers]

    Returns:
        bev_normalized: [num_layers, H, W]，归一化到[0, 1]
        vcd_normalized: [H, W]，归一化到[0, 1]
    """
    # BEV层归一化：除以255
    bev_normalized = bev_layers.astype(np.float32) / 255.0

    # VCD归一化：除以最大层数
    num_layers = bev_layers.shape[0]
    vcd_normalized = vcd.astype(np.float32) / float(num_layers)

    return bev_normalized, vcd_normalized


def stack_bev_with_vcd(bev_layers: np.ndarray, vcd: np.ndarray) -> np.ndarray:
    """
    将BEV层和VCD堆叠为单个张量

    Args:
        bev_layers: [num_layers, H, W]
        vcd: [H, W]

    Returns:
        stacked: [num_layers+1, H, W]
    """
    # 扩展VCD维度
    vcd_expanded = np.expand_dims(vcd, axis=0)  # [1, H, W]

    # 沿通道维度拼接
    stacked = np.concatenate([bev_layers, vcd_expanded], axis=0)

    return stacked


def augment_rotation(bev_stack: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    旋转增强

    Args:
        bev_stack: [C, H, W]
        angle_deg: 旋转角度（度）

    Returns:
        rotated: 旋转后的BEV
    """
    import cv2

    C, H, W = bev_stack.shape
    center = (W // 2, H // 2)

    # 构建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 对每个通道进行旋转
    rotated = np.zeros_like(bev_stack)
    for c in range(C):
        rotated[c] = cv2.warpAffine(
            bev_stack[c],
            rotation_matrix,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    return rotated


def augment_translation(bev_stack: np.ndarray,
                        shift_x_meter: float,
                        shift_y_meter: float,
                        resolution: float) -> np.ndarray:
    """
    平移增强

    Args:
        bev_stack: [C, H, W]
        shift_x_meter: x方向平移（米）
        shift_y_meter: y方向平移（米）
        resolution: BEV分辨率（米/像素）

    Returns:
        translated: 平移后的BEV
    """
    import cv2

    C, H, W = bev_stack.shape

    # 将米转换为像素
    shift_x_pixel = int(shift_x_meter / resolution)
    shift_y_pixel = int(shift_y_meter / resolution)

    # 构建平移矩阵
    translation_matrix = np.float32([[1, 0, shift_x_pixel], [0, 1, shift_y_pixel]])

    # 对每个通道进行平移
    translated = np.zeros_like(bev_stack)
    for c in range(C):
        translated[c] = cv2.warpAffine(
            bev_stack[c],
            translation_matrix,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    return translated


def apply_augmentation(bev_stack: np.ndarray,
                       config: Dict,
                       resolution: float = 0.2) -> np.ndarray:
    """
    应用数据增强（根据配置）

    Args:
        bev_stack: [C, H, W]
        config: 增强配置字典（来自config_base.yaml）
        resolution: BEV分辨率

    Returns:
        augmented: 增强后的BEV
    """
    augmented = bev_stack.copy()

    if not config.get('enabled', False):
        return augmented

    # 旋转增强
    if config.get('rotation', {}).get('enabled', False):
        if np.random.rand() < config['rotation'].get('probability', 0.5):
            angle_range = config['rotation'].get('range', [-5, 5])
            angle = np.random.uniform(angle_range[0], angle_range[1])
            augmented = augment_rotation(augmented, angle)

    # 平移增强
    if config.get('translation', {}).get('enabled', False):
        if np.random.rand() < config['translation'].get('probability', 0.5):
            trans_range = config['translation'].get('range', [-1.0, 1.0])
            shift_x = np.random.uniform(trans_range[0], trans_range[1])
            shift_y = np.random.uniform(trans_range[0], trans_range[1])
            augmented = augment_translation(augmented, shift_x, shift_y, resolution)

    return augmented


def get_dataset_statistics(queries: Dict, base_path: str) -> Dict:
    """
    计算数据集统计信息

    Args:
        queries: 查询字典
        base_path: 数据集根目录

    Returns:
        stats: 统计字典
    """
    total_queries = len(queries)
    total_positives = sum(len(q['positives']) for q in queries.values())
    total_negatives = sum(len(q['negatives']) for q in queries.values())

    # 验证文件存在性
    existing_files = 0
    for query_data in queries.values():
        filepath = construct_pointcloud_path(base_path, query_data['query'])
        if verify_pointcloud_exists(filepath):
            existing_files += 1

    stats = {
        'total_queries': total_queries,
        'total_positives': total_positives,
        'total_negatives': total_negatives,
        'avg_positives_per_query': total_positives / total_queries if total_queries > 0 else 0,
        'avg_negatives_per_query': total_negatives / total_queries if total_queries > 0 else 0,
        'existing_files': existing_files,
        'missing_files': total_queries - existing_files,
    }

    return stats


# ========== 测试代码 ==========
if __name__ == "__main__":
    """
    测试数据工具
    用法: python data/data_utils.py
    """
    print("Testing Data Utilities...")

    # 1. 测试加载训练查询
    train_pickle = "training_queries_chilean_period.pickle"
    if os.path.exists(train_pickle):
        print(f"\n加载训练查询...")
        train_queries = load_training_queries(train_pickle)

        # 显示第一个查询
        first_key = list(train_queries.keys())[0]
        first_query = train_queries[first_key]
        print(f"\n第一个查询示例:")
        print(f"  Query file: {first_query['query']}")
        print(f"  Positives: {len(first_query['positives'])} samples")
        print(f"  Negatives: {len(first_query['negatives'])} samples")

        # 统计信息
        base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
        stats = get_dataset_statistics(train_queries, base_path)
        print(f"\n训练集统计:")
        print(f"  总查询数: {stats['total_queries']}")
        print(f"  总正样本数: {stats['total_positives']}")
        print(f"  总负样本数: {stats['total_negatives']}")
        print(f"  平均正样本/查询: {stats['avg_positives_per_query']:.1f}")
        print(f"  平均负样本/查询: {stats['avg_negatives_per_query']:.1f}")
        print(f"  存在的文件: {stats['existing_files']}/{stats['total_queries']}")

        # 测试路径构建
        test_relative_path = first_query['query']
        test_full_path = construct_pointcloud_path(base_path, test_relative_path)
        print(f"\n路径构建测试:")
        print(f"  相对路径: {test_relative_path}")
        print(f"  完整路径: {test_full_path}")
        print(f"  文件存在: {verify_pointcloud_exists(test_full_path)}")
    else:
        print(f"Warning: 找不到 {train_pickle}")
        print("请先运行 generate_training_tuples_baseline_chilean_period.py")

        # 2. 测试BEV归一化
    print("\n\n测试BEV归一化...")

    # 加载测试BEV数据
    test_bev_path = "/home/wzj/pan1/contour_deep_1++/data/test_bev_output.npz"
    if os.path.exists(test_bev_path):
        loaded = np.load(test_bev_path)
        bev_layers = loaded['bev_layers']
        vcd = loaded['vcd']

        print(f"原始BEV:")
        print(f"  BEV值域: [{bev_layers.min()}, {bev_layers.max()}]")
        print(f"  VCD值域: [{vcd.min()}, {vcd.max()}]")

        # 归一化
        bev_norm, vcd_norm = normalize_bev(bev_layers, vcd)
        print(f"归一化后:")
        print(f"  BEV值域: [{bev_norm.min():.3f}, {bev_norm.max():.3f}]")
        print(f"  VCD值域: [{vcd_norm.min():.3f}, {vcd_norm.max():.3f}]")

        # 堆叠
        stacked = stack_bev_with_vcd(bev_norm, vcd_norm)
        print(f"堆叠后形状: {stacked.shape}")  # 应该是 [9, 200, 200]

        # 3. 测试数据增强
        print("\n\n测试数据增强...")

        # 旋转测试
        print("旋转增强测试 (5度)...")
        rotated = augment_rotation(stacked, angle_deg=5.0)
        print(f"  旋转前非零像素: {np.sum(stacked > 0)}")
        print(f"  旋转后非零像素: {np.sum(rotated > 0)}")

        # 平移测试
        print("平移增强测试 (0.5米)...")
        translated = augment_translation(stacked, shift_x_meter=0.5, shift_y_meter=0.5, resolution=0.2)
        print(f"  平移前非零像素: {np.sum(stacked > 0)}")
        print(f"  平移后非零像素: {np.sum(translated > 0)}")

        # 组合增强测试
        print("组合增强测试...")
        aug_config = {
            'enabled': True,
            'rotation': {
                'enabled': True,
                'range': [-5, 5],
                'probability': 1.0  # 100%触发用于测试
            },
            'translation': {
                'enabled': True,
                'range': [-1.0, 1.0],
                'probability': 1.0
            }
        }
        augmented = apply_augmentation(stacked, aug_config, resolution=0.2)
        print(f"  增强前非零像素: {np.sum(stacked > 0)}")
        print(f"  增强后非零像素: {np.sum(augmented > 0)}")

    else:
        print(f"Warning: 找不到 {test_bev_path}")
        print("请先运行 python data/bev_generator.py")

    # 4. 测试加载测试查询
    test_pickle = "test_queries_chilean_period.pickle"
    if os.path.exists(test_pickle):
        print(f"\n\n加载测试查询...")
        test_queries = load_test_queries(test_pickle)

        # 统计信息
        stats = get_dataset_statistics(test_queries, base_path)
        print(f"\n测试集统计:")
        print(f"  总查询数: {stats['total_queries']}")
        print(f"  总正样本数: {stats['total_positives']}")
        print(f"  平均正样本/查询: {stats['avg_positives_per_query']:.1f}")
    else:
        print(f"\nWarning: 找不到 {test_pickle}")

    print("\n✓ Data Utilities测试完成!")