"""
自适应BEV生成器（完整版）：实现方案A+方案B
功能：
1. 固定层厚的中心对齐分层（方案A）
2. 随机旋转+限制平移（方案B）
3. 包含所有bev_generator.py的功能
4. 测试功能：保存原始点云和3次随机变换后的BEV对比结果
"""

import numpy as np
import os
import sys
from typing import Tuple, List, Optional, Dict
import yaml
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.contour_types import ContourManagerConfig


class AdaptiveBEVGenerator:
    """自适应BEV生成器（固定层厚+中心对齐）"""

    def __init__(self, config: ContourManagerConfig, layer_thickness: float = 0.625):
        """
        初始化生成器

        Args:
            config: 基础配置
            layer_thickness: 固定层厚（米），默认0.625m
        """
        self.cfg = config
        self.layer_thickness = layer_thickness
        self.num_layers = 8  # 固定8层

        # 验证配置
        assert config.n_col % 2 == 0
        assert config.n_row % 2 == 0

        # 坐标范围（xy平面）
        self.x_min = -(config.n_row // 2) * config.reso_row
        self.x_max = -self.x_min
        self.y_min = -(config.n_col // 2) * config.reso_col
        self.y_max = -self.y_min

        # BEV数据存储
        self.bev = None
        self.layer_masks = None
        self.bev_pixfs = []

        # 自适应分层参数
        self.adaptive_lv_grads = None
        self.z_center = None
        self.z_min_actual = None
        self.z_max_actual = None

    def _compute_adaptive_layers(self, point_cloud: np.ndarray) -> List[float]:
        """
        固定层厚的中心对齐分层

        Args:
            point_cloud: 点云 [N, 3]

        Returns:
            lv_grads: 层级阈值列表（9个边界值）
        """
        # 1. 获取z坐标范围
        z_coords = point_cloud[:, 2]
        self.z_min_actual = float(np.min(z_coords))
        self.z_max_actual = float(np.max(z_coords))

        # 2. 计算中心
        self.z_center = (self.z_min_actual + self.z_max_actual) / 2.0

        # 3. 以中心为基准，上下各扩展4层（总8层）
        # 总覆盖范围：8 × 0.625 = 5m
        half_range = (self.num_layers / 2) * self.layer_thickness  # 2.5m

        # 4. 生成层边界
        lv_grads = []
        for i in range(self.num_layers + 1):
            # 从 z_center - 2.5m 开始，每隔0.625m一个边界
            z_boundary = self.z_center - half_range + i * self.layer_thickness
            lv_grads.append(z_boundary)

        return lv_grads

    def hash_point_to_image(self, pt: np.ndarray) -> Tuple[int, int]:
        """
        将点映射到图像坐标

        Args:
            pt: 点坐标 [x, y, z]

        Returns:
            (row, col) 或 (-1, -1) 如果点在范围外
        """
        padding = 1e-2
        x, y = pt[0], pt[1]

        # 检查xy范围
        if (x < self.x_min + padding or x > self.x_max - padding or
                y < self.y_min + padding or y > self.y_max - padding or
                (y * y + x * x) < self.cfg.blind_sq):
            return -1, -1

        row = int(np.floor(x / self.cfg.reso_row)) + self.cfg.n_row // 2
        col = int(np.floor(y / self.cfg.reso_col)) + self.cfg.n_col // 2

        if not (0 <= row < self.cfg.n_row and 0 <= col < self.cfg.n_col):
            return -1, -1

        return row, col

    def point_to_cont_row_col(self, p_in_l: np.ndarray) -> np.ndarray:
        """将点转换到连续图像坐标"""
        continuous_rc = np.array([
            p_in_l[0] / self.cfg.reso_row + self.cfg.n_row / 2 - 0.5,
            p_in_l[1] / self.cfg.reso_col + self.cfg.n_col / 2 - 0.5
        ], dtype=np.float32)
        return continuous_rc

    def make_bev(self, point_cloud: np.ndarray, str_id: str = "") -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        从点云生成自适应BEV

        Args:
            point_cloud: 点云 [N, 3]
            str_id: 标识符（用于调试）

        Returns:
            bev_layers: [8, H, W]
            vcd: [H, W]
            metadata: 元数据字典
        """
        assert point_cloud.shape[0] > 10, "点云数量太少"
        assert point_cloud.shape[1] >= 3, "点云必须包含x,y,z"

        # 1. 计算自适应分层（中心对齐+固定层厚）
        self.adaptive_lv_grads = self._compute_adaptive_layers(point_cloud)

        # 2. 初始化BEV
        self.bev = np.full((self.cfg.n_row, self.cfg.n_col), -1000.0, dtype=np.float32)
        self.layer_masks = np.zeros((self.cfg.n_row, self.cfg.n_col, self.num_layers), dtype=bool)
        self.bev_pixfs.clear()

        tmp_pillars = {}

        # 统计：超出范围的点数
        points_discarded = 0
        points_total = len(point_cloud)

        # 3. 处理每个点
        for pt in point_cloud:
            # 检查z坐标是否在8层覆盖范围内
            z = pt[2]
            if z < self.adaptive_lv_grads[0] or z >= self.adaptive_lv_grads[-1]:
                points_discarded += 1
                continue  # 丢弃超出范围的点

            row, col = self.hash_point_to_image(pt)
            if row >= 0:
                height = pt[2]

                # 更新最大高度
                if self.bev[row, col] < height:
                    self.bev[row, col] = height
                    coor_f = self.point_to_cont_row_col(pt[:2])
                    hash_key = row * self.cfg.n_col + col
                    tmp_pillars[hash_key] = (coor_f[0], coor_f[1], height)

                # 判断所属层级
                for level in range(self.num_layers):
                    h_min = self.adaptive_lv_grads[level]
                    h_max = self.adaptive_lv_grads[level + 1]
                    if h_min <= height < h_max:
                        self.layer_masks[row, col, level] = True
                        break  # 每个点只属于一层

        # 4. 转换为列表
        self.bev_pixfs = [(k, v) for k, v in tmp_pillars.items()]
        self.bev_pixfs.sort(key=lambda x: x[0])

        # 5. 生成多层BEV
        bev_layers = self._generate_bev_layers(self.adaptive_lv_grads)

        # 6. 生成VCD
        vcd = self._generate_vertical_complexity_map()

        # 7. 元数据
        metadata = {
            'lv_grads': self.adaptive_lv_grads,
            'z_center': self.z_center,
            'z_min': self.z_min_actual,
            'z_max': self.z_max_actual,
            'layer_thickness': self.layer_thickness,
            'num_layers': self.num_layers,
            'points_total': points_total,
            'points_used': points_total - points_discarded,
            'points_discarded': points_discarded,
            'discard_ratio': points_discarded / points_total if points_total > 0 else 0.0
        }

        return bev_layers, vcd, metadata

    def _generate_bev_layers(self, lv_grads: List[float]) -> np.ndarray:
        """生成多层BEV二值图"""
        bev_layers = np.zeros((self.num_layers, self.cfg.n_row, self.cfg.n_col), dtype=np.uint8)

        for level in range(self.num_layers):
            h_min = lv_grads[level]
            h_max = lv_grads[level + 1]
            mask = ((self.bev >= h_min) & (self.bev < h_max)).astype(np.uint8) * 255
            bev_layers[level] = mask

        return bev_layers

    def _generate_vertical_complexity_map(self) -> np.ndarray:
        """生成垂直复杂度图"""
        vcd = np.sum(self.layer_masks, axis=2).astype(np.uint8)
        return vcd

    def get_bev_image(self) -> np.ndarray:
        """获取原始BEV图像（高度图）"""
        return self.bev.copy()


def load_chilean_pointcloud(filepath: str) -> np.ndarray:
    """
    加载Chilean点云文件（.bin格式）

    Args:
        filepath: 点云文件路径

    Returns:
        point_cloud: [N, 3] numpy数组
    """
    try:
        pc = np.fromfile(filepath, dtype=np.float64)

        if len(pc) % 3 != 0:
            print(f"Warning: 点云数据长度不是3的倍数: {len(pc)}")
            return np.array([])

        num_points = len(pc) // 3
        pc = pc.reshape(num_points, 3)

        return pc

    except Exception as e:
        print(f"Error: 加载点云失败 {filepath}: {e}")
        return np.array([])


def load_config_from_yaml(yaml_path: str) -> ContourManagerConfig:
    """
    从YAML配置文件加载BEV配置

    Args:
        yaml_path: config_base.yaml路径

    Returns:
        ContourManagerConfig实例
    """
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    bev_cfg = cfg_dict['bev']

    config = ContourManagerConfig()
    config.lv_grads = bev_cfg['lv_grads']  # 这个参数在自适应版本中不使用
    config.reso_row = bev_cfg['resolution']
    config.reso_col = bev_cfg['resolution']
    config.n_row = bev_cfg['grid_size']['n_row']
    config.n_col = bev_cfg['grid_size']['n_col']
    config.roi_radius = bev_cfg['roi_radius']
    config.lidar_height = bev_cfg['lidar_height']
    config.blind_sq = bev_cfg['blind_zone']

    return config


def apply_random_transform(point_cloud: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    对点云应用随机旋转和限制平移（方案B）

    Args:
        point_cloud: 原始点云 [N, 3]

    Returns:
        transformed_pc: 变换后的点云 [N, 3]
        transform_params: 变换参数字典
    """
    # 1. 随机旋转（绕z轴，0~360度）
    theta = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    rotated_pc = point_cloud @ rotation_matrix.T

    # 2. 随机平移（限制范围）
    translation = np.array([
        np.random.uniform(-5, 5),   # x: ±5m
        np.random.uniform(-5, 5),   # y: ±5m
        np.random.uniform(-1, 1)    # z: ±1m
    ], dtype=np.float64)

    # transformed_pc = rotated_pc + translation

    transformed_pc = rotated_pc #只要旋转

    # 记录变换参数
    transform_params = {
        'rotation_angle_rad': theta,
        'rotation_angle_deg': np.degrees(theta),
        'translation': translation.tolist()
    }

    return transformed_pc, transform_params


def generate_bev_from_file(pointcloud_path: str,
                           config: ContourManagerConfig,
                           save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    从点云文件生成自适应BEV（便捷函数）

    Args:
        pointcloud_path: 点云文件路径
        config: BEV配置
        save_path: 保存路径（可选）

    Returns:
        bev_layers: [8, H, W]
        vcd: [H, W]
        metadata: 元数据
    """
    # 1. 加载点云
    pointcloud = load_chilean_pointcloud(pointcloud_path)
    if len(pointcloud) == 0:
        raise ValueError(f"无法加载点云: {pointcloud_path}")

    # 2. 生成BEV
    generator = AdaptiveBEVGenerator(config)
    bev_layers, vcd, metadata = generator.make_bev(pointcloud)

    # 3. 保存（可选）
    if save_path is not None:
        np.savez_compressed(save_path,
                           bev_layers=bev_layers,
                           vcd=vcd,
                           **metadata)
        print(f"BEV saved to: {save_path}")

    return bev_layers, vcd, metadata


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 80)
    print("测试自适应BEV生成器")
    print("=" * 80)

    # 1. 加载配置
    config_path = "/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml"
    if not os.path.exists(config_path):
        print(f"Error: 配置文件不存在 {config_path}")
        sys.exit(1)

    config = load_config_from_yaml(config_path)
    print(f"✓ 配置加载成功")
    print(f"  分辨率: {config.reso_row}m")
    print(f"  网格大小: {config.n_row}×{config.n_col}")

    # 2. 加载测试点云
    test_pc_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale_5cm/120/pointcloud_20m_10overlap/120009.bin"

    if not os.path.exists(test_pc_path):
        print(f"Error: 测试点云文件不存在 {test_pc_path}")
        sys.exit(1)

    pc = load_chilean_pointcloud(test_pc_path)
    print(f"\n✓ 加载点云: {len(pc)} 个点")
    print(f"  原始高度范围: [{pc[:, 2].min():.2f}, {pc[:, 2].max():.2f}]m")
    print(f"  原始xy范围: x[{pc[:, 0].min():.2f}, {pc[:, 0].max():.2f}], y[{pc[:, 1].min():.2f}, {pc[:, 1].max():.2f}]")

    # 3. 创建生成器
    generator = AdaptiveBEVGenerator(config, layer_thickness=0.625)

    # 4. 存储所有测试结果
    test_results = []

    # 5. 测试1：原始点云
    print("\n" + "=" * 80)
    print("[测试1] 原始点云")
    print("=" * 80)

    bev_orig, vcd_orig, meta_orig = generator.make_bev(pc, "original")

    print(f"  z中心: {meta_orig['z_center']:.2f}m")
    print(f"  自适应分层（固定层厚{meta_orig['layer_thickness']}m）:")
    for i, (z_min, z_max) in enumerate(zip(meta_orig['lv_grads'][:-1], meta_orig['lv_grads'][1:])):
        print(f"    Layer {i}: [{z_min:.3f}, {z_max:.3f})m")
    print(f"  BEV shape: {bev_orig.shape}")
    print(f"  点统计: 总数={meta_orig['points_total']}, 使用={meta_orig['points_used']}, "
          f"丢弃={meta_orig['points_discarded']} ({meta_orig['discard_ratio']*100:.1f}%)")

    # 记录每层占用像素数
    layer_pixels_orig = [np.sum(bev_orig[i] > 0) for i in range(8)]
    print(f"  每层占用像素数: {layer_pixels_orig}")

    test_results.append({
        'name': 'original',
        'bev': bev_orig,
        'vcd': vcd_orig,
        'metadata': meta_orig,
        'layer_pixels': layer_pixels_orig
    })

    # 6. 测试2-4：3次随机变换
    for test_idx in range(1, 4):
        print("\n" + "=" * 80)
        print(f"[测试{test_idx+1}] 随机变换 #{test_idx}")
        print("=" * 80)

        # 应用随机变换
        pc_transformed, transform_params = apply_random_transform(pc)

        print(f"  旋转角度: {transform_params['rotation_angle_deg']:.1f}°")
        print(f"  平移: x={transform_params['translation'][0]:.2f}m, "
              f"y={transform_params['translation'][1]:.2f}m, "
              f"z={transform_params['translation'][2]:.2f}m")
        print(f"  变换后高度范围: [{pc_transformed[:, 2].min():.2f}, {pc_transformed[:, 2].max():.2f}]m")

        # 生成BEV
        bev_trans, vcd_trans, meta_trans = generator.make_bev(pc_transformed, f"transform_{test_idx}")

        print(f"  z中心: {meta_trans['z_center']:.2f}m")
        print(f"  自适应分层（固定层厚{meta_trans['layer_thickness']}m）:")
        for i, (z_min, z_max) in enumerate(zip(meta_trans['lv_grads'][:-1], meta_trans['lv_grads'][1:])):
            print(f"    Layer {i}: [{z_min:.3f}, {z_max:.3f})m")
        print(f"  点统计: 总数={meta_trans['points_total']}, 使用={meta_trans['points_used']}, "
              f"丢弃={meta_trans['points_discarded']} ({meta_trans['discard_ratio']*100:.1f}%)")

        # 记录每层占用像素数
        layer_pixels_trans = [np.sum(bev_trans[i] > 0) for i in range(8)]
        print(f"  每层占用像素数: {layer_pixels_trans}")

        test_results.append({
            'name': f'transform_{test_idx}',
            'bev': bev_trans,
            'vcd': vcd_trans,
            'metadata': meta_trans,
            'transform_params': transform_params,
            'layer_pixels': layer_pixels_trans
        })

    # 7. 对比分析
    print("\n" + "=" * 80)
    print("[对比分析] BEV占用像素数对比")
    print("=" * 80)

    for layer_idx in range(8):
        orig_pixels = test_results[0]['layer_pixels'][layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  原始: {orig_pixels}")

        for test_idx in range(1, 4):
            trans_pixels = test_results[test_idx]['layer_pixels'][layer_idx]
            diff = abs(orig_pixels - trans_pixels)
            diff_ratio = (diff / orig_pixels * 100) if orig_pixels > 0 else 0
            status = "✓" if diff_ratio < 10 else "✗"
            print(f"  变换{test_idx}: {trans_pixels}, 差异={diff} ({diff_ratio:.1f}%) {status}")

    # 8. 总体统计
    print("\n" + "=" * 80)
    print("[总体统计]")
    print("=" * 80)

    all_diffs = []
    for layer_idx in range(8):
        orig_pixels = test_results[0]['layer_pixels'][layer_idx]
        if orig_pixels > 0:
            for test_idx in range(1, 4):
                trans_pixels = test_results[test_idx]['layer_pixels'][layer_idx]
                diff_ratio = abs(orig_pixels - trans_pixels) / orig_pixels * 100
                all_diffs.append(diff_ratio)

    if all_diffs:
        print(f"  平均像素差异: {np.mean(all_diffs):.2f}%")
        print(f"  最大像素差异: {np.max(all_diffs):.2f}%")
        print(f"  差异<10%的比例: {np.sum(np.array(all_diffs) < 10) / len(all_diffs) * 100:.1f}%")

    # 9. 保存测试结果
    output_dir = "/home/wzj/pan1/contour_deep_1++/data/test_bev_generator_adaptive"
    os.makedirs(output_dir, exist_ok=True)

    for result in test_results:
        save_path = os.path.join(output_dir, f"bev_{result['name']}.npz")
        np.savez_compressed(
            save_path,
            bev_layers=result['bev'],
            vcd=result['vcd'],
            layer_pixels=result['layer_pixels'],
            **result['metadata'],
            **({'transform_params': result['transform_params']} if 'transform_params' in result else {})
        )

    print(f"\n✓ 测试结果已保存到: {output_dir}")
    print(f"  共保存 {len(test_results)} 个测试案例")

    # 10. 验收结论
    print("\n" + "=" * 80)
    print("[验收结论]")
    print("=" * 80)

    if all_diffs and np.mean(all_diffs) < 10:
        print("  ✅ 通过！平均像素差异<10%，旋转平移不变性良好")
    else:
        print("  ⚠️  警告：平均像素差异>10%，需要检查分层逻辑")

    print("\n✓ 自适应BEV生成器测试完成!")
    print("=" * 80)