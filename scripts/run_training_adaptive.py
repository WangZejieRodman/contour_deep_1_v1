# scripts/run_training_adaptive.py
"""
自适应分层版本：完整训练启动脚本
用法: 在PyCharm中直接运行此脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path


def check_prerequisites():
    """检查训练前提条件"""
    print("=" * 60)
    print("检查训练前提条件（自适应版本）...")
    print("=" * 60)

    errors = []

    # 1. 检查配置文件
    config_files = [
        "/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml",
        "/home/wzj/pan1/contour_deep_1++/configs/config_retrieval.yaml"
    ]

    for config_file in config_files:
        if not os.path.exists(config_file):
            errors.append(f"配置文件不存在: {config_file}")
        else:
            print(f"✓ 配置文件存在: {config_file}")

    # 2. 检查数据文件
    data_files = [
        "/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/training_queries_chilean_period.pickle",
        "/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/test_queries_chilean_period.pickle",
    ]

    for data_file in data_files:
        if not os.path.exists(data_file):
            errors.append(f"数据文件不存在: {data_file}")
        else:
            print(f"✓ 数据文件存在: {data_file}")

    # 3. 检查自适应BEV缓存
    cache_dir = "/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache"
    train_cache = os.path.join(cache_dir, "train_adaptive")
    test_cache = os.path.join(cache_dir, "test_adaptive")

    if not os.path.exists(train_cache):
        errors.append(f"训练集自适应缓存不存在: {train_cache}")
        errors.append("  → 请先运行: python scripts/preprocess_bev_adaptive.py --split train")
    else:
        num_train_files = len(list(Path(train_cache).glob("*.npz")))
        print(f"✓ 训练集自适应缓存: {num_train_files} 个文件")

    if not os.path.exists(test_cache):
        errors.append(f"测试集自适应缓存不存在: {test_cache}")
        errors.append("  → 请先运行: python scripts/preprocess_bev_adaptive.py --split test")
    else:
        num_test_files = len(list(Path(test_cache).glob("*.npz")))
        print(f"✓ 测试集自适应缓存: {num_test_files} 个文件")

    # 4. 检查增强配置
    import yaml
    with open("/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    aug_config = base_config.get('augmentation', {})
    if aug_config.get('enabled', False):
        rotation_config = aug_config.get('rotation', {})
        rotation_range = rotation_config.get('range', [])

        if rotation_range == [0, 360]:
            print(f"✓ 旋转增强配置正确: {rotation_range}°")
        else:
            errors.append(f"旋转增强配置错误: {rotation_range}° (应为 [0, 360])")
            errors.append("  → 请检查 config_base.yaml 中的 augmentation.rotation.range")

    # 5. 报告错误
    if errors:
        print("\n" + "=" * 60)
        print("❌ 发现以下错误:")
        for error in errors:
            print(f"  {error}")
        print("=" * 60)
        return False

    print("\n✓ 所有前提条件满足!")
    return True


def start_training():
    """启动训练"""
    print("\n" + "=" * 60)
    print("启动自适应分层训练...")
    print("=" * 60)

    # 导入训练脚本
    from training.train_retrieval_adaptive import main  # 关键修改：导入自适应版本
    import argparse

    # 构造参数
    args = argparse.Namespace(
        config='/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml',
        train_config='/home/wzj/pan1/contour_deep_1++/configs/config_retrieval.yaml',
        train_pickle='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/training_queries_chilean_period.pickle',
        test_pickle='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/test_queries_chilean_period.pickle',
        experiment_name='retrieval_adaptive_v1',  # 修改实验名称
        resume=None
    )

    # 记录开始时间
    start_time = time.time()

    # 执行训练
    try:
        main(args)

        # 记录结束时间
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        print("\n" + "=" * 60)
        print(f"✓ 训练完成!")
        print(f"总耗时: {hours}小时 {minutes}分钟")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n训练被用户中断")
        print("Checkpoint已保存，可以使用 --resume 参数恢复训练")

    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("自适应分层版本：完整训练")
    print("=" * 60)
    print("特性:")
    print("  - 固定层厚0.625m，中心对齐分层")
    print("  - 训练时动态增强：旋转0~360°")
    print("  - 真正的旋转平移不变性")
    print("=" * 60)
    print("预计耗时: 约4小时")
    print("=" * 60)


    # 1. 检查前提条件
    if not check_prerequisites():
        print("\n请先解决上述问题，然后重新运行")
        return

    # 2. 启动训练
    start_training()

    # 3. 后续提示
    print("\n下一步:")
    print("  1. 查看训练日志: logs/retrieval_adaptive_v1/")
    print("  2. 启动TensorBoard监控: tensorboard --logdir logs")
    print("  3. 训练完成后进行评估")


if __name__ == "__main__":
    main()