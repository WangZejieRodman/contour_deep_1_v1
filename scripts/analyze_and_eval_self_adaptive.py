"""
scripts/analyze_and_eval_self_adaptive.py

自适应版本：训练集自查询验证
功能：
  - 选项A：不加旋转（快速验证模型基本匹配能力）
  - 选项B：加旋转（固定角度验证旋转不变性）

用法: 在PyCharm中直接运行
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
import json
import cv2

# ============================================================
# 配置区域（在这里修改参数）
# ============================================================

# 选项A或选项B
TEST_MODE = "B"  # "A" = 不加旋转, "B" = 固定角度旋转

# 选项B的旋转角度列表（度）
ROTATION_ANGLES = [0, 5, 15, 30, 45, 60, 90, 135, 180]

# 是否保存详细结果
SAVE_DETAILED_RESULTS = True

# 输出目录
OUTPUT_DIR = "logs/eval_self_adaptive"


# ============================================================


def apply_fixed_rotation_to_bev(bev_stack: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    对BEV应用固定角度旋转

    Args:
        bev_stack: [C, H, W] 所有层的BEV
        angle_deg: 旋转角度（度）

    Returns:
        rotated: [C, H, W] 旋转后的BEV
    """
    C, H, W = bev_stack.shape
    center = (W // 2, H // 2)

    # 构建旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 对每个通道旋转
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


def evaluate_recall_without_rotation():
    """选项A：不加旋转的训练集自查询"""
    print("=" * 60)
    print("选项A：训练集自查询（不加旋转）")
    print("=" * 60)
    print("目的: 快速验证模型学习了基本匹配能力")
    print("=" * 60)

    from models.retrieval_net import RetrievalNet
    from data.dataset_retrieval_adaptive import AdaptiveRetrievalDataset
    import yaml
    from sklearn.neighbors import NearestNeighbors

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    model = RetrievalNet(output_dim=128)
    checkpoint = torch.load('checkpoints/retrieval_adaptive_v1/latest.pth',
                            map_location='cuda',
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    print(f"  ✓ 加载Epoch {checkpoint['epoch'] + 1}的模型")
    print(f"  ✓ Val Loss: {checkpoint['metric']:.4f}")

    # 2. 加载配置
    with open('/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 3. 创建训练集（不增强）
    print("\n[2/5] 加载训练集...")
    train_dataset = AdaptiveRetrievalDataset(
        queries_pickle='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/training_queries_chilean_period.pickle',
        cache_root='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache',
        split='train',
        num_negatives=10,
        augmentation_config=None,  # 不增强
        resolution=config['bev']['resolution'],
        use_cache=True
    )
    print(f"  ✓ 训练集大小: {len(train_dataset.query_keys)}")

    # 4. 提取所有特征
    print("\n[3/5] 提取训练集特征...")
    all_features = []
    all_keys = []
    all_ground_truths = []

    with torch.no_grad():
        for query_key in tqdm(train_dataset.query_keys, desc="提取特征"):
            # 加载BEV
            bev_data = train_dataset._load_bev_from_cache(query_key)
            if bev_data is None:
                continue

            # 预处理（不增强）
            bev_tensor = train_dataset._preprocess_bev(*bev_data, apply_aug=False)
            bev_tensor = bev_tensor.unsqueeze(0).cuda()

            # 提取特征
            feat = model(bev_tensor)
            all_features.append(feat.cpu().numpy())
            all_keys.append(query_key)

            # 获取ground truth
            query_data = train_dataset.queries[query_key]
            all_ground_truths.append(set(query_data['positives']))

    all_features = np.vstack(all_features)
    print(f"  ✓ 提取了 {len(all_features)} 个特征")

    # 5. 构建KNN索引
    print("\n[4/5] 构建KNN索引...")
    knn = NearestNeighbors(n_neighbors=26, metric='euclidean', n_jobs=-1)
    knn.fit(all_features)
    print(f"  ✓ 索引构建完成")

    # 6. 计算Recall（排除自身）
    print("\n[5/5] 计算Recall@N（排除自身）...")
    recalls = {1: 0, 5: 0, 10: 0, 25: 0}
    valid_queries = 0
    total_positives = []
    rank_of_first_positive = []

    for i in tqdm(range(len(all_features)), desc="KNN检索"):
        if len(all_ground_truths[i]) == 0:
            continue

        valid_queries += 1
        total_positives.append(len(all_ground_truths[i]))

        # KNN搜索
        distances, indices = knn.kneighbors([all_features[i]])
        retrieved_indices = indices[0][1:]  # 排除自身
        retrieved_keys = [all_keys[idx] for idx in retrieved_indices]

        # 找到第一个正样本的排名
        first_positive_rank = None
        for rank, key in enumerate(retrieved_keys, start=1):
            if key in all_ground_truths[i]:
                first_positive_rank = rank
                break

        if first_positive_rank:
            rank_of_first_positive.append(first_positive_rank)

        # 检查Recall@K
        for k in [1, 5, 10, 25]:
            if any(key in all_ground_truths[i] for key in retrieved_keys[:k]):
                recalls[k] += 1

    # 归一化
    for k in recalls:
        recalls[k] = recalls[k] / valid_queries * 100

    # 7. 输出结果
    print("\n" + "=" * 60)
    print("训练集自查询结果（不加旋转）:")
    print("=" * 60)
    print(f"有效查询数: {valid_queries}")
    print(f"平均正样本数: {np.mean(total_positives):.1f}")
    print(f"\nRecall性能:")
    print(f"  Recall@1:  {recalls[1]:.2f}%")
    print(f"  Recall@5:  {recalls[5]:.2f}%")
    print(f"  Recall@10: {recalls[10]:.2f}%")
    print(f"  Recall@25: {recalls[25]:.2f}%")

    if rank_of_first_positive:
        avg_rank = np.mean(rank_of_first_positive)
        median_rank = np.median(rank_of_first_positive)
        print(f"\n第一个正样本排名统计:")
        print(f"  平均排名: {avg_rank:.1f}")
        print(f"  中位数排名: {median_rank:.1f}")

    return recalls, {
        'valid_queries': valid_queries,
        'avg_positives': float(np.mean(total_positives)),
        'avg_rank': float(avg_rank) if rank_of_first_positive else None,
        'median_rank': float(median_rank) if rank_of_first_positive else None
    }


def evaluate_recall_with_rotation():
    """选项B：固定角度旋转的训练集自查询"""
    print("=" * 60)
    print("选项B：训练集自查询（固定角度旋转）")
    print("=" * 60)
    print("目的: 验证旋转不变性")
    print(f"测试角度: {ROTATION_ANGLES}°")
    print("=" * 60)

    from models.retrieval_net import RetrievalNet
    from data.dataset_retrieval_adaptive import AdaptiveRetrievalDataset
    import yaml
    from sklearn.neighbors import NearestNeighbors
    from data.data_utils import normalize_bev, stack_bev_with_vcd

    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    model = RetrievalNet(output_dim=128)
    checkpoint = torch.load('checkpoints/retrieval_adaptive_v1/latest.pth',
                            map_location='cuda',
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    print(f"  ✓ 加载Epoch {checkpoint['epoch'] + 1}的模型")

    # 2. 加载配置
    with open('/home/wzj/pan1/contour_deep_1++/configs/config_base.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 3. 创建训练集
    print("\n[2/6] 加载训练集...")
    train_dataset = AdaptiveRetrievalDataset(
        queries_pickle='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache/training_queries_chilean_period.pickle',
        cache_root='/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache',
        split='train',
        num_negatives=10,
        augmentation_config=None,
        resolution=config['bev']['resolution'],
        use_cache=True
    )
    print(f"  ✓ 训练集大小: {len(train_dataset.query_keys)}")

    # 4. 提取数据库特征（不旋转）
    print("\n[3/6] 提取数据库特征（不旋转）...")
    db_features = []
    db_keys = []
    ground_truths = []

    with torch.no_grad():
        for query_key in tqdm(train_dataset.query_keys, desc="提取数据库特征"):
            bev_data = train_dataset._load_bev_from_cache(query_key)
            if bev_data is None:
                continue

            bev_tensor = train_dataset._preprocess_bev(*bev_data, apply_aug=False)
            bev_tensor = bev_tensor.unsqueeze(0).cuda()

            feat = model(bev_tensor)
            db_features.append(feat.cpu().numpy())
            db_keys.append(query_key)

            query_data = train_dataset.queries[query_key]
            ground_truths.append(set(query_data['positives']))

    db_features = np.vstack(db_features)
    print(f"  ✓ 提取了 {len(db_features)} 个数据库特征")

    # 5. 构建KNN索引
    print("\n[4/6] 构建KNN索引...")
    knn = NearestNeighbors(n_neighbors=26, metric='euclidean', n_jobs=-1)
    knn.fit(db_features)

    # 6. 对每个角度进行评估
    print("\n[5/6] 对每个旋转角度进行评估...")

    results_per_angle = {}

    for angle in ROTATION_ANGLES:
        print(f"\n  测试角度: {angle}°")

        # 提取旋转后的查询特征
        query_features = []
        query_keys = []

        with torch.no_grad():
            for query_key in tqdm(train_dataset.query_keys, desc=f"  {angle}° 提取查询特征", leave=False):
                bev_data = train_dataset._load_bev_from_cache(query_key)
                if bev_data is None:
                    continue

                # 预处理（不增强）
                bev_layers, vcd = bev_data
                bev_norm, vcd_norm = normalize_bev(bev_layers, vcd)
                stacked = stack_bev_with_vcd(bev_norm, vcd_norm)

                # 应用固定角度旋转
                rotated = apply_fixed_rotation_to_bev(stacked, angle)

                # 转tensor
                bev_tensor = torch.from_numpy(rotated).float().unsqueeze(0).cuda()

                # 提取特征
                feat = model(bev_tensor)
                query_features.append(feat.cpu().numpy())
                query_keys.append(query_key)

        query_features = np.vstack(query_features)

        # 计算Recall
        recalls = {1: 0, 5: 0, 10: 0, 25: 0}
        valid_queries = 0
        rank_of_first_positive = []

        for i in range(len(query_features)):
            if len(ground_truths[i]) == 0:
                continue

            valid_queries += 1

            # KNN搜索
            distances, indices = knn.kneighbors([query_features[i]])
            retrieved_indices = indices[0][1:]  # 排除自身
            retrieved_keys = [db_keys[idx] for idx in retrieved_indices]

            # 找第一个正样本
            first_positive_rank = None
            for rank, key in enumerate(retrieved_keys, start=1):
                if key in ground_truths[i]:
                    first_positive_rank = rank
                    break

            if first_positive_rank:
                rank_of_first_positive.append(first_positive_rank)

            # Recall@K
            for k in [1, 5, 10, 25]:
                if any(key in ground_truths[i] for key in retrieved_keys[:k]):
                    recalls[k] += 1

        # 归一化
        for k in recalls:
            recalls[k] = recalls[k] / valid_queries * 100

        # 保存结果
        results_per_angle[angle] = {
            'recalls': recalls,
            'avg_rank': float(np.mean(rank_of_first_positive)) if rank_of_first_positive else None,
            'median_rank': float(np.median(rank_of_first_positive)) if rank_of_first_positive else None
        }

        print(f"    Recall@1: {recalls[1]:.2f}%")

    # 7. 汇总结果
    print("\n[6/6] 汇总结果...")
    print("\n" + "=" * 60)
    print("旋转不变性测试结果:")
    print("=" * 60)

    # 打印每个角度的Recall@1
    print("\n各角度Recall@1:")
    for angle in ROTATION_ANGLES:
        recall1 = results_per_angle[angle]['recalls'][1]
        print(f"  {angle:3d}°: {recall1:6.2f}%")

    # 统计
    all_recall1 = [results_per_angle[a]['recalls'][1] for a in ROTATION_ANGLES]
    mean_recall1 = np.mean(all_recall1)
    std_recall1 = np.std(all_recall1)
    min_recall1 = np.min(all_recall1)
    max_recall1 = np.max(all_recall1)

    print(f"\nRecall@1 统计:")
    print(f"  平均值: {mean_recall1:.2f}%")
    print(f"  标准差: {std_recall1:.2f}%")
    print(f"  范围: [{min_recall1:.2f}%, {max_recall1:.2f}%]")

    # 全部Recall指标（使用0°的结果）
    baseline_recalls = results_per_angle[0]['recalls']
    print(f"\n基准性能（0°旋转）:")
    print(f"  Recall@1:  {baseline_recalls[1]:.2f}%")
    print(f"  Recall@5:  {baseline_recalls[5]:.2f}%")
    print(f"  Recall@10: {baseline_recalls[10]:.2f}%")
    print(f"  Recall@25: {baseline_recalls[25]:.2f}%")

    return results_per_angle, {
        'mean_recall1': mean_recall1,
        'std_recall1': std_recall1,
        'min_recall1': min_recall1,
        'max_recall1': max_recall1
    }


def print_diagnosis(recalls, stats):
    """打印诊断结论"""
    print("\n" + "=" * 60)
    print("诊断结论:")
    print("=" * 60)

    recall1 = recalls[1]

    if recall1 > 80:
        print(f"  ✅ 模型学习正常 (Recall@1 = {recall1:.2f}%)")
        print(f"  ✅ 特征空间良好，正样本被拉近")
    elif recall1 > 50:
        print(f"  ⚠️  模型学习一般 (Recall@1 = {recall1:.2f}%)")
        print(f"  ⚠️  特征有一定区分能力，但还不够强")
    elif recall1 > 20:
        print(f"  ❌ 模型学习较差 (Recall@1 = {recall1:.2f}%)")
        print(f"  ❌ 特征空间混乱，正负样本未分离")
    else:
        print(f"  ❌ 模型几乎没学到东西 (Recall@1 = {recall1:.2f}%)")
        print(f"  ❌ 特征可能是随机的")


def print_rotation_invariance_diagnosis(summary):
    """打印旋转不变性诊断"""
    print("\n" + "=" * 60)
    print("旋转不变性诊断:")
    print("=" * 60)

    mean = summary['mean_recall1']
    std = summary['std_recall1']

    if std < 5:
        print(f"  ✅ 旋转不变性优秀 (标准差 {std:.2f}%)")
        print(f"  ✅ 所有角度性能稳定")
    elif std < 10:
        print(f"  ⚠️  旋转不变性良好 (标准差 {std:.2f}%)")
        print(f"  ⚠️  存在轻微的角度敏感性")
    else:
        print(f"  ❌ 旋转不变性较差 (标准差 {std:.2f}%)")
        print(f"  ❌ 不同角度性能差异较大")
        print(f"  → 建议增加训练epochs或调整增强参数")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("自适应版本：训练集自查询验证")
    print("=" * 60)

    # 检查checkpoint
    checkpoint_path = "checkpoints/retrieval_adaptive_v1/latest.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return

    # 根据TEST_MODE选择评估方式
    if TEST_MODE == "A":
        # 选项A：不加旋转
        recalls, stats = evaluate_recall_without_rotation()
        print_diagnosis(recalls, stats)

        # 保存结果
        if SAVE_DETAILED_RESULTS:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            results = {
                'test_mode': 'A_no_rotation',
                'recalls': recalls,
                'stats': stats,
                'checkpoint': checkpoint_path
            }
            output_file = os.path.join(OUTPUT_DIR, 'results_no_rotation.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ 结果已保存: {output_file}")

    elif TEST_MODE == "B":
        # 选项B：固定角度旋转
        results_per_angle, summary = evaluate_recall_with_rotation()
        print_rotation_invariance_diagnosis(summary)

        # 保存结果
        if SAVE_DETAILED_RESULTS:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            results = {
                'test_mode': 'B_fixed_rotation',
                'rotation_angles': ROTATION_ANGLES,
                'results_per_angle': results_per_angle,
                'summary': summary,
                'checkpoint': checkpoint_path
            }
            output_file = os.path.join(OUTPUT_DIR, 'results_with_rotation.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ 结果已保存: {output_file}")

    else:
        print(f"❌ 无效的TEST_MODE: {TEST_MODE}")
        print("   请设置为 'A' 或 'B'")


if __name__ == "__main__":
    main()