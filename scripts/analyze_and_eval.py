"""
Day 9: 基于验证集的跨时间段评估（使用BEV缓存）
参考 analyze_and_eval_day9.py 的结构
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
import pickle
import yaml
from sklearn.neighbors import NearestNeighbors
import json
from pathlib import Path


def load_pickle(filepath):
    """加载pickle文件"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Loaded: {filepath}")
    return data


def extract_features_from_cache(model, data_sets, cache_root, cache_type, device):
    """
    从BEV缓存提取特征

    Args:
        model: 训练好的模型
        data_sets: List[Dict], 每个session的数据字典
        cache_root: BEV缓存根目录
        cache_type: 'evaluation_database' 或 'evaluation_query'
        device: 设备

    Returns:
        features: [N, 128] 所有特征
        metadata: List[Dict] 每个特征的元数据
    """
    from data.data_utils import normalize_bev, stack_bev_with_vcd

    print(f"\n[提取{cache_type}特征]")
    print("=" * 60)

    features = []
    metadata = []
    global_idx = 0

    model.eval()
    with torch.no_grad():
        for session_idx, data_dict in enumerate(tqdm(data_sets, desc=f"{cache_type} Sessions")):
            if len(data_dict) == 0:
                continue

            session_cache_dir = os.path.join(cache_root, cache_type, f'session_{session_idx}')

            if not os.path.exists(session_cache_dir):
                print(f"  ⚠️  Session {session_idx} 缓存不存在: {session_cache_dir}")
                continue

            for local_idx in tqdm(sorted(data_dict.keys()),
                                 desc=f"  Session {session_idx}",
                                 leave=False):
                entry = data_dict[local_idx]

                # 缓存文件路径
                cache_filename = f"{local_idx:06d}.npz"
                cache_path = os.path.join(session_cache_dir, cache_filename)

                if not os.path.exists(cache_path):
                    print(f"  ⚠️  缓存不存在: {cache_path}")
                    continue

                try:
                    # 从缓存加载BEV
                    data = np.load(cache_path)
                    bev_layers = data['bev_layers']
                    vcd = data['vcd']

                    # 预处理
                    bev_norm, vcd_norm = normalize_bev(bev_layers, vcd)
                    stacked = stack_bev_with_vcd(bev_norm, vcd_norm)
                    bev_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device)

                    # 提取特征
                    feat = model(bev_tensor)
                    features.append(feat.cpu().numpy()[0])

                    # 记录元数据
                    meta = {
                        'session_idx': session_idx,
                        'local_idx': local_idx,
                        'global_idx': global_idx,
                        'file': entry['query']
                    }

                    # 如果是查询集，添加正样本信息
                    if 'positives' in entry:
                        meta['positives'] = entry['positives']

                    metadata.append(meta)
                    global_idx += 1

                except Exception as e:
                    print(f"  ❌ 处理失败 {cache_path}: {e}")
                    continue

    features = np.vstack(features)
    print(f"✓ 提取了 {len(features)} 个特征")

    return features, metadata


def map_positives_to_global(query_metadata, db_metadata):
    """
    将查询的正样本从 {session_idx: [local_indices]} 映射到全局索引

    Args:
        query_metadata: 查询元数据
        db_metadata: 数据库元数据

    Returns:
        global_positives: List[Set[int]], 每个查询的全局正样本集合
    """
    print("\n[映射正样本到全局索引]")
    print("=" * 60)

    # 构建 (session_idx, local_idx) -> global_idx 的映射
    db_local_to_global = {}
    for db_meta in db_metadata:
        key = (db_meta['session_idx'], db_meta['local_idx'])
        db_local_to_global[key] = db_meta['global_idx']

    global_positives = []
    total_positives = 0
    queries_with_positives = 0

    for query_meta in query_metadata:
        positives_set = set()

        # 遍历该查询在各个数据库session中的正样本
        if 'positives' in query_meta:
            for db_session_idx, local_indices in query_meta['positives'].items():
                for local_idx in local_indices:
                    key = (db_session_idx, local_idx)
                    if key in db_local_to_global:
                        global_idx = db_local_to_global[key]
                        positives_set.add(global_idx)

        global_positives.append(positives_set)
        total_positives += len(positives_set)

        if len(positives_set) > 0:
            queries_with_positives += 1

    print(f"✓ 总查询数: {len(global_positives)}")
    print(f"✓ 有正样本的查询数: {queries_with_positives}")
    print(f"✓ 无正样本的查询数: {len(global_positives) - queries_with_positives}")
    print(f"✓ 总正样本对数: {total_positives}")

    if queries_with_positives > 0:
        print(f"✓ 平均每查询正样本数: {total_positives / queries_with_positives:.1f}")

    return global_positives


def compute_recall(query_features, db_features, global_positives, k_values=[1, 5, 10, 25]):
    """
    计算Recall@K

    Args:
        query_features: [N_query, D]
        db_features: [N_db, D]
        global_positives: List[Set[int]], 每个查询的全局正样本集合
        k_values: 要计算的K值列表

    Returns:
        recalls: Dict[int, float]
        rank_stats: Dict
    """
    print("\n[计算Recall@K]")
    print("=" * 60)

    # 构建KNN索引
    max_k = max(k_values)
    if len(db_features) < max_k:
        max_k = len(db_features)
        print(f"  ⚠️  数据库大小 < {max(k_values)}，调整K为{max_k}")

    knn = NearestNeighbors(n_neighbors=max_k, metric='euclidean', n_jobs=-1)
    knn.fit(db_features)
    print(f"✓ KNN索引构建完成 (K={max_k})")

    recalls = {k: 0 for k in k_values}
    rank_of_first_positive = []
    valid_queries = 0

    print("检索中...")
    for i in tqdm(range(len(query_features))):
        if len(global_positives[i]) == 0:
            continue  # 跳过没有正样本的查询

        valid_queries += 1

        # KNN搜索
        distances, indices = knn.kneighbors([query_features[i]])
        retrieved_indices = indices[0]

        # 找到第一个正样本的排名
        first_positive_rank = None
        for rank, idx in enumerate(retrieved_indices, start=1):
            if idx in global_positives[i]:
                first_positive_rank = rank
                break

        if first_positive_rank:
            rank_of_first_positive.append(first_positive_rank)

        # 检查Recall@K
        for k in k_values:
            if k <= max_k:
                if any(idx in global_positives[i] for idx in retrieved_indices[:k]):
                    recalls[k] += 1

    # 归一化
    if valid_queries > 0:
        for k in recalls:
            recalls[k] = recalls[k] / valid_queries * 100
    else:
        print("  ⚠️  没有有效查询!")

    # 统计
    rank_stats = {}
    if rank_of_first_positive:
        rank_stats = {
            'mean': float(np.mean(rank_of_first_positive)),
            'median': float(np.median(rank_of_first_positive)),
            'min': int(np.min(rank_of_first_positive)),
            'max': int(np.max(rank_of_first_positive))
        }

    print(f"✓ 有效查询数: {valid_queries}")

    return recalls, rank_stats


def main():
    """主评估流程"""
    print("=" * 60)
    print("Day 9: 基于验证集的跨时间段评估")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/6] 加载模型...")
    from models.retrieval_net import RetrievalNet

    model = RetrievalNet(output_dim=128)
    checkpoint_path = 'checkpoints/retrieval_baseline_day8/latest.pth'

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()

    print(f"  ✓ 加载Epoch {checkpoint['epoch']+1}的模型")
    print(f"  ✓ Val Loss: {checkpoint['metric']:.4f}")

    # 2. 加载数据库和查询集pickle
    print("\n[2/6] 加载数据集...")
    database_pickle = '/home/wzj/pan1/contour_deep_1++/data/chilean_evaluation_database_190_194.pickle'
    query_pickle = '/home/wzj/pan1/contour_deep_1++/data/chilean_evaluation_query_195_199.pickle'

    database_sets = load_pickle(database_pickle)
    query_sets = load_pickle(query_pickle)

    print(f"  ✓ 数据库sessions: {len(database_sets)}")
    print(f"  ✓ 查询sessions: {len(query_sets)}")

    # 3. 提取数据库特征
    print("\n[3/6] 提取数据库特征...")
    cache_root = '/home/wzj/pan1/contour_deep_1++/data/Chilean_BEV_Cache'

    db_features, db_metadata = extract_features_from_cache(
        model, database_sets, cache_root, 'evaluation_database', 'cuda'
    )

    # 4. 提取查询特征
    print("\n[4/6] 提取查询特征...")
    query_features, query_metadata = extract_features_from_cache(
        model, query_sets, cache_root, 'evaluation_query', 'cuda'
    )

    # 5. 映射正样本
    print("\n[5/6] 映射正样本...")
    global_positives = map_positives_to_global(query_metadata, db_metadata)

    # 6. 计算Recall
    print("\n[6/6] 计算Recall...")
    recalls, rank_stats = compute_recall(query_features, db_features, global_positives)

    # 输出结果
    print("\n" + "=" * 60)
    print("评估结果（跨时间段）")
    print("=" * 60)
    print(f"数据库: Session 190-194 ({len(db_features)} entries)")
    print(f"查询:   Session 195-199 ({len(query_features)} entries)")
    print(f"\nRecall性能:")
    print(f"  Recall@1:  {recalls[1]:.2f}%")
    print(f"  Recall@5:  {recalls[5]:.2f}%")
    print(f"  Recall@10: {recalls[10]:.2f}%")
    print(f"  Recall@25: {recalls[25]:.2f}%")

    if rank_stats:
        print(f"\n第一个正样本排名统计:")
        print(f"  平均排名: {rank_stats['mean']:.1f}")
        print(f"  中位数排名: {rank_stats['median']:.1f}")
        print(f"  最小排名: {rank_stats['min']}")
        print(f"  最大排名: {rank_stats['max']}")

    # 诊断结论
    print("\n" + "=" * 60)
    print("诊断结论:")
    print("=" * 60)

    if recalls[1] >= 70:
        print(f"  🌟 跨时间段泛化能力优秀 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  ✅ 接近或超过传统方法")
    elif recalls[1] >= 60:
        print(f"  ✅ 跨时间段泛化能力良好 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  → 可以进入方向2的开发")
    elif recalls[1] >= 50:
        print(f"  ⚠️  跨时间段泛化能力一般 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  → 建议: 调整超参数或增加训练数据")
    else:
        print(f"  ❌ 跨时间段泛化能力较差 (Recall@1 = {recalls[1]:.2f}%)")
        print(f"  → 建议: 检查模型架构或训练策略")

    # 保存结果
    results = {
        'evaluation_type': 'cross_temporal',
        'database_sessions': '190-194',
        'query_sessions': '195-199',
        'num_database': len(db_features),
        'num_queries': len(query_features),
        'recalls': recalls,
        'rank_stats': rank_stats,
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': int(checkpoint['epoch']) + 1,
        'checkpoint_val_loss': float(checkpoint['metric'])
    }

    os.makedirs('logs/day9_evaluation_cross_temporal', exist_ok=True)
    result_file = 'logs/day9_evaluation_cross_temporal/evaluation_results-4.json'

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 结果已保存: {result_file}")

    # 对比分析（如果有训练集自查询结果）
    train_self_result_file = 'logs/day9_evaluation_self/train_self_query-4.json'
    if os.path.exists(train_self_result_file):
        with open(train_self_result_file, 'r') as f:
            train_self_results = json.load(f)

        # 修复：处理JSON中key可能是字符串的情况
        recalls_dict = train_self_results['recalls']
        if isinstance(list(recalls_dict.keys())[0], str):
            # key是字符串，转换为整数
            train_self_recall1 = float(recalls_dict['1'])
        else:
            # key已经是整数
            train_self_recall1 = recalls_dict[1]

        print("\n" + "=" * 60)
        print("对比分析")
        print("=" * 60)
        print(f"训练集自查询 Recall@1: {train_self_recall1:.2f}%")
        print(f"跨时间段评估 Recall@1: {recalls[1]:.2f}%")
        print(f"性能差距: {train_self_recall1 - recalls[1]:.2f}%")

        if abs(train_self_recall1 - recalls[1]) < 10:
            print("\n✅ 泛化能力优秀，训练集和验证集性能接近!")
        elif abs(train_self_recall1 - recalls[1]) < 20:
            print("\n⚠️  存在一定的泛化gap，可接受")
        else:
            print("\n❌ 泛化能力差，存在明显过拟合")


if __name__ == "__main__":
    main()