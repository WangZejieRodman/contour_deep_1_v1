"""
生成用于评估的查询集和数据库集
方案C：跨时间段拆分（最真实）

数据库: Session 190-194 (历史地图)
查询:   Session 195-199 (当前观测)
"""

import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
from pathlib import Path

# 基于时间/session的划分策略
DATABASE_SESSION_START = 190  # 数据库使用的session范围（历史数据）
DATABASE_SESSION_END = 194
QUERY_SESSION_START = 195  # 查询使用的session范围（当前数据）
QUERY_SESSION_END = 199


def output_to_file(output, filename):
    """保存pickle文件"""
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved: {filename}")


def construct_query_and_database_sets_separate(base_path, runs_folder,
                                               database_folders, query_folders,
                                               pointcloud_fols, filename, output_name):
    """
    分别构建数据库集和查询集

    Args:
        database_folders: 数据库session列表 (190-194)
        query_folders: 查询session列表 (195-199)

    Returns:
        database_sets: List[Dict], 每个数据库session的点云字典
        query_sets: List[Dict], 每个查询session的点云字典
    """

    print(f"\n{'='*60}")
    print(f"构建数据库集和查询集")
    print(f"{'='*60}")
    print(f"Pointcloud folder: {pointcloud_fols}")
    print(f"Database sessions: {database_folders}")
    print(f"Query sessions: {query_folders}")

    # ==================== 第一步：构建数据库集 ====================
    print(f"\n[1/4] 构建数据库集...")
    database_sets = []
    database_coordinates_list = []  # 存储每个数据库session的有效坐标

    for folder in database_folders:
        database = {}
        valid_coordinates = []

        csv_path = os.path.join(base_path, runs_folder, folder, filename)
        if not os.path.exists(csv_path):
            print(f"  ⚠️  Skipping database {folder}: CSV file not found")
            database_sets.append(database)
            database_coordinates_list.append(np.array([]).reshape(0, 2))
            continue

        folder_path = os.path.join(base_path, runs_folder, folder, pointcloud_fols.strip('/'))
        if not os.path.exists(folder_path):
            print(f"  ⚠️  Skipping database {folder}: Pointcloud folder not found")
            database_sets.append(database)
            database_coordinates_list.append(np.array([]).reshape(0, 2))
            continue

        df_locations = pd.read_csv(csv_path, sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(
            str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            full_path = os.path.join(base_path, row['file'])
            if not os.path.exists(full_path):
                continue

            database[len(database.keys())] = {
                'query': row['file'],
                'northing': row['northing'],
                'easting': row['easting']
            }

            # 同时记录有效坐标，确保索引一致
            valid_coordinates.append([row['northing'], row['easting']])

        database_sets.append(database)

        # 转换为numpy数组
        if valid_coordinates:
            database_coordinates_list.append(np.array(valid_coordinates))
        else:
            database_coordinates_list.append(np.array([]).reshape(0, 2))

        print(f"  ✓ Database {folder}: {len(database)} entries")

    # ==================== 第二步：构建KDTree ====================
    print(f"\n[2/4] 构建KDTree索引...")
    database_trees = []
    for i, coords in enumerate(database_coordinates_list):
        if len(coords) > 0:
            database_tree = KDTree(coords)
            database_trees.append(database_tree)
            print(f"  ✓ Database {database_folders[i]}: KDTree with {len(coords)} points")
        else:
            database_trees.append(None)
            print(f"  ⚠️  Database {database_folders[i]}: Empty")

    # ==================== 第三步：构建查询集 ====================
    print(f"\n[3/4] 构建查询集...")
    query_sets = []
    for folder in query_folders:
        queries = {}

        csv_path = os.path.join(base_path, runs_folder, folder, filename)
        if not os.path.exists(csv_path):
            print(f"  ⚠️  Skipping query {folder}: CSV file not found")
            query_sets.append(queries)
            continue

        folder_path = os.path.join(base_path, runs_folder, folder, pointcloud_fols.strip('/'))
        if not os.path.exists(folder_path):
            print(f"  ⚠️  Skipping query {folder}: Pointcloud folder not found")
            query_sets.append(queries)
            continue

        df_locations = pd.read_csv(csv_path, sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(
            str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            full_path = os.path.join(base_path, row['file'])
            if not os.path.exists(full_path):
                continue

            queries[len(queries.keys())] = {
                'query': row['file'],
                'northing': row['northing'],
                'easting': row['easting']
            }

        query_sets.append(queries)
        print(f"  ✓ Query {folder}: {len(queries)} entries")

    # ==================== 第四步：计算正样本匹配 ====================
    print(f"\n[4/4] 计算正样本匹配...")

    # 为每个查询添加positives字段（存储在所有数据库中的正样本）
    for j, query_set in enumerate(query_sets):
        for key in query_set.keys():
            # 初始化positives为字典，key是数据库session索引
            query_set[key]['positives'] = {}

    total_positive_pairs = 0
    positive_distribution = {}  # 统计每个数据库session的正样本分布

    for i, (database_tree, database_set) in enumerate(zip(database_trees, database_sets)):
        if database_tree is None or len(database_set) == 0:
            print(f"  ⚠️  Database session {database_folders[i]}: Empty, skipped")
            continue

        session_positive_count = 0

        for j, query_set in enumerate(query_sets):
            for key in query_set.keys():
                query_coord = np.array([[query_set[key]["northing"], query_set[key]["easting"]]])

                # 在数据库session i中找到距离15米内的正样本
                positive_indices = database_tree.query_radius(query_coord, r=15)[0].tolist()

                # 存储到查询集中（使用数据库session索引作为key）
                query_set[key]['positives'][i] = positive_indices

                session_positive_count += len(positive_indices)
                total_positive_pairs += len(positive_indices)

        positive_distribution[database_folders[i]] = session_positive_count
        print(f"  ✓ Database {database_folders[i]}: {session_positive_count} positive matches")

    print(f"\n  总正样本对数: {total_positive_pairs}")

    # ==================== 输出文件 ====================
    database_filename = f'{output_name}_evaluation_database_{DATABASE_SESSION_START}_{DATABASE_SESSION_END}.pickle'
    query_filename = f'{output_name}_evaluation_query_{QUERY_SESSION_START}_{QUERY_SESSION_END}.pickle'

    output_to_file(database_sets, database_filename)
    output_to_file(query_sets, query_filename)

    # ==================== 验证索引一致性 ====================
    print(f"\n{'='*60}")
    print(f"验证索引一致性")
    print(f"{'='*60}")

    validation_passed = True
    queries_with_positives = 0
    queries_without_positives = 0

    for j, query_set in enumerate(query_sets):
        for key in query_set.keys():
            has_positive = False

            for i, database_set in enumerate(database_sets):
                if i not in query_set[key]['positives']:
                    continue

                positive_indices = query_set[key]['positives'][i]

                if len(positive_indices) > 0:
                    has_positive = True

                # 验证索引有效性
                for pos_idx in positive_indices:
                    if pos_idx not in database_set:
                        print(f"❌ 错误：查询 {key} 在数据库session {i} 的正样本索引 {pos_idx} 无效")
                        validation_passed = False

            if has_positive:
                queries_with_positives += 1
            else:
                queries_without_positives += 1

    if validation_passed:
        print(f"✅ 所有索引验证通过!")
    else:
        print(f"❌ 存在无效索引!")

    print(f"\n有正样本的查询数: {queries_with_positives}")
    print(f"无正样本的查询数: {queries_without_positives}")

    # ==================== 统计摘要 ====================
    print(f"\n{'='*60}")
    print(f"最终统计")
    print(f"{'='*60}")

    total_db_entries = sum(len(db_set) for db_set in database_sets)
    total_query_entries = sum(len(query_set) for query_set in query_sets)

    print(f"数据库集: {len(database_sets)} sessions, {total_db_entries} entries")
    print(f"查询集: {len(query_sets)} sessions, {total_query_entries} entries")
    print(f"总正样本对数: {total_positive_pairs}")
    print(f"平均每查询正样本数: {total_positive_pairs / max(total_query_entries, 1):.1f}")

    print(f"\n正样本分布（按数据库session）:")
    for db_session, count in positive_distribution.items():
        print(f"  Session {db_session}: {count} matches")

    print(f"\n{'='*60}")
    print(f"应用场景")
    print(f"{'='*60}")
    print(f"数据库 (190-194): 历史点云作为参考地图")
    print(f"查询 (195-199): 当前观测用于定位")
    print(f"任务: 在历史地图中为当前观测找到匹配位置")

    return database_sets, query_sets


# ==================== 主执行部分 ====================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"生成评估数据集：跨时间段拆分")
    print(f"{'='*60}")

    # 路径配置
    base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"
    runs_folder = "chilean_NoRot_NoScale_5cm/"
    pointcloud_fols = "/pointcloud_20m_10overlap/"
    filename = "pointcloud_locations_20m_10overlap.csv"
    output_name = "chilean"

    # 验证基础路径
    path = os.path.join(base_path, runs_folder)
    print(f"Base path: {path}")

    if not os.path.exists(path):
        print(f"❌ Error: Base path {path} does not exist!")
        exit(1)

    # 获取所有session folders
    all_folders = sorted(os.listdir(path))
    print(f"Found {len(all_folders)} total folders")

    # 筛选出有效的session folders（数字命名）
    valid_folders = []
    for folder in all_folders:
        if not folder.startswith('.') and folder.isdigit():
            valid_folders.append(folder)

    valid_folders.sort(key=int)
    print(f"Valid session folders: {len(valid_folders)}")

    if len(valid_folders) > 0:
        print(f"Session range: {valid_folders[0]} - {valid_folders[-1]}")

    # 划分数据库和查询sessions
    database_folders = []
    query_folders = []

    for folder in valid_folders:
        session_num = int(folder)
        if DATABASE_SESSION_START <= session_num <= DATABASE_SESSION_END:
            database_folders.append(folder)
        elif QUERY_SESSION_START <= session_num <= QUERY_SESSION_END:
            query_folders.append(folder)

    print(f"\n{'='*60}")
    print(f"Session划分")
    print(f"{'='*60}")
    print(f"数据库sessions ({DATABASE_SESSION_START}-{DATABASE_SESSION_END}): {len(database_folders)} sessions")
    if database_folders:
        print(f"  Sessions: {', '.join(database_folders)}")
    print(f"查询sessions ({QUERY_SESSION_START}-{QUERY_SESSION_END}): {len(query_folders)} sessions")
    if query_folders:
        print(f"  Sessions: {', '.join(query_folders)}")

    # 执行构建
    database_sets, query_sets = construct_query_and_database_sets_separate(
        base_path,
        runs_folder,
        database_folders,
        query_folders,
        pointcloud_fols,
        filename,
        output_name
    )

    print(f"\n{'='*60}")
    print(f"✓ 完成!")
    print(f"{'='*60}")
    print(f"生成的文件:")
    print(f"  1. {output_name}_evaluation_database_{DATABASE_SESSION_START}_{DATABASE_SESSION_END}.pickle")
    print(f"  2. {output_name}_evaluation_query_{QUERY_SESSION_START}_{QUERY_SESSION_END}.pickle")