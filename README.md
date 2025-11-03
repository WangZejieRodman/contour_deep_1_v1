# Contour Deep 1+

深度学习场景识别系统 - 基于多层BEV表示的点云检索网络

## 📋 项目简介

本项目是对传统Contour Context方法的深度学习改进（方向1），通过端到端训练的神经网络提取点云场景的全局特征，用于大规模场景识别与定位。

**数据集**: Chilean Underground Mine Dataset  
**Baseline**: Contour Context (Recall@1: 73.14%)  
**目标**: 超越传统方法，提升检索准确率

## 🚀 快速开始

### 1. 环境配置
```bash
# Python 3.8+, PyTorch 2.0+, CUDA 11.8+
pip install -r requirements.txt
```

### 2. 数据预处理
```bash
# 生成训练/验证集BEV缓存
python scripts/preprocess_bev_adaptive.py --split all

# 生成评估集BEV缓存（跨时间段）
python scripts/preprocess_bev_evaluation_adaptive.py --split all
```

### 3. 训练
```bash
python scripts/run_training_adaptive.py
```

### 4. 评估
```bash
# 训练集自查询（模型正确性检查）
python scripts/analyze_and_eval_self_adaptive.py

# 跨时间段评估（真实性能）
python scripts/analyze_and_eval_adaptive.py
```

## 📊 核心创新

- **多层BEV表示**: 8层高度分层 + 垂直复杂度图
- **多尺度特征提取**: 3×3, 7×7, 15×15卷积并行
- **双重注意力机制**: 空间注意力 + 跨层注意力
- **Triplet Loss**: Hard Mining策略

## 📁 项目结构

```
contour_deep_1/
├── configs/          # YAML配置文件
├── data/             # 数据加载与预处理
├── models/           # 网络架构
├── training/         # 训练框架
├── scripts/          # 执行脚本
└── utils/            # 工具函数
```

## 🎯 当前进展

- ✅ **Day 1-7**: 框架搭建、数据准备、网络实现
- ✅ **Day 8**: 完整训练（50 epochs）
- ✅ **Day 9**: 评估分析

**最新结果**:
- 训练集自查询: Recall@1 = XX%
- 跨时间段评估: Recall@1 = XX%（目标 >78%）

## ⚙️ 配置说明

- `config_base.yaml`: BEV生成、数据集、硬件配置
- `config_retrieval.yaml`: 模型、优化器、训练超参数

## 📈 监控训练

```bash
tensorboard --logdir logs
```

## 🔧 故障排查

1. **OOM错误**: 减小batch_size或使用梯度累积
2. **缓存缺失**: 先运行预处理脚本
3. **性能不佳**: 检查数据增强、调整margin参数

## 📝 下一步

- [ ] Day 10: 超参数优化
- [ ] Day 11-14: 方向2（BCI匹配网络）
- [ ] Day 15+: 端到端联合训练

## 📄 许可

MIT License

## 🙏 致谢

基于原始Contour Context工作进行改进
