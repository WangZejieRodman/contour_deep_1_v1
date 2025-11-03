"""
读取TensorBoard日志并打印/绘图
用法: python scripts/read_tensorboard_logs.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np


def read_tensorboard_logs(log_dir):
    """读取TensorBoard日志"""
    print(f"读取日志: {log_dir}")

    # 找到events文件
    events_file = None
    for file in os.listdir(log_dir):
        if file.startswith('events.out.tfevents.1762104923.wzj-System-Product-Name.1425064.0'):
            events_file = os.path.join(log_dir, file)
            break

    if events_file is None:
        print("错误: 找不到events文件")
        return None

    print(f"找到文件: {events_file}")

    # 加载events
    ea = event_accumulator.EventAccumulator(events_file)
    ea.Reload()

    # 获取所有标量tags
    tags = ea.Tags()['scalars']
    print(f"\n可用的标量tags: {tags}")

    # 读取数据
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
        print(f"  {tag}: {len(steps)} 个数据点")

    return data


def plot_training_curves(data):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Epoch损失
    ax = axes[0, 0]
    if 'Epoch/train_loss' in data:
        ax.plot(data['Epoch/train_loss']['steps'],
                data['Epoch/train_loss']['values'],
                label='Train Loss', linewidth=2, marker='o')
    if 'Epoch/val_loss' in data:
        ax.plot(data['Epoch/val_loss']['steps'],
                data['Epoch/val_loss']['values'],
                label='Val Loss', linewidth=2, marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 学习率
    ax = axes[0, 1]
    if 'Train/lr' in data:
        ax.plot(data['Train/lr']['steps'],
                data['Train/lr']['values'],
                linewidth=2, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 3. 训练损失（step级别）
    ax = axes[1, 0]
    if 'Train/loss_step' in data:
        steps = data['Train/loss_step']['steps']
        values = data['Train/loss_step']['values']
        # 平滑处理
        window = 50
        if len(values) > window:
            smoothed = np.convolve(values, np.ones(window) / window, mode='valid')
            ax.plot(steps[:len(smoothed)], smoothed, linewidth=2, alpha=0.8)
        else:
            ax.plot(steps, values, linewidth=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (Smoothed)')
    ax.grid(True, alpha=0.3)

    # 4. 统计摘要
    ax = axes[1, 1]
    ax.axis('off')

    # 提取关键信息
    summary_text = "training abstract\n" + "=" * 40 + "\n\n"

    if 'Epoch/train_loss' in data:
        train_losses = data['Epoch/train_loss']['values']
        summary_text += f"train_loss:\n"
        summary_text += f"  start: {train_losses[0]:.4f}\n"
        summary_text += f"  final: {train_losses[-1]:.4f}\n"
        summary_text += f"  start-final: {train_losses[0] - train_losses[-1]:.4f}\n\n"

    if 'Epoch/val_loss' in data:
        val_losses = data['Epoch/val_loss']['values']
        summary_text += f"val_loss:\n"
        summary_text += f"  start: {val_losses[0]:.4f}\n"
        summary_text += f"  final: {val_losses[-1]:.4f}\n"
        summary_text += f"  best: {min(val_losses):.4f}\n"
        summary_text += f"  start-final: {val_losses[0] - min(val_losses):.4f}\n\n"

    if 'Train/lr' in data:
        lrs = data['Train/lr']['values']
        summary_text += f"lr:\n"
        summary_text += f"  start: {lrs[0]:.6f}\n"
        summary_text += f"  final: {lrs[-1]:.6f}\n"

    ax.text(0.1, 0.5, summary_text,
            fontsize=11,
            family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    # 保存
    output_path = '/home/wzj/pan1/contour_deep_1++/scripts/logs/retrieval_adaptive_v1/retrieval_adaptive_v1.png'
    os.makedirs('/home/wzj/pan1/contour_deep_1++/scripts/logs/retrieval_adaptive_v1', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 训练曲线已保存: {output_path}")

    plt.show()


def print_summary(data):
    """打印训练摘要"""
    print("\n" + "=" * 60)
    print("训练摘要")
    print("=" * 60)

    if 'Epoch/train_loss' in data:
        train_losses = data['Epoch/train_loss']['values']
        print(f"\n训练损失:")
        print(f"  初始: {train_losses[0]:.4f}")
        print(f"  最终: {train_losses[-1]:.4f}")
        print(f"  最小: {min(train_losses):.4f}")
        print(f"  下降: {train_losses[0] - train_losses[-1]:.4f}")

    if 'Epoch/val_loss' in data:
        val_losses = data['Epoch/val_loss']['values']
        best_epoch = val_losses.index(min(val_losses)) + 1
        print(f"\n验证损失:")
        print(f"  初始: {val_losses[0]:.4f}")
        print(f"  最终: {val_losses[-1]:.4f}")
        print(f"  最佳: {min(val_losses):.4f} (Epoch {best_epoch})")
        print(f"  下降: {val_losses[0] - min(val_losses):.4f}")

    if 'Train/lr' in data:
        lrs = data['Train/lr']['values']
        print(f"\n学习率:")
        print(f"  初始: {lrs[0]:.6f}")
        print(f"  最终: {lrs[-1]:.6f}")

def print_per_epoch_details(data):
    """打印每轮训练后的详细指标：Epoch/train_loss, Epoch/val_loss, 对应的Train/lr和附近的Train/loss_step"""
    print("\n" + "="*60)
    print("每轮训练详细指标 (Epoch-level)")
    print("="*60)

    train_loss_data = data.get('Epoch/train_loss', {'values': [], 'steps': []})
    val_loss_data = data.get('Epoch/val_loss', {'values': [], 'steps': []})
    lr_data = data.get('Train/lr', {'values': [], 'steps': []})
    step_loss_data = data.get('Train/loss_step', {'values': [], 'steps': []})

    num_epochs = len(train_loss_data['values'])
    print(f"\n共训练 {num_epochs} 个 Epoch。\n")

    for epoch_idx in range(num_epochs):
        epoch_num = epoch_idx + 1  # Epoch 从 1 开始更直观
        train_loss = train_loss_data['values'][epoch_idx]
        val_loss = val_loss_data['values'][epoch_idx]
        epoch_step = train_loss_data['steps'][epoch_idx]

        # 找到最接近该 epoch_step 的 lr
        lr_values = lr_data['values']
        lr_steps = lr_data['steps']
        lr_at_epoch = None
        for i, step in enumerate(lr_steps):
            if step <= epoch_step:
                lr_at_epoch = lr_values[i]
            else:
                break  # lr 是按 step 递增记录的，超过就停止查找

        # 如果没有找到合适的 lr，给出提示
        if lr_at_epoch is None:
            lr_at_epoch = float('nan')
            lr_warning = " (未找到对应学习率)"
        else:
            lr_warning = ""

        # Train/loss_step: 也是按 step，可以找该 epoch 内的部分 step loss
        step_loss_values = step_loss_data['values']
        step_loss_steps = step_loss_data['steps']
        step_losses_in_epoch = []
        for i, step in enumerate(step_loss_steps):
            if step <= epoch_step:
                step_losses_in_epoch.append((step, step_loss_values[i]))
            else:
                break

        # 例如：打印该 epoch 内的第一个、最后一个 step loss
        step_loss_samples = ""
        if step_losses_in_epoch:
            step_loss_samples = f"\n    Step Loss Samples (in this epoch):\n"
            step_loss_samples += f"      Step {step_losses_in_epoch[0][0]}: Loss = {step_losses_in_epoch[0][1]:.4f}"
            if len(step_losses_in_epoch) > 1:
                step_loss_samples += f"\n      Step {step_losses_in_epoch[-1][0]}: Loss = {step_losses_in_epoch[-1][1]:.4f}"
            if len(step_losses_in_epoch) > 2:
                mid_idx = len(step_losses_in_epoch) // 2
                step_loss_samples += f"\n      Step {step_losses_in_epoch[mid_idx][0]}: Loss = {step_losses_in_epoch[mid_idx][1]:.4f}"

        print(f"Epoch {epoch_num}:")
        print(f"  Train Loss:    {train_loss:.4f}")
        print(f"  Val Loss:      {val_loss:.4f}")
        print("-" * 40)

def main():
    """主函数"""
    print("=" * 60)
    print("读取TensorBoard日志")
    print("=" * 60)

    log_dir = '/home/wzj/pan1/contour_deep_1++/scripts/logs/retrieval_adaptive_v1'

    if not os.path.exists(log_dir):
        print(f"错误: 日志目录不存在 {log_dir}")
        return

    # 读取日志
    data = read_tensorboard_logs(log_dir)

    if data is None:
        return

    # 打印摘要
    print_summary(data)

    # 打印每轮详细指标
    print_per_epoch_details(data)

    # 绘制曲线
    plot_training_curves(data)

    print("\n✓ 完成!")


if __name__ == "__main__":
    main()