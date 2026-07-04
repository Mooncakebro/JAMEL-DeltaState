#!/usr/bin/env python3
"""
aggregate_eval_results.py
遍历指定前缀的所有实验结果文件夹，汇总每个app的reward统计信息并绘图。

用法:
    python aggregate_eval_results.py --base_dir /home/songyuebing/JAMEL-DeltaState/outputs/ --prefix eval_hybrid8_test10
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def find_matching_folders(base_dir, prefix):
    """在base_dir下查找所有以prefix开头的文件夹"""
    matched = []
    for item in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith(prefix):
            # 检查是否包含aggregate_results.json
            json_path = os.path.join(full_path, "aggregate_results.json")
            if os.path.exists(json_path):
                matched.append(full_path)
    return matched


def load_aggregate_results(folder_path):
    """加载单个文件夹中的aggregate_results.json"""
    json_path = os.path.join(folder_path, "aggregate_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def collect_rewards_by_app(folders):
    """
    从所有文件夹中收集每个app的cumulative_reward。
    返回: dict, key=app_name(env_id), value=list of cumulative_rewards
    """
    app_rewards = defaultdict(list)
    app_order = []  # 保持app出现的顺序

    for folder in folders:
        data = load_aggregate_results(folder)
        per_app = data.get("per_app", [])
        for app_info in per_app:
            env_id = app_info.get("env_id", "unknown")
            cumulative_reward = app_info.get("cumulative_reward", 0.0)
            if env_id not in app_rewards:
                app_order.append(env_id)
            app_rewards[env_id].append(cumulative_reward)

    return app_rewards, app_order


def compute_statistics(app_rewards, app_order):
    """
    对每个app计算统计量: mean, std, min, max
    返回: dict, key=app_name, value=dict(mean, std, min, max, n)
    """
    stats = {}
    for app_name in app_order:
        rewards = np.array(app_rewards[app_name])
        stats[app_name] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0,
            "min": np.min(rewards),
            "max": np.max(rewards),
            "n": len(rewards),
        }
    return stats


def plot_results(stats, app_order, prefix, save_dir):
    """绘制结果图表"""
    n_apps = len(app_order)
    x = np.arange(n_apps)
    width = 0.6

    means = [stats[app]["mean"] for app in app_order]
    stds = [stats[app]["std"] for app in app_order]
    mins = [stats[app]["min"] for app in app_order]
    maxs = [stats[app]["max"] for app in app_order]

    fig, ax = plt.subplots(figsize=(max(10, n_apps * 1.2), 7))

    # 绘制均值柱状图，误差条为1倍sigma
    bars = ax.bar(
        x, means, width,
        yerr=stds,
        capsize=5,
        color="#4C72B0",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.8,
        label="Mean ± 1σ",
        error_kw={"elinewidth": 1.5, "ecolor": "#DD8452"},
    )

    # 在每个柱子上标注min和max
    for i, app in enumerate(app_order):
        ax.plot(
            [x[i], x[i]],
            [mins[i], maxs[i]],
            color="#555555",
            linewidth=1.5,
            zorder=3,
        )
        # 最高值标记（上三角）
        ax.plot(x[i], maxs[i], marker="^", color="#D62728", markersize=7, zorder=4)
        # 最低值标记（下三角）
        ax.plot(x[i], mins[i], marker="v", color="#2CA02C", markersize=7, zorder=4)

        # 标注数值
        ax.text(
            x[i], maxs[i] + 0.5,
            f"{maxs[i]:.1f}",
            ha="center", va="bottom", fontsize=8, color="#D62728", fontweight="bold",
        )
        ax.text(
            x[i], mins[i] - 0.5,
            f"{mins[i]:.1f}",
            ha="center", va="top", fontsize=8, color="#2CA02C", fontweight="bold",
        )
        # 均值标注
        ax.text(
            x[i], means[i],
            f"{means[i]:.1f}",
            ha="center", va="center", fontsize=9, color="black", fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(app_order, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Cumulative Reward", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Per-App Reward Statistics (Prefix: {prefix}, {stats[app_order[0]]['n'] if app_order else 0} runs)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 计算整体累加的最高和最低reward
    overall_max = sum(stats[app]["max"] for app in app_order)
    overall_min = sum(stats[app]["min"] for app in app_order)
    overall_mean = sum(stats[app]["mean"] for app in app_order)

    # 在图上添加整体统计信息文本框
    textstr = (
        f"Overall (10 apps summed):\n"
        f"  Max Reward:   {overall_max:.1f}\n"
        f"  Min Reward:   {overall_min:.1f}\n"
        f"  Mean Reward:  {overall_mean:.1f}\n"
        f"  Range:        {overall_max - overall_min:.1f}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.9, edgecolor="gray")
    ax.text(
        0.4, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        fontfamily="monospace",
    )

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(save_dir, f"{prefix}_aggregate_reward.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存至: {save_path}")

    # 同时保存统计数据为JSON
    json_save_path = os.path.join(save_dir, f"{prefix}_aggregate_stats.json")
    summary = {
        "prefix": prefix,
        "num_runs": len(stats[app_order[0]]) if app_order else 0,
        "per_app_stats": {app: {k: float(v) for k, v in s.items()} for app, s in stats.items()},
        "overall": {
            "max_sum": float(overall_max),
            "min_sum": float(overall_min),
            "mean_sum": float(overall_mean),
            "range": float(overall_max - overall_min),
        },
    }
    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"统计数据已保存至: {json_save_path}")

    plt.close(fig)
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="汇总指定前缀的实验结果并绘制reward统计图"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/songyuebing/JAMEL-DeltaState/outputs/",
        help="实验结果存放的根目录",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="实验结果文件夹的前缀，例如 eval_hybrid8_test10",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    prefix = args.prefix

    # 1. 查找匹配的文件夹
    print(f"在 {base_dir} 中查找前缀为 '{prefix}' 的文件夹...")
    matched_folders = find_matching_folders(base_dir, prefix)

    if not matched_folders:
        print(f"未找到任何以 '{prefix}' 开头的文件夹，请检查路径和前缀。")
        return

    print(f"找到 {len(matched_folders)} 个匹配文件夹:")
    for f in matched_folders:
        print(f"  - {f}")

    # 2. 收集每个app的reward
    print("\n正在收集各app的reward数据...")
    app_rewards, app_order = collect_rewards_by_app(matched_folders)

    if not app_order:
        print("未找到任何app数据，请检查aggregate_results.json格式。")
        return

    print(f"共发现 {len(app_order)} 个app: {app_order}")

    # 3. 计算统计量
    stats = compute_statistics(app_rewards, app_order)

    # 打印统计摘要
    print("\n" + "=" * 70)
    print(f"{'App':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Runs':>6}")
    print("-" * 70)
    for app in app_order:
        s = stats[app]
        print(f"{app:<20} {s['mean']:>8.2f} {s['std']:>8.2f} {s['min']:>8.1f} {s['max']:>8.1f} {s['n']:>6d}")
    print("=" * 70)

    overall_max = sum(stats[app]["max"] for app in app_order)
    overall_min = sum(stats[app]["min"] for app in app_order)
    print(f"\n整体累加最高Reward: {overall_max:.1f}")
    print(f"整体累加最低Reward: {overall_min:.1f}")
    print(f"差值: {overall_max - overall_min:.1f}")

    # 4. 绘图并保存
    print("\n正在绘图...")
    plot_results(stats, app_order, prefix, base_dir)

    print("\n完成！")


if __name__ == "__main__":
    main()