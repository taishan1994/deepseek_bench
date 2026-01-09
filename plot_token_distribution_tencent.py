"""\
绘制腾讯评测用数据集的 token 分布图。

- 输入：data/sharegpt_normal_distribution_3000_tencent.json
- 模型：/nfs/FM/checkpoints/DeepSeek-R1-BF16
- 输出图片：token_length_distribution_tencent.png

用法：
    cd /nfs/FM/gongoubo/new_project/workflow/deepseek_bench
    python3 plot_token_distribution_tencent.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


def calculate_input_tokens(conversations, tokenizer):
    """计算一条样本的输入 token 数（只统计最后一轮之前的对话）。"""
    messages = []
    for conv in conversations[:-1]:  # 排除最后一个（视作输出）
        role = "user" if conv.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": conv.get("value", "")})

    if not messages:
        return 0

    inps = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(inps, return_tensors="pt").input_ids
    return int(input_ids.shape[1])


def main():
    repo_root = Path(__file__).resolve().parent
    dataset_path = repo_root / "data" / "sharegpt_normal_distribution_3000_tencent.json"
    model_path = "/nfs/FM/checkpoints/DeepSeek-R1-BF16"
    output_png = repo_root / "token_length_distribution_tencent.png"

    print("=" * 60)
    print("绘制腾讯评测数据集 token 分布图")
    print("=" * 60)
    print(f"数据集: {dataset_path}")
    print(f"模型:   {model_path}")
    print(f"输出图: {output_png}")

    # 加载数据
    print("\n[main] 加载数据…")
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[main] 样本数: {len(data)}")

    # 加载 tokenizer
    print("\n[main] 加载 tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 计算每条样本的 token 数
    print("\n[main] 计算每条样本的 token 数…")
    token_counts = []
    for idx, item in enumerate(data):
        if idx % 100 == 0:
            print(f"  进度: {idx}/{len(data)}")
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue
        token_counts.append(calculate_input_tokens(convs, tokenizer))

    token_counts = np.array(token_counts, dtype=np.int32)
    print("\n[stat] token 统计:")
    print(f"  样本数: {len(token_counts)}")
    print(f"  平均值: {np.mean(token_counts):.2f}")
    print(f"  标准差: {np.std(token_counts):.2f}")
    print(f"  中位数: {np.median(token_counts):.2f}")
    print(f"  最小值: {np.min(token_counts)}")
    print(f"  最大值: {np.max(token_counts)}")

    # 绘图
    print("\n[plot] 绘制直方图…")
    plt.figure(figsize=(10, 6))

    # 直方图
    max_tokens = 16000
    bins = 50
    plt.hist(
        token_counts,
        bins=bins,
        range=(0, max_tokens),
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
        label="Empirical distribution",
    )

    # 叠加一个基于样本均值/方差的正态分布曲线，方便对比
    mu = np.mean(token_counts)
    sigma = np.std(token_counts)
    x = np.linspace(0, max_tokens, 1000)
    normal_pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    plt.plot(x, normal_pdf, "r-", lw=2, label=f"Normal($\\mu$={mu:.0f}, $\\sigma$={sigma:.0f})")

    plt.title("Token Length Distribution (Tencent Dataset)")
    plt.xlabel("Input token length")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    print(f"[plot] 保存图片到: {output_png}")


if __name__ == "__main__":
    main()
