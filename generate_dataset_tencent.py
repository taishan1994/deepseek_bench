"""
从 ShareGPT_V3_unfiltered_cleaned_split.json 中采样 3000 条数据（腾讯评测用）：
- 最大输入 16k tokens
- 目标最终平均输入约 3.5k tokens
- 数据 token 数近似正态分布
- 每条源样本 **最多使用一次**（样本 ID 不重复）
- 允许在样本内部通过删除 / 复制对话轮次来调整 token 长度
- 不改变样本的基本结构：仍然是 ShareGPT 格式的 conversations

实现细节：
- 由于实际调整后的均值会略低于采样目标均值，脚本在采样时会
  使用略高于 3.5k 的目标均值（例如 3.8k），以使最终实际均值
  更接近 3.5k。

用法：
    cd /nfs/FM/gongoubo/new_project/workflow/deepseek_bench
    python3 generate_dataset_tencent.py

输出：
    data/sharegpt_normal_distribution_3000_tencent.json
"""
import copy
import json
from pathlib import Path

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


def adjust_to_target_exact(conversations, tokenizer, target_tokens, max_tokens=16000):
    """对单条样本做精细调整，使其输入 token 数尽量接近 target_tokens。

    策略：
    1. 如果超过 max_tokens：先删除末尾对话轮次，仍超则截断最后一个输入轮次的文本。
    2. 如果明显低于 target_tokens：复制已有输入轮次（例如从前往后循环复制），拉长上下文，直到接近目标或接近上限。
    3. 最后再做一次“如果仍超过 target_tokens，则继续删除末尾轮次”以细调。

    注意：
    - build_index 已经过滤掉 len(conversations) < 2 的样本，所以这里默认至少有 1 轮输入 + 1 轮输出。
    """
    input_convs = conversations[:-1]
    output_conv = conversations[-1]

    if not input_convs:
        # 理论上不会出现（因为 build_index 过滤了），以防万一保守处理
        return conversations

    # 当前 token 数（注意 calculate_input_tokens 会忽略最后一轮）
    current_tokens = calculate_input_tokens(input_convs + [output_conv], tokenizer)

    # 1) 先确保不超过 max_tokens：删除末尾轮次
    while current_tokens > max_tokens and len(input_convs) > 1:
        input_convs = input_convs[:-1]
        current_tokens = calculate_input_tokens(input_convs + [output_conv], tokenizer)

    # 如果仍然超过 max_tokens，截断最后一个输入轮次
    if current_tokens > max_tokens and len(input_convs) > 0:
        last_input = input_convs[-1]
        original_text = last_input.get("value", "")
        ratio = max_tokens / max(current_tokens, 1)
        if ratio > 0.01:  # 至少保留 1%
            truncated_text = original_text[: int(len(original_text) * ratio)]
            input_convs[-1] = {"from": last_input.get("from", "human"), "value": truncated_text}
            current_tokens = calculate_input_tokens(input_convs + [output_conv], tokenizer)

    # 2) 如果低于目标，复制对话轮次拉长上下文
    if current_tokens < target_tokens and len(input_convs) >= 1:
        # 使用最后一轮输入来估算“复制一轮”的增量
        base_tokens = calculate_input_tokens(input_convs + [output_conv], tokenizer)
        test_convs = input_convs + [input_convs[-1]]
        new_tokens = calculate_input_tokens(test_convs + [output_conv], tokenizer)
        tokens_per_conv = new_tokens - base_tokens

        # 极端情况下（模板导致差值为 0），用一个保守估计避免卡死
        if tokens_per_conv <= 0:
            tokens_per_conv = max(1, base_tokens // max(len(input_convs), 1))

        needed = target_tokens - current_tokens
        num_to_add = int(needed / tokens_per_conv) + 3  # 稍微多加几轮，保证能到目标

        max_convs = 80  # 控制单条对话轮次数，允许比之前更长一些
        max_iterations = max(0, min(num_to_add, max_convs - len(input_convs)))

        for i in range(max_iterations):
            conv_to_copy = input_convs[i % len(input_convs)]
            input_convs.append(conv_to_copy)
            current_tokens = calculate_input_tokens(input_convs + [output_conv], tokenizer)
            if current_tokens >= target_tokens or current_tokens >= max_tokens:
                break

        # 再检查一次是否超过 max_tokens，若超过则删掉末尾轮次
        while current_tokens > max_tokens and len(input_convs) > 1:
            input_convs = input_convs[:-1]
            current_tokens = calculate_input_tokens(input_convs + [output_conv], tokenizer)

    # 3) 最终细调：如果略微超过 target_tokens，就删除末尾轮次
    while current_tokens > target_tokens and len(input_convs) > 1:
        input_convs = input_convs[:-1]
        current_tokens = calculate_input_tokens(input_convs + [output_conv], tokenizer)

    return input_convs + [output_conv]


def build_index(source_data, tokenizer, max_tokens=16000):
    """为源数据构建按输入 token 数排序的索引。"""
    data_with_tokens = []

    for idx, item in enumerate(source_data):
        if idx % 1000 == 0:
            print(f"[build_index] 计算 token 中: {idx}/{len(source_data)}")

        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue

        token_count = calculate_input_tokens(conversations, tokenizer)
        if 0 < token_count <= max_tokens:
            data_with_tokens.append({
                "data": item,
                "tokens": token_count,
            })

    print(f"[build_index] 有效样本数: {len(data_with_tokens)}")

    # 按 token 数排序，便于后续用二分查找匹配目标 token
    data_with_tokens.sort(key=lambda x: x["tokens"])
    token_list = np.array([d["tokens"] for d in data_with_tokens], dtype=np.int32)

    print(
        f"[build_index] token 统计 -> 最小: {token_list.min()}, 最大: {token_list.max()}, "
        f"平均: {token_list.mean():.2f}, 标准差: {token_list.std():.2f}"
    )

    return data_with_tokens, token_list


def generate_tencent_dataset(
    source_data,
    tokenizer,
    target_count=3000,
    mean_tokens=3800,
    max_tokens=16000,
    std_dev=2000,
    min_tokens=2000,
):
    """生成腾讯评测用数据集：
    - 先根据正态分布生成 3000 个目标 token 数；
    - 再为每个目标 token 选一个“最接近且未使用过”的样本；
    - 然后在该样本内部通过删除 / 复制对话轮次，把 token 数调到接近目标值。

    这里的 mean_tokens 和 min_tokens 稍微偏大，是为了抵消实际调整
    过程中的偏差，使得最终数据集的均值更接近 3.5k。
    """
    # 1) 生成目标 token 序列
    np.random.seed(42)
    target_tokens = np.random.normal(mean_tokens, std_dev, target_count)
    target_tokens = np.clip(target_tokens, min_tokens, max_tokens).astype(int)

    print("[target] 目标 token 分布统计:")
    print(f"  数量: {len(target_tokens)}")
    print(f"  采样用均值: {mean_tokens}")
    print(f"  实际目标均值: {np.mean(target_tokens):.2f}")
    print(f"  标准差: {np.std(target_tokens):.2f}")
    print(f"  最小值: {np.min(target_tokens)}")
    print(f"  最大值: {np.max(target_tokens)}")

    # 2) 构建按 token 排序的索引
    print("\n[index] 构建索引…")
    data_with_tokens, token_list = build_index(source_data, tokenizer, max_tokens=max_tokens)

    # 3) 为每个目标 token 选一个唯一样本，并在内部做精细调整
    print("\n[select] 开始选择并调整样本…")
    selected_data = []
    used_indices = set()
    n = len(data_with_tokens)

    for i, target in enumerate(target_tokens):
        if i % 100 == 0:
            print(f"[select] 进度: {i}/{target_count}")

        # 在 token_list 中找到最接近 target 的位置
        pos = int(np.searchsorted(token_list, target))
        window = 10
        best_idx = None
        best_diff = None

        # 先在 pos 附近的小窗口内找还没用过的样本
        for offset in range(-window, window + 1):
            j = pos + offset
            if 0 <= j < n and j not in used_indices:
                diff = abs(token_list[j] - target)
                if best_idx is None or diff < best_diff:
                    best_idx = j
                    best_diff = diff

        # 如果窗口内都被用完，逐步扩大半径继续找未使用样本
        if best_idx is None:
            for radius in range(window + 1, n):
                left = pos - radius
                right = pos + radius
                found = False

                if 0 <= left < n and left not in used_indices:
                    best_idx = left
                    found = True
                elif 0 <= right < n and right not in used_indices:
                    best_idx = right
                    found = True

                if found:
                    break

        if best_idx is None:
            print(f"[select][警告] 第 {i} 个目标 {target} 未找到可用样本，跳过。")
            continue

        # 深拷贝样本，并在内部进行 token 长度调整
        sample = copy.deepcopy(data_with_tokens[best_idx]["data"])
        conversations = sample.get("conversations", [])

        if len(conversations) >= 2:
            adjusted_convs = adjust_to_target_exact(
                conversations, tokenizer, target_tokens=target, max_tokens=max_tokens
            )
            sample["conversations"] = adjusted_convs

        selected_data.append(sample)
        used_indices.add(best_idx)

    print(f"[select] 最终成功生成样本数: {len(selected_data)}")
    return selected_data


def analyze_selected(selected_data, tokenizer):
    """对最终采样结果做一次 token 统计，验证是否符合预期。"""
    print("[analyze] 开始统计最终数据集 token 分布…")
    token_counts = []

    for idx, item in enumerate(selected_data):
        if idx % 100 == 0:
            print(f"[analyze] 计算 token: {idx}/{len(selected_data)}")
        conversations = item.get("conversations", [])
        token_counts.append(calculate_input_tokens(conversations, tokenizer))

    token_counts = np.array(token_counts, dtype=np.int32)

    print("[analyze] 最终数据集 token 统计:")
    print(f"  样本数: {len(token_counts)}")
    print(f"  平均值: {np.mean(token_counts):.2f}")
    print(f"  标准差: {np.std(token_counts):.2f}")
    print(f"  中位数: {np.median(token_counts):.2f}")
    print(f"  最小值: {np.min(token_counts)}")
    print(f"  最大值: {np.max(token_counts)}")

    return token_counts


def main():
    # 文件与模型路径（可按需修改）
    repo_root = Path(__file__).resolve().parent
    source_file = repo_root / "data" / "ShareGPT_V3_unfiltered_cleaned_split.json"
    output_file = repo_root / "data" / "sharegpt_normal_distribution_3000.json"
    model_path = "/nfs/FM/checkpoints/DeepSeek-R1-BF16"

    target_count = 3000
    desired_mean_tokens = 3500  # 希望最终接近的均值
    sampling_mean_tokens = 3800  # 采样时使用的目标均值，略高以补偿偏差
    max_tokens = 16000
    std_dev = 2000

    print("=" * 60)
    print("从 ShareGPT 源数据采样并调整 3000 条（腾讯评测用）")
    print("=" * 60)
    print(f"源数据: {source_file}")
    print(f"输出文件: {output_file}")
    print(f"目标样本数: {target_count}")
    print(f"期望最终平均 token: {desired_mean_tokens}")
    print(f"采样用均值: {sampling_mean_tokens}")
    print(f"最大 token: {max_tokens}")
    print(f"标准差: {std_dev}")

    # 加载源数据
    print("\n[main] 加载源数据…")
    with source_file.open("r", encoding="utf-8") as f:
        source_data = json.load(f)
    print(f"[main] 源数据总量: {len(source_data)}")

    # 加载 tokenizer
    print("\n[main] 加载 tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 生成数据集（含内部文本调整）
    print("\n[main] 生成数据集…")
    selected_data = generate_tencent_dataset(
        source_data,
        tokenizer,
        target_count=target_count,
        mean_tokens=sampling_mean_tokens,
        max_tokens=max_tokens,
        std_dev=std_dev,
        min_tokens=2000,
    )

    # 验证分布
    print("\n[main] 验证最终分布…")
    _ = analyze_selected(selected_data, tokenizer)

    # 保存结果
    print(f"\n[main] 保存结果到: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)

    print("\n[main] 完成！")


if __name__ == "__main__":
    main()
