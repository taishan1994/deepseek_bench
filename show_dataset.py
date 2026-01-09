"""
用于构建腾讯对应的3000tiao数据集
"""
import json
import copy

from transformers import AutoTokenizer

path = "data/ShareGPT_V3_unfiltered_cleaned_split.json"
path = "/nfs/FM/gongoubo/new_project/workflow/deepseek_bench/data/sharegpt_normal_distribution_3000_tencent.json"

with open(path, "r") as f:
    data = json.load(f)

model_path = "/nfs/FM/checkpoints/DeepSeek-R1-BF16"

tokenizer = AutoTokenizer.from_pretrained(model_path)

for d in data:
    print(d)
    conversations = d["conversations"][:-1]
    messages = []
    tmp_conversation = copy.deepcopy(conversations)
    for d in conversations:
        if d["from"] == "human":
            messages.append({"role": "user", "content": d["value"]})
        else:
            messages.append({"role": "assistant", "content": d["value"]})

    inps = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(inps, return_tensors="pt").input_ids.numpy().tolist()[0]
    print(len(input_ids))
    break