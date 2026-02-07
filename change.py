import os
import random
import json

# 读取现有 JSON 文件
input_json = "/home/hongfei/emollm/data/train_add.json"
output_json = "/home/hongfei/emollm/data/train_new_reduce_neutral_2200.json"

# 加载数据
with open(input_json, "r") as f:
    data = json.load(f)

neutral_samples = []
other_samples = []

# 遍历并修改情感标签，同时筛选 neutral 样本路径
for item in data["annotation"]:
    # 替换 sadness 为 sad
    if item["emotion"] == "sadness":
        item["emotion"] = "sad"

    # 检查路径和情感类型
    if item["emotion"] == "neutral" and item["path"].startswith("/data/hongfei/MELD.Raw/train_splits"):
        neutral_samples.append(item)  # 只添加符合路径的 neutral 样本
    elif item["emotion"] != "neutral":
        other_samples.append(item)  # 添加非 neutral 的其他情感样本

# 下采样 Neutral 类别到 2200 条
reduced_neutral_samples = random.sample(neutral_samples, min(len(neutral_samples), 2200))

# 合并其他类别样本和下采样后的 Neutral 样本
final_annotations = reduced_neutral_samples + other_samples

# 保存新的 JSON 文件
with open(output_json, "w") as f:
    json.dump({"annotation": final_annotations}, f, indent=4)

print(f"仅 /data/hongfei/MELD.Raw/train_splits 目录下的 neutral 样本已下采样，sadness 已改为 sad，结果已保存到 {output_json}")
