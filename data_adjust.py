import json

# 读取 JSON 文件
input_file = "/home/hongfei/emollm/data/train_annotation.json"
output_file = "/home/hongfei/emollm/data/train_annotation_emotion.json"

# 加载数据
with open(input_file, "r") as f:
    data = json.load(f)

# 替换 "sadness" 为 "sad"
for item in data["annotation"]:
    if item["emotion"] == "sadness":
        item["emotion"] = "sad"

# 保存修改后的数据
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"替换完成，已保存到 {output_file}")
