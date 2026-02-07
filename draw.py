import matplotlib.pyplot as plt

# 数据
mechanisms = ["Self-attn", "Cross-Attn (A)", "Cross-Attn (S)", "Dual Cross-Attn"]
accuracy = [0.675, 0.671, 0.610, 0.687]

# 颜色方案（每种机制不同颜色）
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 蓝色、橙色、绿色、红色

# 画图
plt.figure(figsize=(3.2, 2.5))  # 适合单栏论文的尺寸
bars = plt.bar(mechanisms, accuracy, color=colors, width=0.6)

plt.ylabel("Accuracy", fontsize=8)
plt.ylim(0, 0.7)

# 调整 x 轴标签，使其更紧凑
plt.xticks(rotation=20, fontsize=7)
plt.yticks(fontsize=7)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# 去掉外边框，让图表更清爽
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# 添加图例
plt.legend(bars, mechanisms, loc="upper right", fontsize=7, frameon=False)

# 保存高分辨率图片
plt.savefig("attention_mechanisms_legend.png", dpi=600, bbox_inches="tight")
plt.show()
