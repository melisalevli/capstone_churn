import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("data/leakage_safe_cv_results.csv")


plt.figure(figsize=(12, 6))
sns.barplot(data=df.sort_values("F1", ascending=False),
            x="Model", y="F1", hue="Resampling")
plt.title("Model Performance Based on F1 Scores (No Data Leakage)")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/leakage_f1_barplot.png")
plt.show()


pivot = df.pivot(index="Model", columns="Resampling", values="F1")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("F1 Heatmap (Leakage-Free CV)")
plt.tight_layout()
plt.savefig("figures/leakage_f1_heatmap.png")
plt.show()

grouped = df.groupby("Resampling")[["F1", "ROC-AUC", "PR-AUC"]].mean().reset_index()
melted = grouped.melt(id_vars="Resampling", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.lineplot(data=melted, x="Resampling", y="Score", hue="Metric", marker="o")
plt.title("Average Performance of Resampling Methods (No Data Leakage)")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("figures/leakage_resampling_avg_scores.png")
plt.show()

print("All images were saved in the figures/ folder.")
