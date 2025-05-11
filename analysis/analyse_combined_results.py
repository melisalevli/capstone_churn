import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv("data/combined_model_results.csv")


best_f1 = df.loc[df["F1"].idxmax()]
print("Highest F1 score:\n", best_f1)


best_auc = df.loc[df["ROC-AUC"].idxmax()]
print("\nHighest ROC-AUC score:\n", best_auc)


grouped = df.groupby("Resampling")[["F1", "ROC-AUC", "PR-AUC"]].mean().round(3)
print("\nAverage metrics of resampling methods:\n", grouped)


plt.figure(figsize=(12, 6))
sns.barplot(data=df, x="Model", y="F1", hue="Resampling")
plt.title("Model Performance Based on F1-Scores")
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/f1_by_model_resampling.png")
print("\nThe chart has been saved: figures/f1_by_model_resampling.png")
plt.show()


grouped.reset_index(inplace=True)
melted = grouped.melt(id_vars="Resampling", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.lineplot(data=melted, x="Resampling", y="Score", hue="Metric", marker="o")
plt.title("Average Performance by Resampling Methods")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("figures/avg_metric_by_resampling.png")
print("The chart has been saved: figures/avg_metric_by_resampling.png")
plt.show()
