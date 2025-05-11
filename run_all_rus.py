from utils.preprocessing import load_and_preprocess
from utils.evaluation import evaluate_model
from utils.plots import plot_curves

from resampling.random_undersample import apply_random_undersampler  

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/churn-bigml-80.csv", "data/churn-bigml-20.csv"
)


X_train, y_train = apply_random_undersampler(X_train, y_train)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}


results = []


for name, model in models.items():
    print(f"\n {name} Results:\n" + "-"*50)

    metrics, roc_data, pr_data = evaluate_model(model, X_train, y_train, X_test, y_test)

 
    print("Classification Report:\n", metrics["classification_report"])
    print(f"Accuracy : {metrics['accuracy']:.2f}")
    print(f"F1 Score : {metrics['f1']:.2f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.2f}")
    print(f"PR-AUC   : {metrics['pr_auc']:.2f}")

    
    plot_curves(roc_data, pr_data, model_name=f"{name} (RandomUnderSampler)")

    
    results.append({
        "Model": name,
        "Accuracy": round(metrics["accuracy"], 2),
        "Recall": round(metrics["recall"], 2),
        "Precision": round(metrics["precision"], 2),
        "F1": round(metrics["f1"], 2),
        "ROC-AUC": round(metrics["roc_auc"], 2),
        "PR-AUC": round(metrics["pr_auc"], 2)
    })


results_df = pd.DataFrame(results)
print("\n Model Performance Table:\n")
print(results_df)


os.makedirs("data", exist_ok=True)
results_df.to_csv("data/run_all_results_RUS.csv", index=False)
print("\n Results saved to data/run_all_results_RUS.csv file.")


metrics_to_plot = ["F1", "ROC-AUC", "PR-AUC"]
melted = results_df.melt(id_vars="Model", value_vars=metrics_to_plot,
                         var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=melted, x="Model", y="Score", hue="Metric")
plt.title("Model Comparison (with RandomUnderSampler)")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.tight_layout()


os.makedirs("figures", exist_ok=True)
plt.savefig("figures/barplot_RUS.png")
print("Graph saved: figures/barplot_RUS.png")


plt.show()
