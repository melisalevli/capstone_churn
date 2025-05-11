import os
import pickle
import pandas as pd
from utils.preprocessing import load_and_preprocess

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/churn-bigml-80.csv", "data/churn-bigml-20.csv"
)


model_classes = {
    "LogisticRegression": LogisticRegression,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "SVM": SVC,
    "ANN": MLPClassifier,
    "XGBoost": XGBClassifier
}


optuna_df = pd.read_csv("data/optuna_tuning_results.csv")

os.makedirs("models", exist_ok=True)


for _, row in optuna_df.iterrows():
    resampling = row["Resampling"]
    model_name = row["Model"]
    params = eval(row["Best Params"])

    print(f"\nTraining and saving model: {resampling} + {model_name}...")


    model_class = model_classes[model_name]
    model = model_class(**params)
    model.fit(X_train, y_train)

    model_path = f"models/{resampling}_{model_name}_best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {model_path}")
