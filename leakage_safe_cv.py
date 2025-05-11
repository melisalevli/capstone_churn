import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.pipeline import Pipeline  
from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from utils.preprocessing import load_and_preprocess

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN  


resamplers = {
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
    "ROS": RandomOverSampler(random_state=42),
    "RUS": RandomUnderSampler(random_state=42)
}


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}


X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/churn-bigml-80.csv", "data/churn-bigml-20.csv"
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for resample_name, resampler in resamplers.items():
    for model_name, model in models.items():
        print(f"\n Starting cross-validation for {resample_name} + {model_name}...")


        pipeline = Pipeline([
            ('resample', resampler),
            ('clf', model)
        ])

        
        f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1').mean()
        roc_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc').mean()

        
        pr_auc_scores = []
        for train_idx, test_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[test_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

            pipeline.fit(X_tr, y_tr)
            try:
                y_probs = pipeline.predict_proba(X_val)[:, 1]
            except AttributeError:
                y_probs = pipeline.decision_function(X_val)

            precision, recall, _ = precision_recall_curve(y_val, y_probs)
            pr_auc = auc(recall, precision)
            pr_auc_scores.append(pr_auc)

        pr_auc_mean = np.mean(pr_auc_scores)

        results.append({
            "Resampling": resample_name,
            "Model": model_name,
            "F1": round(f1, 3),
            "ROC-AUC": round(roc_auc, 3),
            "PR-AUC": round(pr_auc_mean, 3)
        })


results_df = pd.DataFrame(results)
os.makedirs("data", exist_ok=True)
results_df.to_csv("data/leakage_safe_cv_results.csv", index=False)
print("\n All combinations completed successfully!")
print(" Results saved to: data/leakage_safe_cv_results.csv")

