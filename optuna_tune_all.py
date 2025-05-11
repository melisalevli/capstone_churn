import pandas as pd
import optuna
import os
from utils.preprocessing import load_and_preprocess
from resampling.smote import apply_smote
from resampling.smote_enn import apply_smote_enn
from resampling.adasyn import apply_adasyn
from resampling.random_oversample import apply_random_oversampler
from resampling.random_undersample import apply_random_undersampler
from tuning_utils import (
    optimize_xgboost, optimize_rf, optimize_ann,
    optimize_svm, optimize_logreg, optimize_dtree
)


resampling_methods = {
    "SMOTE": apply_smote,
    "SMOTEENN": apply_smote_enn,
    "ADASYN": apply_adasyn,
    "ROS": apply_random_oversampler,
    "RUS": apply_random_undersampler
}


models = {
    "XGBoost": optimize_xgboost,
    "RandomForest": optimize_rf,
    "ANN": optimize_ann,
    "SVM": optimize_svm,
    "LogisticRegression": optimize_logreg,
    "DecisionTree": optimize_dtree
}

results = []


X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/churn-bigml-80.csv", "data/churn-bigml-20.csv"
)


for resample_name, resample_func in resampling_methods.items():
    X_res, y_res = resample_func(X_train, y_train)

    for model_name, objective_func in models.items():
        print(f"\n {resample_name} + {model_name} tuning başlatılıyor...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_func(trial, X_res, y_res), n_trials=30, show_progress_bar=True)

        best_score = study.best_value
        best_params = study.best_params

        results.append({
            "Resampling": resample_name,
            "Model": model_name,
            "Best F1 Score": round(best_score, 4),
            "Best Params": best_params
        })


results_df = pd.DataFrame(results)
os.makedirs("data", exist_ok=True)
results_df.to_json("data/optuna_tuning_results.json", orient="records", lines=True)
results_df.to_csv("data/optuna_tuning_results.csv", index=False)

print("\nOptuna tuning completed. Results saved to data folder.")