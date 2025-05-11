import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from utils.preprocessing import load_raw_data  


X_train_df, _, y_train, _ = load_raw_data(
    "data/churn-bigml-80.csv", "data/churn-bigml-20.csv"
)
feature_names = X_train_df.columns


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}


importance_results = []


for model_name, model in models.items():
    print(f"\nTraining {model_name} with SMOTE...")

    steps = []
    if isinstance(model, LogisticRegression):
        steps.append(('scaler', StandardScaler()))

    steps += [
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ]

    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    if model_name == "Logistic Regression":
        importances = abs(pipeline.named_steps['clf'].coef_[0])
    else:
        importances = pipeline.named_steps['clf'].feature_importances_

    model_df = pd.DataFrame({
        'Model': model_name,
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    importance_results.append(model_df)

    print(model_df.head(10).to_string(index=False))


all_importances_df = pd.concat(importance_results)
all_importances_df.to_csv("data/feature_importances_SMOTE.csv", index=False)
print("\nFeature importances saved to: data/feature_importances_SMOTE.csv")
