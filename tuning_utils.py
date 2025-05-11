import optuna
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def optimize_xgboost(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
    return score


def optimize_rf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
    }
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
    return score


def optimize_svm(trial, X, y):
    params = {
        'C': trial.suggest_float('C', 0.1, 10.0),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0),
        'kernel': 'rbf',
        'probability': True
    }
    model = SVC(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
    return score


def optimize_ann(trial, X, y):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64,), (64, 32), (128,)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1),
        'max_iter': 300
    }
    model = MLPClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
    return score


def optimize_logreg(trial, X, y):
    params = {
        'C': trial.suggest_float('C', 0.01, 10.0),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': trial.suggest_categorical('solver', ['liblinear'])
    }
    model = LogisticRegression(**params, random_state=42, max_iter=1000)
    score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
    return score


def optimize_dtree(trial, X, y):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
    }
    model = DecisionTreeClassifier(**params, random_state=42)
    score = cross_val_score(model, X, y, cv=3, scoring='f1').mean()
    return score
