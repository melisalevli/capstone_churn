from imblearn.combine import SMOTEENN

def apply_smote_enn(X, y):
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled
