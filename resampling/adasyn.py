from imblearn.over_sampling import ADASYN

def apply_adasyn(X, y):
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled
