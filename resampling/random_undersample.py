from imblearn.under_sampling import RandomUnderSampler

def apply_random_undersampler(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled
