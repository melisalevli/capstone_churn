import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(train_path, test_path):
    X_train, X_test, y_train, y_test = load_raw_data(train_path, test_path)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def load_raw_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    
    binary_cols = ['International plan', 'Voice mail plan']
    for col in binary_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])

    
    df_train = pd.get_dummies(df_train, columns=['State'], drop_first=True)
    df_test = pd.get_dummies(df_test, columns=['State'], drop_first=True)
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)

    
    X_train = df_train.drop('Churn', axis=1)
    y_train = df_train['Churn'].astype(int)

    X_test = df_test.drop('Churn', axis=1)
    y_test = df_test['Churn'].astype(int)

    return X_train, X_test, y_train, y_test
