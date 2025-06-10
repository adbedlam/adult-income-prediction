import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


def read_n_preproc():
    df = pd.read_csv('../data/adult.csv')

    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)

    df['income'] = df['income'].apply(lambda x: 1 if x == ">50K" else 0)

    new_df = df.copy()

    target = new_df['income']
    new_df.drop('income', axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(new_df, target, test_size=0.2, random_state=42, stratify=target,
                                                        shuffle=True)

    cat_cols = new_df.select_dtypes(include='object').columns
    num_cols = new_df.select_dtypes(include=['int64', 'float64']).columns

    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(X_train[num_cols])
    X_test_num = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    X_train_cat = ohe.fit_transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])

    dump(ohe, '../models/onehot_encoder.joblib')

    dump(scaler, '../models/standard_scaler.joblib')

    X_train_final = np.hstack((X_train_num, X_train_cat))
    X_test_final = np.hstack((X_test_num, X_test_cat))

    smote = SMOTE(random_state=42)

    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_final, y_train)

    return X_train_balanced, X_test_final, y_train_balanced, y_test


def inference_preproc(inp_data):
    data = inp_data.split(',')
    ohe = load('../models/onehot_encoder.joblib')
    scaler = load('../models/standard_scaler.joblib')

    feature_names = [
        "age", "workclass", "fnlwgt", "education", "education.num", "marital.status",
        "occupation", "relationship", "race", "sex", "capital.gain",
        "capital.loss", "hours.per.week", "native.country"
    ]

    df = pd.DataFrame([data], columns=feature_names)

    df.replace('?', pd.NA, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    X_infer_num = scaler.transform(df[num_cols])
    X_infer_cat = ohe.transform(df[cat_cols])

    X_train_final = np.hstack((X_infer_num, X_infer_cat))

    names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))


    return X_train_final, names
