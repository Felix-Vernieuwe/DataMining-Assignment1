import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


# + Preserve shape of the original distribution
def minmax_scale(df):
    for column in df.columns:
        if df[column].dtype == 'float64':
            scaler = MinMaxScaler()
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return df


# + Each feature is centered around 0 and has a variance of 1 (normal distribution)
def standard_scale(df):
    for column in df.columns:
        if df[column].dtype == 'float64':
            scaler = StandardScaler()
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return df


# + Reduces the impact of outliers
def robust_scale(df):
    for column in df.columns:
        if df[column].dtype == 'float64':
            scaler = RobustScaler()
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return df


def feature_selection(df, selected_labels):
    return df[selected_labels]
