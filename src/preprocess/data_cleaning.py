from sklearn.linear_model import LinearRegression
import numpy as np


def clear_empty_rows(df):
    # Remove rows with empty values
    df = df.dropna()
    return df


def use_mean(df):
    # Replace empty values with the mean of the column, if the column is numeric,
    # or with the mode of the column, if the column is categorical
    for column in df.columns:
        if df[column].dtype == 'float64':
            mean = df[column].mean()
            df[column] = df[column].fillna(mean)
        else:
            df[column] = df[column].fillna(df[column].value_counts().index[0])
    return df


def use_forward_fill(df):
    # Replace empty values with the value of the previous row
    return df.fillna(method='ffill')


def use_backward_fill(df):
    # Replace empty values with the value of the next row
    return df.fillna(method='bfill')


def use_sampling(df):
    # Replace empty values with a random value from the column
    for column in df.columns:
        if df[column].dtype == 'float64':
            # Turn df[column] numeric values into distribution of buckets
            # Then use np.random.choice to choose a random value from the buckets
            # Then fill the empty values with the random value
            df[column] = pd.cut(df[column], 10)
            distribution = df[column].value_counts(normalize=True)
            df[column] = df[column].fillna(np.random.choice(distribution.index, p=distribution.values))
        else:
            distribution = df[column].value_counts(normalize=True)
            df[column] = df[column].fillna(np.random.choice(distribution.index, p=distribution.values))
    return df


def binning(df, bucket_size=10):
    # Turn numeric values into buckets
    for column in df.columns:
        if df[column].dtype == 'float64':
            # Apply quantile bucketing
            df[column] = pd.qcut(df[column], bucket_size)
    return df
