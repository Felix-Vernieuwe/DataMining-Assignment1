import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer


def label_encoding(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = LabelEncoder().fit_transform(df[column])
    return df


def hot_encoding(df):
    categorical_columns = df.select_dtypes(include=['object']).columns

    encoder = make_column_transformer(
        (OneHotEncoder(min_frequency=5, sparse_output=False), categorical_columns),
        remainder='passthrough'
    ).fit(df)
    encoded = encoder.transform(df)

    df = pd.DataFrame(encoded, columns=[label.split('__')[1] for label in encoder.get_feature_names_out()])
    return df
