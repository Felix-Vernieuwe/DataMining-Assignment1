import pandas as pd
from preprocess import *

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, classification_report

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline


import random

train_data = pd.read_csv('./data/existing-customers.csv', delimiter=';')
classification_data = pd.read_csv('./data/potential-customers.csv', delimiter=';')


training_param = {
    'test_size': 0.2,
    'random_state': 42,
    'n_splits': 5,
}

hyperparam = {
    'max_depth': 5,
    'k': 2,
    'n_clusters': 2,
    'equal_proportions': True
}


models = [
    KMeans(n_clusters=hyperparam['n_clusters'], n_init=10, random_state=training_param['random_state']),
    KNeighborsClassifier(n_neighbors=hyperparam['k']),
    DecisionTreeClassifier(max_depth=hyperparam['max_depth'], random_state=training_param['random_state']),
    RandomForestClassifier(max_depth=hyperparam['max_depth'], random_state=training_param['random_state']),
    GaussianNB(),
]


available_features = list(train_data.columns)
excluded_features = ['RowID', 'class']
available_features = [feature for feature in available_features if feature not in excluded_features]
hyperparam['selected_labels'] = available_features
training_labels = LabelEncoder().fit_transform(train_data['class'])

train_data = train_data[available_features]

# Preprocessing stage
training_data = preprocess(train_data, hyperparam, training_param)



# Training stage
#   Split the data into training and test set using stratified k-fold cross validation
skf = StratifiedKFold(n_splits=training_param['n_splits'], shuffle=True, random_state=training_param['random_state'])



for classifier in models:
    scalar = StandardScaler()
    pipeline = Pipeline([
        ('transformer', scalar),
        ('classifier', classifier)
    ])

    y_pred = cross_val_predict(pipeline, training_data, training_labels, cv=skf)
    print(classifier.__class__.__name__)
    print(classification_report(training_labels, y_pred))
    print("\n\n")


# Selected features based on:
# 1. Correlation between features
# 2. Statistical Tests
# 3. Recursive Feature Elimination
# 4. Variance Threshold


