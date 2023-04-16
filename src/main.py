import pandas as pd
from preprocess import *

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline


import random


def calculate_profit(tp, fn, fp, tn):
    return (tp * 980 * 0.1) - (fp * 310 * 0.05) - (tp + fp) * 10


def total_profit_ratio(tp, fn, fp, tn):
    return calculate_profit(tp, fn, fp, tn) / ((tp + fn) * 970 * 0.1) * 100


# Loss function: closeness to the optimal profit
def scoring_function(y_true, y_pred):
    tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
    return total_profit_ratio(tp, fn, fp, tn) / 100

def evaluation_report(classifier, confusion, scores):
    print("="*80)
    print(classifier.__class__.__name__)


    print(f"\tAccuracy: {scores['test_accuracy'].mean():.2f} (+/- {scores['test_accuracy'].std() * 2:.2f})")
    print(f"\tPrecision: {scores['test_precision'].mean():.2f} (+/- {scores['test_precision'].std() * 2:.2f})")
    print(f"\tRecall: {scores['test_recall'].mean():.2f} (+/- {scores['test_recall'].std() * 2:.2f})")
    print(f"\tF1: {scores['test_f1'].mean():.2f} (+/- {scores['test_f1'].std() * 2:.2f})")
    print(f"\tProfit: {scores['test_profit'].mean():.2f} (+/- {scores['test_profit'].std() * 2:.2f})")

    tp, fn, fp, tn = confusion.ravel()

    print("\n")

    print(f"Total profit: {calculate_profit(tp, fn, fp, tn):.2f} ({scores['test_profit'].mean()*100:.2f})")
    print(f"Selecting {tp + fp} out of {tp + fn + fp + tn} customers (with {fp} low-profit customers, and {fn} forgotten customers)")
    # print(f"Confusion matrix:\n{confusion}")
    print("="*80, '\n\n')



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


performances = {model: {} for model in models}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'profit': make_scorer(scoring_function),
}

for classifier in models:
    scalar = StandardScaler()
    pipeline = Pipeline([
        ('transformer', scalar),
        ('classifier', classifier),
    ])

    y_pred = cross_val_predict(pipeline, training_data, training_labels, cv=skf)
    scores = cross_validate(pipeline, training_data, training_labels, cv=skf, scoring=scoring)
    performances[classifier]['performance'] = scores['test_profit'].mean()

    evaluation_report(classifier, confusion_matrix(training_labels, y_pred), scores)

top_performing = sorted(performances.items(), key=lambda x: x[1]['performance'], reverse=True)[:3]
top_performing = [(model.__class__.__name__, model) for model, _ in top_performing]

ensemble_models = [
    VotingClassifier(estimators=top_performing, voting='hard'),
    BaggingClassifier(estimator=top_performing[0][1], n_estimators=10, random_state=training_param['random_state']),
    AdaBoostClassifier(estimator=top_performing[0][1], n_estimators=10, random_state=training_param['random_state']),
]

for classifier in ensemble_models:
    scalar = StandardScaler()
    pipeline = Pipeline([
        ('transformer', scalar),
        ('classifier', classifier)
    ])

    # Use model_selection.cross_val_score to evaluate the model
    y_pred = cross_val_predict(pipeline, training_data, training_labels, cv=skf)
    scores = cross_validate(pipeline, training_data, training_labels, cv=skf, scoring=scoring)
    performances[classifier] = {'performance': scores['test_profit'].mean()}

    evaluation_report(classifier, confusion_matrix(training_labels, y_pred), scores)


top_classifier = sorted(performances.items(), key=lambda x: x[1]['performance'], reverse=True)[0][0]
classification_data = preprocess(classification_data, hyperparam, training_param)

# pipeline = Pipeline([
#     ('transformer', StandardScaler()),
#     ('classifier', top_classifier),
#     ('scorer', make_scorer(calculate_profit))
# ])
#
# pipeline.fit(training_data, training_labels)
# print(pipeline.score(training_data, training_labels))
# print(predictions)
#






# Selected features based on:
# 1. Correlation between features
# 2. Statistical Tests
# 3. Recursive Feature Elimination
# 4. Variance Threshold


