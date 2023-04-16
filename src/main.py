import pandas as pd
from preprocess import *
import matplotlib.pyplot as plt
import random

# Preprocessing imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, train_test_split, cross_validate, cross_val_predict

# Learning Flow imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Evaluation imports
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Classifier imports
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

# Ensemble imports
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier




def calculate_profit(tp, fn, fp, tn):
    return (tp * 980 * 0.1) - (fp * 310 * 0.05) - (tp + fp) * 10


def total_profit_ratio(tp, fn, fp, tn):
    profit = calculate_profit(tp, fn, fp, tn)
    total = ((tp + fn) * 980 * 0.1) - ((tp + fn) * 10)
    return profit / total * 100


def scoring_function(y_true, y_pred):
    """Scoring function for classifiers"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return total_profit_ratio(tp, fn, fp, tn) / 100


def evaluation_report(classifier, scores):
    """Prints evaluation report for classifier"""
    print("-" * 80)
    print(classifier.__class__.__name__)

    print(f"\tAccuracy: {scores['test_accuracy'].mean():.2f} (+/- {scores['test_accuracy'].std() * 2:.2f})")
    print(f"\tPrecision: {scores['test_precision'].mean():.2f} (+/- {scores['test_precision'].std() * 2:.2f})")
    print(f"\tRecall: {scores['test_recall'].mean():.2f} (+/- {scores['test_recall'].std() * 2:.2f})")
    print(f"\tF1: {scores['test_f1'].mean():.2f} (+/- {scores['test_f1'].std() * 2:.2f})")
    print(f"\tProfit: {scores['test_profit'].mean():.2f} (+/- {scores['test_profit'].std() * 2:.2f})")
    print("-" * 80, '\n\n')


def testing_report(classifier, y_test, y_pred):
    """Prints testing report for classifier"""
    print("-" * 80)
    print(classifier.__class__.__name__)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"Total profit: {calculate_profit(tp, fn, fp, tn):.2f} ({total_profit_ratio(tp, fn, fp, tn):.2f}%)")
    print(f"""
    Selected {tp} out of {tp + fn} high-income customers
    {fp} low-income customers were incorrectly selected ({fp / tp * 100:.2f}%)
    {fn} high-income customers were not selected ({fn / (tp + fn) * 100:.2f}%)
    """)
    print("-" * 80, '\n\n')

    return total_profit_ratio(tp, fn, fp, tn)


def print_title(title, length=120):
    left_part = (length - len(title)) // 2
    right_part = length - len(title) - left_part
    print("=" * left_part, " " + title + " ", "=" * right_part)


train_data = pd.read_csv('./data/existing-customers.csv', delimiter=';')
classification_data = pd.read_csv('./data/potential-customers.csv', delimiter=';')

training_param = {
    'test_size': 0.2,
    'random_state': 42,
    'n_splits': 5,
    'k': 6,
    'equal_proportions': True,
    'n_clusters': 2,
    'n_features': 7,
}
hyperparam = {
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'max_depth': [2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'weights': ['uniform', 'distance'],
    'n_clusters': [2],
    'n_estimators': [5, 10, 20, 50],
    'n_neighbors': [5, 10, 20, 50],
}

kmeans_hyperparam = {
    'n_clusters': hyperparam['n_clusters'],
    'n_init': ['auto']
}

knn_hyperparam = {
    'n_neighbors': hyperparam['n_neighbors'],
    # 'algorithm': hyperparam['algorithm'],
    # 'weights': hyperparam['weights'],
}

dt_hyperparam = {
    'max_depth': hyperparam['max_depth'],
    'min_samples_split': hyperparam['min_samples_split'],
    'min_samples_leaf': hyperparam['min_samples_leaf'],
}

rf_hyperparam = {
    'n_estimators': hyperparam['n_estimators'],
    'max_depth': hyperparam['max_depth'],
    'min_samples_split': hyperparam['min_samples_split'],
    'min_samples_leaf': hyperparam['min_samples_leaf'],
}

sgd_hyperparam = {
    'alpha': [1e-3, 1e-2, 1e-1, 1],
    'max_iter': [1000, 2000, 5000],
}

models = [
    (KMeans(random_state=training_param['random_state']), kmeans_hyperparam),
    (KNeighborsClassifier(), knn_hyperparam),
    (DecisionTreeClassifier(random_state=training_param['random_state']), dt_hyperparam),
    (RandomForestClassifier(random_state=training_param['random_state']), rf_hyperparam),
    (GaussianNB(), {}),
    (SGDClassifier(random_state=training_param['random_state']), sgd_hyperparam),
]

available_features = list(train_data.columns)
excluded_features = ['RowID', 'class']
available_features = [feature for feature in available_features if feature not in excluded_features]
hyperparam['selected_labels'] = available_features
training_labels = LabelEncoder().fit_transform(train_data['class'])

train_data = train_data[available_features]
classification_data = classification_data[available_features]

numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns

categorical_features = train_data.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
        ]), numerical_features.values),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(min_frequency=5, sparse_output=False)),
        ]), categorical_features.values),
    ]
)

train_data = preprocessor.fit_transform(train_data)
classification_data = preprocessor.transform(classification_data)

# Split the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(train_data, training_labels, test_size=training_param['test_size'],
                                                    random_state=training_param['random_state'])

#   Split the data into training and test set using stratified k-fold cross validation
if training_param['equal_proportions']:
    cv = StratifiedKFold(n_splits=training_param['n_splits'], shuffle=True, random_state=training_param['random_state'])
else:
    cv = KFold(n_splits=training_param['n_splits'], shuffle=True, random_state=training_param['random_state'])

performances = {model: {} for model, _ in models}

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'profit': make_scorer(scoring_function),
}






print_title("HYPERPARAMETER-TUNING PHASE")
for classifier, param_grid in models:
    if param_grid:
        pipeline = Pipeline([
            ('transformer', StandardScaler()),
            ('classifier',
             GridSearchCV(classifier, param_grid, scoring=scoring, refit='profit', cv=cv, return_train_score=True,
                          n_jobs=-1))
        ])

        pipeline.fit(X_train, y_train)
        scores = pipeline['classifier'].cv_results_
        performances[classifier] = {
            'params': pipeline['classifier'].best_params_,
        }
        print(classifier.__class__.__name__)
        print(f"\tNew hyper-parameters: {pipeline['classifier'].best_params_}")
print("=" * 120, '\n\n')







print_title("TRAINING PHASE")
for classifier, _ in models:
    if 'params' in performances[classifier]:
        classifier.set_params(**performances[classifier]['params'])
    scores = cross_validate(classifier, X_train, y_train, scoring=scoring, cv=cv, return_train_score=True, n_jobs=-1)
    performances[classifier]['performance'] = np.mean(scores['test_profit'])
    evaluation_report(classifier, scores)
print("=" * 120, '\n\n')

top_performing = sorted(performances.items(), key=lambda x: x[1]['performance'], reverse=True)[:3]
print(f"Top performing model: {top_performing[0][0]} (performance: {top_performing[0][1]['performance']}")
top_models = [model for model, _ in top_performing]
# Set hyperparameters for the ensemble models
for model in top_models:
    if 'params' in performances[model]:
        model.set_params(**performances[model]['params'])
top_models = [(model.__class__.__name__, model) for model in top_models]

ensemble_models = [
    VotingClassifier(estimators=top_models, voting='hard'),
    BaggingClassifier(estimator=top_models[0][1], n_estimators=10, random_state=training_param['random_state']),
    AdaBoostClassifier(estimator=top_models[0][1], n_estimators=10, random_state=training_param['random_state']),
]

for classifier in ensemble_models:
    scalar = StandardScaler()
    pipeline = Pipeline([
        ('transformer', scalar),
        ('classifier', classifier)
    ])

    scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=cv, return_train_score=True, n_jobs=-1)
    performances[classifier] = {
        'performance': scores['test_profit'].mean(),
        'estimator': classifier,
    }
    evaluation_report(classifier, scores)
print("=" * 120, "\n\n")







print_title("EVALUATION PHASE")

top_performer = (None, None)
for classifier, data in performances.items():
    if 'params' in data:
        classifier.set_params(**data['params'])
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    score = testing_report(classifier, y_test, y_pred)
    if top_performer[1] is None or score > top_performer[1]:
        top_performer = (classifier, score)

print("=" * 120, "\n\n")


print(f"Top performing model: {top_performer[0]} (performance: {top_performer[1]:.2f}%)")
print(f"Best possible performance: {calculate_profit(np.unique(y_test, return_counts=True)[1][1],0,0,0)}")
print("\n\n")






print_title("PREDICTION PHASE")

y_pred = top_performer[0].predict(classification_data)
print(f"Predicted profit: {calculate_profit(np.unique(y_pred, return_counts=True)[1][1],0,0,0) * top_performer[1] / 100}")
print(f"Client list: {np.where(y_pred == 1)[0]}")

np.savetxt("./output/client_list.txt", np.where(y_pred == 1)[0], fmt='%d')

print(f"Selected {np.unique(y_pred, return_counts=True)[1][1]} clients out of {len(y_pred)}")

print("="*120)