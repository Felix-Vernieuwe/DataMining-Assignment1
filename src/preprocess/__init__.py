from .data_cleaning import *
from .data_analysis import *
from .data_encoding import *
from .data_transformation import *

from sklearn.model_selection import train_test_split


def preprocess(data, hyperparam, training_param):
    # Step 1: Data Cleaning/Cleansing
    #   a. Filling in missing values
    data = use_sampling(data)

    #   b. Smooth noisy data
    data = binning(data)


    # Step 2: Data Transformation/Integration/Editing
    #    a. Feature Selection
    data = feature_selection(data, hyperparam['selected_labels'])

    #    b. Category Encoding
    data = hot_encoding(data)

    return data
