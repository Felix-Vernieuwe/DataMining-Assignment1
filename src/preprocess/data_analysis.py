import pandas as pd
import matplotlib.pyplot as plt
import os


def analyse_data(dataset: pd.DataFrame):
    print(f"Amount of rows: {len(dataset)}")
    print(f"Rows with missing values: {dataset.isnull().any(axis=1).sum()}")
    if 'class' in dataset:
        print(
            f"Rows with income >50K: {dataset[dataset['class'] == '>50K'].shape[0]} ({dataset[dataset['class'] == '>50K'].shape[0] / len(dataset) * 100:.2f}%)")

    is_train = 'class' in dataset

    # For each column, get the distribution of the values
    color = 'tab:blue' if is_train else 'tab:red'

    for column in dataset:
        if column == 'RowID':
            continue

        # If categorical, plot a bar chart, else a histogram
        if dataset[column].dtype == 'object':
            dataset[column].value_counts().plot.bar()
        else:
            dataset[column].plot.hist()
        plt.xticks(rotation=45)
        plt.gcf().subplots_adjust(bottom=0.25)

        for rect in plt.gca().patches:
            rect.set_color(color)

        plt.title(column)

        # if not os.path.exists('./output'):
        #     os.makedirs('./output')
        # plt.savefig(f"./output/{'train' if is_train else 'classify'}_{column}.png")
        plt.show()
