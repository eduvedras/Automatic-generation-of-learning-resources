from numpy import array, ndarray
from pandas import read_csv, DataFrame
import numpy as np

file_tag = "Breast_Cancer"
target = "diagnosis"
data = read_csv("datasets/" + file_tag + ".csv", index_col='id', sep=',', decimal='.')
labels: list = list(data[target].unique())
labels.sort()
print(f"Labels={labels}")

values: dict[str, list[int]] = {
    "Original": [
        len(data[data[target] == 'M']),
        len(data[data[target] == 'B']),
        #len(data[data[target] == '3']),
    ]
}

y: array = data.pop(target).to_list()
X: ndarray = data.values

from pandas import concat
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split
from dslabs_functions import plot_multibar_chart


trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train: DataFrame = concat(
    [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])], axis=1
)
train.to_csv(f"data/{file_tag}_train.csv", index=False)

test: DataFrame = concat(
    [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])], axis=1
)
test.to_csv(f"data/{file_tag}_test.csv", index=False)