from numpy import array, ndarray
from pandas import read_csv, DataFrame
import numpy as np
from model.dslabs_functions import get_variable_types

file_tag = "OriginalDataset"
target = "Question"
data = read_csv("datasets/" + file_tag + ".csv", sep=';')
labels: list = list(data[target].unique())
labels.sort()
print(f"Labels={labels}")

data = data.dropna()

aux_lst = list(data.columns)
symbolic_vars = []

variables_types: dict[str, list] = get_variable_types(data)

def to_str(x):
    if type(x) != float:
        return str(x)
    return x

for var in aux_lst:
    if var not in variables_types["numeric"]:
        data[var] = data[var].apply(to_str)
        symbolic_vars.append(var)

if target in symbolic_vars:
    symbolic_vars.remove(target)

data = data.drop(symbolic_vars, axis=1)

values: dict[str, list[int]] = {
    "Original": [
        #len(data[data[target] == '1']),
        #len(data[data[target] == '0']),
        #len(data[data[target] == '2']),
        #len(data[data[target] == '3']),
        #len(data[data[target] == '4']),
        len(data[data[target] == '<=50K']),
        len(data[data[target] == '>50K']),
    ]
}

y: array = data.pop(target).to_list()
X: ndarray = data.values

from pandas import concat
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split
from model.dslabs_functions import plot_multibar_chart


trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

train: DataFrame = concat(
    [DataFrame(trnX, columns=data.columns), DataFrame(trnY, columns=[target])], axis=1
)
train.to_csv(f"data/{file_tag}_train.csv", index=False)

test: DataFrame = concat(
    [DataFrame(tstX, columns=data.columns), DataFrame(tstY, columns=[target])], axis=1
)
test.to_csv(f"data/{file_tag}_test.csv", index=False)