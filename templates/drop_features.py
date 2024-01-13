from pandas import DataFrame, Index, read_csv
from matplotlib.pyplot import figure, savefig, show
from matplotlib.pyplot import gca
from dslabs_functions import (
    select_low_variance_variables,
    study_variance_for_feature_selection,
    apply_feature_selection,
    select_redundant_variables,
    study_redundancy_for_feature_selection,
    plot_multiline_chart, HEIGHT
)

target = "diagnosis"
file_tag = "Breast_Cancer"
train: DataFrame = read_csv(f"datasets/{file_tag}.csv",index_col='id',sep=",",decimal=".")


def select_low_variance_variables(
    data: DataFrame, max_threshold: float, target: str = "class"
) -> list:
    summary5: DataFrame = data.describe()
    vars2drop: Index[str] = summary5.columns[
        summary5.loc["std"] * summary5.loc["std"] < max_threshold
    ]
    vars2drop = vars2drop.drop(target) if target in vars2drop else vars2drop
    return list(vars2drop.values)




print("Original variables", train.columns.to_list())
vars2drop1: list[str] = select_low_variance_variables(train, 3, target=target)


from pandas import Series


def select_redundant_variables(
    data: DataFrame, min_threshold: float = 0.90, target: str = "class"
) -> list:
    df: DataFrame = data.drop(target, axis=1, inplace=False)
    corr_matrix: DataFrame = abs(df.corr())
    variables: Index[str] = corr_matrix.columns
    vars2drop: list = []
    for v1 in variables:
        vars_corr: Series = (corr_matrix[v1]).loc[corr_matrix[v1] >= min_threshold]
        vars_corr.drop(v1, inplace=True)
        if len(vars_corr) > 1:
            lst_corr = list(vars_corr.index)
            for v2 in lst_corr:
                if v2 not in vars2drop:
                    vars2drop.append(v2)
    return vars2drop


print("Original variables", train.columns.values)
vars2drop2: list[str] = select_redundant_variables(
    train, target=target, min_threshold=0.5
)

vars2drop = []
possible_drops = []
for v in vars2drop2:
    if v in vars2drop1:
        vars2drop.append(v)
    else:
        possible_drops.append(v)
        
for v in vars2drop1:
    if v not in vars2drop:
        possible_drops.append(v)
        
print("Variables to drop", vars2drop)
print("Number of variables to drop", len(vars2drop))

dataset: DataFrame = train.drop(vars2drop, axis=1, inplace=False)

from sklearn.decomposition import PCA
from pandas import Series, Index
from matplotlib.axes import Axes
from dslabs_functions import plot_bar_chart

import itertools
def get_combinations(lst): # creating a user-defined method
    combination = [] # empty list 
    combination.extend(itertools.combinations(lst, 3))
    return combination

all_combinations = get_combinations(possible_drops) # method call
all_combinations2 = []
for comb in all_combinations:
    lst = []
    lst.append(comb[0])
    lst.append(comb[1])
    lst.append(comb[2])
    all_combinations2.append(lst)
#print(all_combinations2)

'''
for comb in all_combinations2:
    aux = dataset.copy()
    aux = aux.drop(comb, axis=1)
    target_data: Series = aux.pop(target)
    index: Index = aux.index
    pca = PCA()
    pca.fit(aux)

    xvalues: list[str] = [f"PC{i+1}" for i in range(len(pca.components_))]
    figure()
    ax: Axes = gca()
    plot_bar_chart(
        xvalues,
        pca.explained_variance_ratio_,
        ax=ax,
        title="Explained variance ratio",
        xlabel="PC",
        ylabel="ratio",
        percentage=True,
    )
    ax.plot(pca.explained_variance_ratio_)
    savefig(f"{file_tag}_pca_{comb[0]}_{comb[1]}_{comb[2]}.png")
    show()'''
'''    
aux = dataset.copy()
aux = aux.drop(['area_se','perimeter_se','smoothness_se'], axis=1)

target_data: Series = aux.pop(target)
index: Index = aux.index
pca = PCA()
pca.fit(aux)


xvalues: list[str] = [f"PC{i+1}" for i in range(len(pca.components_))]
figure()
ax: Axes = gca()
plot_bar_chart(
    xvalues,
    pca.explained_variance_ratio_,
    ax=ax,
    title="Explained variance ratio",
    xlabel="PC",
    ylabel="ratio",
    percentage=True,
)
ax.plot(pca.explained_variance_ratio_)
savefig(f"{file_tag}_pca.png")
show()'''

dataset = dataset.drop(['area_mean','area_worst','radius_mean'], axis=1)

dataset.to_csv(f"datasets/{file_tag}_reduced.csv", sep=",", decimal=".", index=True)