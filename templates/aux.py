import pandas as pd

dataset = pd.read_csv("datasets/smoking_driking.csv",sep=",",decimal=".")

#dataset.drop(columns=["Id"], inplace=True)

dataset.drop(index=dataset.index[:970000], axis=0, inplace=True)

dataset.to_csv("datasets/smoking_drinking.csv", index=False)