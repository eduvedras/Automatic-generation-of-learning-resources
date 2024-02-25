import pandas as pd

dataset = pd.read_csv("datasets/Liver_Patient.csv",sep=",",decimal=".")

#dataset.drop(columns=["Unnamed: 0"], inplace=True)
#dataset.drop(columns=["b"], inplace=True)

#dataset.drop(index=dataset.index[:15000], axis=0, inplace=True)

#dataset.dropna(subset=["RainTomorrow"],inplace=True)

#print(list(dataset["Class"].unique()))
#dataset = dataset.drop(dataset[dataset['ReachedOnTime'] == 2].index)

#dataset['ReachedOnTime'] = dataset['ReachedOnTime'].replace(1,'Yes')
#dataset['ReachedOnTime'] = dataset['ReachedOnTime'].replace(0,'No')

print(list(dataset["Selector"].unique()))

#dataset.to_csv("datasets/LI-Small_Trans.csv", index=False)