import pandas as pd

dataset = pd.read_csv("datasets/weatherAUS.csv",sep=",",decimal=".")

dataset.drop(columns=['MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Temp9am'], inplace=True)

#dataset.drop(columns=["MajorAxisLength"], inplace=True)

#dataset.drop(index=dataset.index[:15000], axis=0, inplace=True)

#dataset.dropna(subset=["RainTomorrow"],inplace=True)

#print(list(dataset["Class"].unique()))
#dataset = dataset.drop(dataset[dataset['ReachedOnTime'] == 2].index)

#dataset['ReachedOnTime'] = dataset['ReachedOnTime'].replace(1,'Yes')
#dataset['ReachedOnTime'] = dataset['ReachedOnTime'].replace(0,'No')

print(list(dataset["RainTomorrow"].unique()))

dataset.to_csv("datasets/weatherAUS1.csv", index=False)