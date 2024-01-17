import pandas as pd
'''
dataset = pd.read_csv('datasets/wine.data')

dataset.columns = ["Class","Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
                     "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines",
                        "Proline"]

dataset.to_csv('Wine.csv', sep=',', index=False)'''

data = pd.read_csv('datasets/adult.csv', sep=',', decimal='.')

data = data.drop(['native-country'], axis=1)

data.to_csv('datasets/adult1.csv', sep=',', index=False)

