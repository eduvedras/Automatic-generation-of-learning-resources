from pandas import read_csv
import pandas as pd
from dslabs_functions import get_variable_types
import numpy as np

def get_variable_type(variable, variables_types):
    for key in variables_types:
        if variable in variables_types[key]:
            return key
    return None

def get_variables_description(data):
    variables = data.columns
    variables_description = {}
    for variable in variables:
        variables_description[variable] = {}
        variables_description[variable]['Type'] = get_variable_type(variable,get_variable_types(data))
        if variables_description[variable]['Type'] == 'numeric':
            variables_description[variable]['Range'] = [data[variable].min(), data[variable].max()]
        else:
            variables_description[variable]['Unique values'] = data[variable].unique()
    
    return variables_description

result = pd.DataFrame(columns=['Dataset', 'Description'])

# Iris
target = 'Species'

iris = read_csv('datasets/Iris.csv', index_col='Id', sep=',', decimal='.')

new_row = {'Dataset': 'Iris', 'Description': {'Records':len(iris),'Variables': get_variables_description(iris),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# Wine
target = 'Class'

wine = read_csv('datasets/Wine.csv', sep=',', decimal='.')

new_row = {'Dataset': 'Wine', 'Description': {'Records':len(wine),'Variables': get_variables_description(wine),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# Breast Cancer
target = 'diagnosis'

breast_cancer = read_csv('datasets/Breast_Cancer.csv', index_col='id', sep=',', decimal='.')

new_row = {'Dataset': 'Breast_Cancer', 'Description': {'Records':len(breast_cancer),'Variables': get_variables_description(breast_cancer),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False) 