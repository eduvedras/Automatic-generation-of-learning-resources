from pandas import read_csv
import pandas as pd
from model.dslabs_functions import get_variable_types
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

# Titanic
target = 'Survived'

titanic = read_csv('datasets/Titanic.csv', index_col='PassengerId', sep=',', decimal='.')

new_row = {'Dataset': 'Titanic', 'Description': {'Records':len(titanic),'Variables': get_variables_description(titanic),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# Heart
target = 'target'

heart = read_csv('datasets/heart.csv', sep = ',', decimal='.')

new_row = {'Dataset': 'heart', 'Description': {'Records':len(heart),'Variables': get_variables_description(heart),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# Diabetes
target = 'Outcome'

diabetes = read_csv('datasets/diabetes.csv', sep = ',', decimal='.')

new_row = {'Dataset': 'diabetes', 'Description': {'Records':len(diabetes),'Variables': get_variables_description(diabetes),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# WineQT
target = 'quality'

wineQT = read_csv('datasets/WineQT.csv', index_col='Id', sep = ',', decimal='.')

new_row = {'Dataset': 'WineQT', 'Description': {'Records':len(wineQT),'Variables': get_variables_description(wineQT),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# Adult
target = 'income'

adult = read_csv('datasets/adult.csv', sep = ',', decimal='.', na_values='?')

new_row = {'Dataset': 'adult', 'Description': {'Records':len(adult),'Variables': get_variables_description(adult),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# Churn_Modelling
target = 'Exited'

churn_modelling = read_csv('datasets/Churn_Modelling.csv', index_col='CustomerId', sep = ',', decimal='.')

new_row = {'Dataset': 'Churn_Modelling', 'Description': {'Records':len(churn_modelling),'Variables': get_variables_description(churn_modelling),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# BankNoteAuthentication
target = 'class'

bank_note_authentication = read_csv('datasets/BankNoteAuthentication.csv', sep = ',', decimal='.')

new_row = {'Dataset': 'BankNoteAuthentication', 'Description': {'Records':len(bank_note_authentication),'Variables': get_variables_description(bank_note_authentication),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)

# Vehicle
target = 'target'

vehicle = read_csv('datasets/vehicle.csv', sep = ',', decimal='.')

new_row = {'Dataset': 'vehicle', 'Description': {'Records':len(vehicle),'Variables': get_variables_description(vehicle),'Class': target}}
result.loc[len(result)] = new_row

result.to_csv('descriptions.csv', sep=';', index=False)