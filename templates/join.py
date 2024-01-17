from pandas import read_csv
import pandas as pd

import os
 
# assign directory
directory = 'Questions_Chart'

final_dataset = pd.DataFrame(columns=['Question', 'Charts_id'])

for filename in os.scandir(directory):
    if filename.is_file():
        print(filename.path)
        dataset = read_csv(filename.path, sep=';')
        
        #if filename.name == 'adult.csv':
            #Nothing to do
        if filename.name == 'BankNoteAuthentication_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if 'overfitting' in row['Charts_id'] or '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index) #Podemos deixar DT e dt_acc_rec caso não haja o suficiente
        
        if filename.name == 'BreastCancer_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
        
        if filename.name == 'Churn_Modelling_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
                    
        if filename.name == 'diabetes_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
        
        if filename.name == 'heart_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
                    
        if filename.name == 'Iris_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
        
        #if filename.name == 'Titanic_Questions_Chart.csv':
        
        if filename.name == 'vehicle_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
                    
        if filename.name == 'Wine_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if 'overfitting_gb' in row['Charts_id'] or 'overfitting_rf' in row['Charts_id'] or '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
        
        if filename.name == 'WineQT_Questions_Chart.csv':
            for index, row in dataset.iterrows():
                if '_mv' in row['Charts_id']:
                    dataset = dataset.drop(index)
                    
        final_dataset = pd.concat([final_dataset, dataset])
   
final_dataset.to_csv('Final.csv', sep=';', index=False)
