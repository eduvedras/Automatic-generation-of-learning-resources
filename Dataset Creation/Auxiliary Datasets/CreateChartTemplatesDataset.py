import pandas as pd
from pandas import DataFrame, Series, to_datetime, to_numeric
import os

def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": [], "date": [], "symbolic": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                    variable_types["symbolic"].append(c)

    return variable_types


data = pd.read_csv('metadata1.csv')

new_data = pd.DataFrame(columns=['Chart', 'description'])

images_aux = []

for index, row in data.iterrows():
    if row['Chart'] not in images_aux and 'adult' in row['Chart']:
        images_aux.append(row['Chart'])

images = []
for image in images_aux:
    images.append(image[5:])

directory = '/home/eduvedras/tese/templates/datasets'

variable_types = {}

classes = {'adult': ['income'],'BankNoteAuthentication': ['class'],
           'Breast_Cancer': ['diagnosis'], 'Churn_Modelling': ['Exited'],
           'diabetes':['Outcome'], 'heart': ['target'], 'Iris': ['Species'],
           'Titanic': ['Survived'], 'Wine': ['Class'], 'WineQT': ['quality'],
           'vehicle': ['target'], 'apple_quality': ['Quality'], 'loan_data': ['Loan_Status'],
           'credit_customers': ['class'], 'smoking_drinking': ['DRK_YN'], 'sky_survey': ['class'],
           'weatherAUS': ['RainTomorrow'], 'Dry_Bean_Dataset': ['Class'],'abalone': ['Sex'],
           'car_insurance': ['is_claim'], 'Covid_Data': ['CLASSIFICATION'],'customer_segmentation': ['Segmentation'],
           'detect_dataset': ['Output'],'e-commerce': ['ReachedOnTime'], 'Employee': ['LeaveOrNot'],
           'Hotel_Reservations': ['booking_status'], 'Liver_Patient': ['Selector'], 'maintenance': ['Machine_failure'],
           'ObesityDataSet': ['NObeyesdad'], 'phone': ['price_range'], 'Placement': ['status'],
           'StressLevelDataset': ['stress_level'], 'urinalysis_tests': ['Diagnosis'], 'water_potability': ['Potability']}

vars = {}
for filename in os.scandir(directory):
    if filename.is_file():
        dataset = pd.read_csv(filename.path)
        file_tag = filename.name[:-4]
        variable_types[file_tag] = get_variable_types(dataset)
        
        #all_vars = []
        #for c in dataset.columns:
        #   if c not in classes[file_tag]:
        #       all_vars.append(c)
                
        numeric_vars = []
        for c in variable_types[file_tag]['numeric']:
            if c not in classes[file_tag]:
                numeric_vars.append(c)
                
        symbolic_vars = []
        for c in variable_types[file_tag]['symbolic']:
            if c not in classes[file_tag]:
                symbolic_vars.append(c)
        
        for c in variable_types[file_tag]['binary']:
            if c not in classes[file_tag]:
                symbolic_vars.append(c)
        
        mv = []
        for var in dataset.columns:
            nr: int = dataset[var].isna().sum()
            if nr > 0:
                mv.append(var)
                
        vars[file_tag] = {'target': classes[file_tag][0], 
                          'binary': classes[file_tag][0] in variable_types[file_tag]['binary'],
                          'numeric_vars': numeric_vars, 
                          'symbolic_vars': symbolic_vars,
                          'missing_values': mv}


for filename in os.scandir(directory):
    if filename.is_file():
        file_tag = filename.name[:-4]
        for image in images:
            new_row = {'Chart': file_tag + image, 'description': ''}
            if 'overfitting_decision_tree' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of a decision tree where the y-axis represents the accuracy and the x-axis represents the max depth ranging from 2 to 25.'
            elif 'decision_tree' in image:
                new_row['description'] = 'An image showing a decision tree with depth = 2 where the first decision is made with the condition [] and the second with the condition [].'
            elif 'overfitting_mlp' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of a mlp where the y-axis represents the accuracy and the x-axis represents the number of iterations ranging from 100 to 1000.'
            elif 'overfitting_gb' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of gradient boosting where the y-axis represents the accuracy and the x-axis represents the number of estimators ranging from 2 to 2002.'
            elif 'overfitting_rf' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of random forest where the y-axis represents the accuracy and the x-axis represents the number of estimators ranging from 2 to 2002.'
            elif 'overfitting_knn' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of k-nearest neighbors where the y-axis represents the accuracy and the x-axis represents the number of neighbors ranging from 1 to 23.'
            elif 'overfitting_dt_acc_rec' in image:
                if vars[file_tag]['binary'] == False:
                    continue
                new_row['description'] = 'A multi-line chart showing the overfitting of decision tree where the y-axis represents the performance of both accuracy and recall and the x-axis represents the max depth ranging from 2 to 25.'
            elif 'pca' in image:
                new_row['description'] = 'A bar chart showing the explained variance ratio of [] principal components.'
            elif 'correlation_heatmap' in image:
                new_row['description'] = 'A heatmap showing the correlation between the variables of the dataset. The variables are [].'
            elif 'boxplots' in image:
                new_row['description'] = 'A set of boxplots of the variables [].'
            elif 'histograms_numeric' in image:
                new_row['description'] = 'A set of histograms of the variables [].'
            elif 'histograms_symbolic' in image:
                if len(vars[file_tag]["symbolic_vars"]) == 0:
                    continue
                new_row['description'] = 'A set of bar charts of the variables [].'
            elif 'mv' in image:
                if len(vars[file_tag]["missing_values"]) == 0:
                    continue
                new_row['description'] = 'A bar chart showing the number of missing values per variable of the dataset. The variables that have missing values are: [].'
            elif 'class_histogram' in image:
                new_row['description'] = 'A bar chart showing the distribution of the target variable [].'
            elif 'nr_records_nr_variables' in image:
                new_row['description'] = 'A bar chart showing the number of records and variables of the dataset.'
            #elif 'scatter-plots' in image:
            #    new_row['description'] = f'A set of scatter plots of the variables {vars[file_tag]["symbolic_vars"]}.'
            if 'scatter-plots' in image:
                continue
            new_data.loc[len(new_data)] = new_row

new_data.to_csv('desc_dataset.csv', sep =";", index=False)