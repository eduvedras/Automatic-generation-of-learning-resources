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
                first_var = ''
                second_var = ''
                if file_tag == 'adult':
                    first_var = 'hours-per-week'
                    second_var = 'capital-loss'
                if file_tag == 'BankNoteAuthentication':
                    first_var = 'skewness'
                    second_var = 'curtosis'
                if file_tag == 'Breast_Cancer':
                    first_var = 'perimeter_mean'
                    second_var = 'texture_worst'
                if file_tag == 'Churn_Modelling':
                    first_var = 'Age'
                    second_var = 'NumOfProducts'
                if file_tag == 'diabetes':
                    first_var = 'BMI'
                    second_var = 'Age'
                if file_tag == 'heart':
                    first_var = 'slope'
                    second_var = 'restecg'
                if file_tag == 'Titanic':
                    first_var = 'Pclass'
                    second_var = 'Parch'
                if file_tag == 'vehicle':
                    first_var = 'MAJORSKEWNESS'
                    second_var = 'CIRCULARITY'
                if file_tag == 'Wine':
                    first_var = 'Total phenols'
                    second_var = 'Proanthocyanins'
                if file_tag == 'WineQT':
                    first_var = 'density'
                    second_var = 'chlorides'
                if file_tag == 'apple_quality':
                    first_var = 'Juiciness'
                    second_var = 'Crunchiness'
                if file_tag == 'loan_data':
                    first_var = 'Loan_Amount_Term'
                    second_var = 'ApplicantIncome'
                if file_tag == 'credit_customers':
                    first_var = 'existing_credits'
                    second_var = 'residence_since'
                if file_tag == 'smoking_drinking':
                    first_var = 'SMK_stat_type_cd'
                    second_var = 'gamma_GTP'
                if file_tag == 'sky_survey':
                    first_var = 'dec'
                    second_var = 'mjd'
                if file_tag == 'weatherAUS':
                    first_var = 'Rainfall'
                    second_var = 'Pressure3pm'
                if file_tag == 'Dry_Bean_Dataset':
                    first_var = 'Area'
                    second_var = 'AspectRation'
                if file_tag == 'abalone':
                    first_var = 'Height'
                    second_var = 'Diameter'
                if file_tag == 'car_insurance':
                    first_var = 'displacement'
                    second_var = 'height'
                if file_tag == 'Covid_Data':
                    first_var = 'CARDIOVASCULAR'
                    second_var = 'ASHTMA'
                if file_tag == 'customer_segmentation':
                    first_var = 'Family_Size'
                    second_var = 'Work_Experience'
                if file_tag == 'detect_dataset':
                    first_var = 'Ic'
                    second_var = 'Vb'
                if file_tag == 'e-commerce':
                    first_var = 'Prior_purchases'
                    second_var = 'Customer_care_calls'
                if file_tag == 'Employee':
                    first_var = 'JoiningYear'
                    second_var = 'ExperienceInCurrentDomain'
                if file_tag == 'Hotel_Reservations':
                    first_var = 'lead_time'
                    second_var = 'no_of_special_requests'
                if file_tag == 'Liver_Patient':
                    first_var = 'Alkphos'
                    second_var = 'Sgot'
                if file_tag == 'maintenance':
                    first_var = 'Rotational speed [rpm]'
                    second_var = 'Torque [Nm]'
                if file_tag == 'ObesityDataSet':
                    first_var = 'FAF'
                    second_var = 'Height'
                if file_tag == 'phone':
                    first_var = 'int_memory'
                    second_var = 'mobile_wt'
                if file_tag == 'Placement':
                    first_var = 'ssc_p'
                    second_var = 'hsc_p'
                if file_tag == 'StressLevelDataset':
                    first_var = 'basic_needs'
                    second_var = 'bullying'
                if file_tag == 'urinalysis_tests':
                    first_var = 'Age'
                    second_var = 'pH'
                if file_tag == 'water_potability':
                    first_var = 'Hardness'
                    second_var = 'Chloramines'
                if file_tag == 'Iris':
                    new_row['description'] = f'An image showing a decision tree with depth = 2 where the first and second decisions are made with variable PetalWidthCm.'
                else:
                    new_row['description'] = f'An image showing a decision tree with depth = 2 where the first decision is made with variable {first_var} and the second with variable {second_var}.'
            elif 'overfitting_mlp' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of a mlp where the y-axis represents the accuracy and the x-axis represents the number of iterations ranging from 100 to 1000.'
            elif 'overfitting_gb' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of gradient boosting where the y-axis represents the accuracy and the x-axis represents the number of estimators ranging from 2 to 2002.'
            elif 'overfitting_rf' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of random forest where the y-axis represents the accuracy and the x-axis represents the number of estimators ranging from 2 to 2002.'
            elif 'overfitting_knn' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of k-nearest neighbors where the y-axis represents the accuracy and the x-axis represents the number of neighbors ranging from 1 to 23.'
            elif 'overfitting_dt_acc_rec' in image:
                new_row['description'] = 'A multi-line chart showing the overfitting of decision tree where the y-axis represents the performance of both accuracy and recall and the x-axis represents the max depth ranging from 2 to 25.'
            elif 'pca' in image:
                n_pc = 0
                if file_tag == 'adult':
                    n_pc = 6
                if file_tag == 'BankNoteAuthentication':
                    n_pc = 4
                if file_tag == 'Breast_Cancer':
                    n_pc = 10
                if file_tag == 'Churn_Modelling':
                    n_pc = 6
                if file_tag == 'diabetes':
                    n_pc = 8
                if file_tag == 'heart':
                    n_pc = 10
                if file_tag == 'Titanic':
                    n_pc = 5
                if file_tag == 'vehicle':
                    n_pc = 11
                if file_tag == 'Wine':
                    n_pc = 11
                if file_tag == 'WineQT':
                    n_pc = 11
                if file_tag == 'apple_quality':
                    n_pc = 7
                if file_tag == 'loan_data':
                    n_pc = 4
                if file_tag == 'credit_customers':
                    n_pc = 6
                if file_tag == 'smoking_drinking':
                    n_pc = 12
                if file_tag == 'sky_survey':
                    n_pc = 8
                if file_tag == 'weatherAUS':
                    n_pc = 7
                if file_tag == 'Dry_Bean_Dataset':
                    n_pc = 9
                if file_tag == 'abalone':
                    n_pc = 8
                if file_tag == 'car_insurance':
                    n_pc = 9
                if file_tag == 'Covid_Data':
                    n_pc = 12
                if file_tag == 'customer_segmentation':
                    n_pc = 3
                if file_tag == 'detect_dataset':
                    n_pc = 6
                if file_tag == 'e-commerce':
                    n_pc = 6
                if file_tag == 'Employee':
                    n_pc = 4
                if file_tag == 'Hotel_Reservations':
                    n_pc = 9
                if file_tag == 'Liver_Patient':
                    n_pc = 9
                if file_tag == 'maintenance':
                    n_pc = 5
                if file_tag == 'ObesityDataSet':
                    n_pc = 8
                if file_tag == 'phone':
                    n_pc = 12
                if file_tag == 'Placement':
                    n_pc = 5
                if file_tag == 'StressLevelDataset':
                    n_pc = 10
                if file_tag == 'urinalysis_tests':
                    n_pc = 3
                if file_tag == 'water_potability':
                    n_pc = 7
                if file_tag == 'Iris':
                    n_pc = 4
                new_row['description'] = f'A bar chart showing the explained variance ratio of {n_pc} principal components.'
            elif 'correlation_heatmap' in image:
                new_row['description'] = f'A heatmap showing the correlation between the variables of the dataset. The variables are {vars[file_tag]["numeric_vars"]}.'
            elif 'boxplots' in image:
                new_row['description'] = f'A set of boxplots of the variables {vars[file_tag]["numeric_vars"]}.'
            elif 'histograms_numeric' in image:
                new_row['description'] = f'A set of histograms of the variables {vars[file_tag]["numeric_vars"]}.'
            elif 'histograms_symbolic' in image:
                new_row['description'] = f'A set of bar charts of the variables {vars[file_tag]["symbolic_vars"]}.'
            elif 'mv' in image:
                if len(vars[file_tag]["missing_values"]) == 0:
                    continue
                new_row['description'] = f'A bar chart showing the number of missing values per variable of the dataset. The variables that have missing values are: {vars[file_tag]["missing_values"]}.'
            elif 'class_histogram' in image:
                new_row['description'] = f'A bar chart showing the distribution of the target variable {vars[file_tag]["target"]}.'
            elif 'nr_records_nr_variables' in image:
                new_row['description'] = f'A bar chart showing the number of records and variables of the dataset.'
            #elif 'scatter-plots' in image:
            #    new_row['description'] = f'A set of scatter plots of the variables {vars[file_tag]["symbolic_vars"]}.'
            if 'scatter-plots' in image:
                continue
            new_data.loc[len(new_data)] = new_row

new_data.to_csv('desc_dataset.csv', sep =";", index=False)
    
