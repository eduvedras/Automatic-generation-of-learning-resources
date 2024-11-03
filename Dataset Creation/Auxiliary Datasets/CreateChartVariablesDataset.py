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
                
        vars[file_tag] = {'target': [classes[file_tag][0]], 
                          'binary': classes[file_tag][0] in variable_types[file_tag]['binary'],
                          'numeric_vars': numeric_vars, 
                          'symbolic_vars': symbolic_vars,
                          'missing_values': mv}
        
def create_string(list):
    string = ''
    for i in range(len(list)):
        if i == len(list) - 1:
            string += list[i]
        else:
            string += list[i] + ', '
    return string


for filename in os.scandir(directory):
    if filename.is_file():
        file_tag = filename.name[:-4]
        for image in images:
            new_row = {'Chart': file_tag + image, 'description': ''}
            if 'overfitting_decision_tree' in image:
                continue
            elif 'decision_tree' in image:
                first_var = ''
                second_var = ''
                if file_tag == 'adult':
                    first_var = 'hours-per-week <= 41.5'
                    second_var = 'capital-loss <= 1820.5'
                if file_tag == 'BankNoteAuthentication':
                    first_var = 'skewness <= 5.16'
                    second_var = 'curtosis <= 0.19'
                if file_tag == 'Breast_Cancer':
                    first_var = 'perimeter_mean <= 90.47'
                    second_var = 'texture_worst <= 27.89'
                if file_tag == 'Churn_Modelling':
                    first_var = 'Age <= 42.5'
                    second_var = 'NumOfProducts <= 2.5'
                if file_tag == 'diabetes':
                    first_var = 'BMI <= 29.85'
                    second_var = 'Age <= 27.5'
                if file_tag == 'heart':
                    first_var = 'slope <= 1.5'
                    second_var = 'restecg <= 0.5'
                if file_tag == 'Titanic':
                    first_var = 'Pclass <= 2.5'
                    second_var = 'Parch <= 0.5'
                if file_tag == 'vehicle':
                    first_var = 'MAJORSKEWNESS <= 74.5'
                    second_var = 'CIRCULARITY <= 49.5'
                if file_tag == 'Wine':
                    first_var = 'Total phenols <= 2.36'
                    second_var = 'Proanthocyanins <= 1.58'
                if file_tag == 'WineQT':
                    first_var = 'density <= 1.0'
                    second_var = 'chlorides <= 0.08'
                if file_tag == 'apple_quality':
                    first_var = 'Juiciness <= -0.3'
                    second_var = 'Crunchiness <= 2.25'
                if file_tag == 'loan_data':
                    first_var = 'Loan_Amount_Term <= 420.0'
                    second_var = 'ApplicantIncome <= 1519.0'
                if file_tag == 'credit_customers':
                    first_var = 'existing_credits <= 1.5'
                    second_var = 'residence_since <= 3.5'
                if file_tag == 'smoking_drinking':
                    first_var = 'SMK_stat_type_cd <= 1.5'
                    second_var = 'gamma_GTP <= 35.5'
                if file_tag == 'sky_survey':
                    first_var = 'dec <= 22.21'
                    second_var = 'mjd <= 55090.5'
                if file_tag == 'weatherAUS':
                    first_var = 'Rainfall <= 0.1'
                    second_var = 'Pressure3pm <= 1009.65'
                if file_tag == 'Dry_Bean_Dataset':
                    first_var = 'Area <= 39172.5'
                    second_var = 'AspectRation <= 1.86'
                if file_tag == 'abalone':
                    first_var = 'Height <= 0.13'
                    second_var = 'Diameter <= 0.45'
                if file_tag == 'car_insurance':
                    first_var = 'displacement <= 1196.5'
                    second_var = 'height <= 1519.0'
                if file_tag == 'Covid_Data':
                    first_var = 'CARDIOVASCULAR <= 50.0'
                    second_var = 'ASHTMA <= 1.5'
                if file_tag == 'customer_segmentation':
                    first_var = 'Family_Size <= 2.5'
                    second_var = 'Work_Experience <= 9.5'
                if file_tag == 'detect_dataset':
                    first_var = 'Ic <= 71.01'
                    second_var = 'Vb <= -0.37'
                if file_tag == 'e-commerce':
                    first_var = 'Prior_purchases <= 3.5'
                    second_var = 'Customer_care_calls <= 4.5'
                if file_tag == 'Employee':
                    first_var = 'JoiningYear <= 2017.5'
                    second_var = 'ExperienceInCurrentDomain <= 3.5'
                if file_tag == 'Hotel_Reservations':
                    first_var = 'lead_time <= 151.5'
                    second_var = 'no_of_special_requests <= 2.5'
                if file_tag == 'Liver_Patient':
                    first_var = 'Alkphos <= 211.5'
                    second_var = 'Sgot <= 26.5'
                if file_tag == 'maintenance':
                    first_var = 'Rotational speed [rpm] <= 1381.5'
                    second_var = 'Torque [Nm] <= 65.05'
                if file_tag == 'ObesityDataSet':
                    first_var = 'FAF <= 2.0'
                    second_var = 'Height <= 1.72'
                if file_tag == 'phone':
                    first_var = 'int_memory <= 30.5'
                    second_var = 'mobile_wt <= 91.5'
                if file_tag == 'Placement':
                    first_var = 'ssc_p <= 60.09'
                    second_var = 'hsc_p <= 70.24'
                if file_tag == 'StressLevelDataset':
                    first_var = 'basic_needs <= 3.5'
                    second_var = 'bullying <= 1.5'
                if file_tag == 'urinalysis_tests':
                    first_var = 'Age <= 0.1'
                    second_var = 'pH <= 5.5'
                if file_tag == 'water_potability':
                    first_var = 'Hardness <= 278.29'
                    second_var = 'Chloramines <= 6.7'
                if file_tag == 'Iris':
                    new_row['description'] = create_string(['PetalWidthCm <= 0.7', 'PetalWidthCm <= 1.75'])
                else:
                    new_row['description'] = create_string([first_var,second_var])
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
                new_row['description'] = n_pc
            elif 'correlation_heatmap' in image:
                new_row['description'] = create_string(vars[file_tag]["numeric_vars"])
            elif 'boxplots' in image:
                new_row['description'] = create_string(vars[file_tag]["numeric_vars"])
            elif 'histograms_numeric' in image:
                new_row['description'] = create_string(vars[file_tag]["numeric_vars"])
            elif 'histograms_symbolic' in image:
                if len(vars[file_tag]["symbolic_vars"]) == 0:
                    continue
                new_row['description'] = create_string(vars[file_tag]["symbolic_vars"])
            elif 'mv' in image:
                if len(vars[file_tag]["missing_values"]) == 0:
                    continue
                new_row['description'] = create_string(vars[file_tag]["missing_values"])
            elif 'class_histogram' in image:
                new_row['description'] = create_string(vars[file_tag]["target"])
            else:
                continue
            new_data.loc[len(new_data)] = new_row

new_data.to_csv('vars_dataset.csv', sep =";", index=False)