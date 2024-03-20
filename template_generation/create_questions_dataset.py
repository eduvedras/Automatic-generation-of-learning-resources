# iterar templatesAssigned e gerar todas as questoes possiveis

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

def are_all_elements_unique(list):
    return len(list) == len(set(list))


data = pd.read_csv('TemplatesAssigned.csv',sep=';')

new_data = pd.DataFrame(columns=['Chart', 'Question', 'Id'])

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

conditions = {'adult':['hours-per-week <= 41.5','capital-loss <= 1820.5'], 'BankNoteAuthentication':['skewness <= 5.16','curtosis <= 0.19'],
                'Breast_Cancer':['perimeter_mean <= 90.47','texture_worst <= 27.89'],'Churn_Modelling':['Age <= 42.5','NumOfProducts <= 2.5'],
                'diabetes':['BMI <= 29.85','Age <= 27.5'],'heart':['slope <= 1.5','restecg <= 0.5'],
                'Titanic':['Pclass <= 2.5','Parch <= 0.5'],'vehicle':['MAJORSKEWNESS <= 74.5','CIRCULARITY <= 49.5'],
                'Wine':['Total phenols <= 2.36','Proanthocyanins <= 1.58'],'WineQT':['density <= 1.0','chlorides <= 0.08'],
                'apple_quality':['Juiciness <= -0.3','Crunchiness <= 2.25'],'loan_data':['Loan_Amount_Term <= 420.0','ApplicantIncome <= 1519.0'],
                'credit_customers':['existing_credits <= 1.5','residence_since <= 3.5'],'smoking_drinking':['SMK_stat_type_cd <= 1.5','gamma_GTP <= 35.5'],
                'sky_survey':['dec <= 22.21','mjd <= 55090.5'],'weatherAUS':['Rainfall <= 0.1','Pressure3pm <= 1009.65'],
                'Dry_Bean_Dataset':['Area <= 39172.5','AspectRation <= 1.86'],'abalone':['Height <= 0.13','Diameter <= 0.45'],
                'car_insurance':['displacement <= 1196.5','height <= 1519.0'],'Covid_Data':['CARDIOVASCULAR <= 50.0','ASHTMA <= 1.5'],
                'customer_segmentation':['Family_Size <= 2.5','Work_Experience <= 9.5'],'detect_dataset':['Ic <= 71.01','Vb <= -0.37'],
                'e-commerce':['Prior_purchases <= 3.5','Customer_care_calls <= 4.5'],'Employee':['JoiningYear <= 2017.5','ExperienceInCurrentDomain <= 3.5'],
                'Hotel_Reservations':['lead_time <= 151.5','no_of_special_requests <= 2.5'],'Liver_Patient':['Alkphos <= 211.5','Sgot <= 26.5'],
                'maintenance':['Rotational speed [rpm] <= 1381.5','Torque [Nm] <= 65.05'],'ObesityDataSet':['FAF <= 2.0','Height <= 1.72'],
                'phone':['int_memory <= 30.5','mobile_wt <= 91.5'],'Placement':['ssc_p <= 60.09','hsc_p <= 70.24'],
                'StressLevelDataset':['basic_needs <= 3.5','bullying <= 1.5'],'urinalysis_tests':['Age <= 0.1','pH <= 5.5'],
                'water_potability':['Hardness <= 278.29','Chloramines <= 6.7'],'Iris':['PetalWidthCm <= 0.7','PetalWidthCm <= 1.75']}

neighbors = {'abalone': ['932','117','683','1191'],'adult':['21974','541','9274','434'],
             'apple_quality': ['148','784','1625','243'],'BankNoteAuthentication': ['214','436','179','131'],
             'Breast_Cancer': ['184','50','20','144'],'car_insurance': ['2141','686','774','3813'],
             'Churn_Modelling': ['4831','114','1931','124'],'Covid_Data': ['173','7971','16','46'],
             'credit_customers': ['264','183','146','107'],'customer_segmentation': ['249','524','723','11'],
             'detect_dataset': ['797','6394','3','1206'],'diabetes': ['111','98','167','161'],
             'Dry_Bean_Dataset': ['760','2501','4982','1284'],'e-commerce': ['3657','906','1540','1596'],
             'Employee': ['1781','1215','44','217'],'heart': ['202','181','137','197'],'Hotel_Reservations': ['10612','9756','4955','69'],
             'Iris': ['35','38','32'],'Liver_Patient': ['77','125','109','94'],'loan_data': ['3','204','2','6'],
             'maintenance': ['943','46','21','5990'],'ObesityDataSet': ['840','370','107','160'],'phone': ['469','209','86','636'],
             'Placement': ['16','20','68','46'],'sky_survey': ['208','11119','945','1728'],'smoking_drinking': ['7218','1796','3135','2793'],
             'StressLevelDataset': ['271','240','223','36'],'Titanic': ['181','72','188','57'],'urinalysis_tests': ['3','23','215','763'],
             'vehicle': ['1','2','3','4'],'Wine': ['305','109','53','125'],'water_potability': ['8','1388','5','6'],
             'weatherAUS':['1154','1686','251','608'],'Wine':['49','12','2','60'],'WineQT':['154','27','172','447']}
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
                          'missing_values': mv,
                          'conditions': conditions[file_tag],
                          'neighbors': neighbors[file_tag]}
        
def all_variables_in_list(list):
    for el in list:
        if '<all_variables>' in el:
            return True
    return False
        
from tqdm import tqdm
i = 0
for filename in tqdm(os.scandir(directory)):
    if filename.is_file():
        file_tag = filename.name[:-4]
        for index,row in tqdm(data.iterrows()):
            options = {'1':[],'2':[],'3':[],'4':[]}
            j = 1
            while j <= 4:
                if type(row['Space'+str(j)]) != float:
                    if row['Space'+str(j)][0] == '[':
                        row['Space'+str(j)] = row['Space'+str(j)][1:-1]
                        
                    if row['Space'+str(j)] == '<class>':
                        options[str(j)] = vars[file_tag]['target']
                    elif row['Space'+str(j)] == '<all_variables>':
                        options[str(j)] = ['<all_variables>']
                    elif row['Space'+str(j)] == '<variables>':
                        options[str(j)] = vars[file_tag]['numeric_vars']
                    elif row['Space'+str(j)] == '<target-values>':
                        dataset = pd.read_csv(filename.path)
                        options[str(j)] = list(dataset[vars[file_tag]['target'][0]].unique())
                        options[str(j)] = [str(x) for x in options[str(j)]]
                    elif row['Space'+str(j)] == '<neighbors>':
                        options[str(j)] = vars[file_tag]['neighbors']
                    elif row['Space'+str(j)] == '<pca>':
                        for index in range(len(vars[file_tag]['numeric_vars']) - 2):
                            options[str(j)].append(str(index + 2))
                    elif 'Considering that A=True' in row['Template'] and j == 1:
                        options[str(j)] = row['Space'+str(j)].split('/')
                    else:
                        options[str(j)] = row['Space'+str(j)].split(',')
                
                j += 1
                        
            charts = row['Charts'][1:-1].split(',')
            for elu in charts:
                el = elu[1:-1]
                
                all_combs = []
                for index in range(len(options['1'])):
                    if len(options['2']) == 0:
                        all_combs.append([options['1'][index]])
                    for j in range(len(options['2'])):
                        if len(options['3']) == 0:
                            if options['2'][j] == '<all_variables>' or (options['1'][index] != options['2'][j]):
                                all_combs.append([options['1'][index], options['2'][j]])
                        for k in range(len(options['3'])):
                            if len(options['4']) == 0:
                                if options['3'][k] == '<all_variables>' or (options['1'][index] != options['2'][j] and options['1'][index] != options['3'][k] and options['2'][j] != options['3'][k]):
                                    all_combs.append([options['1'][index], options['2'][j], options['3'][k]])
                            for l in range(len(options['4'])):
                                if options['4'][l] == '<all_variables>' or (options['1'][index] != options['2'][j] and options['1'][index] != options['3'][k] and options['1'][index] != options['4'][l] and options['2'][j] != options['3'][k] and options['2'][j] != options['4'][l] and options['3'][k] != options['4'][l]):
                                    all_combs.append([options['1'][index], options['2'][j], options['3'][k], options['4'][l]])
            
                if len(charts) == 2 and el == 'histograms_numeric':
                    variables = vars[file_tag]['numeric_vars']
                elif len(charts) == 2 and el == 'histograms_symbolic':
                    variables = vars[file_tag]['symbolic_vars']
                    if len(variables) == 0:
                        continue
                
                aux_comb=[]            
                if all_variables_in_list(all_combs):
                    if el == 'histograms_numeric':
                        variables = vars[file_tag]['numeric_vars']
                    elif el == 'histograms_symbolic':
                        variables = vars[file_tag]['symbolic_vars']
                        if len(variables) == 0:
                            continue
                    elif el == 'mv':
                        variables = vars[file_tag]['missing_values']
                        if len(variables) == 0:
                            continue
                        
                    for comb in all_combs:
                        for val in variables:
                            new_comb = []
                            for element in range(len(comb)):
                                if comb[element] == '<all_variables>':
                                    if new_comb == []:
                                        new_comb = comb.copy()
                                    if val in new_comb:
                                        for valaux in variables:
                                            if valaux not in new_comb:
                                                new_comb[element] = valaux
                                                break
                                    else:
                                        new_comb[element] = val
                            if len(new_comb) > 0 and are_all_elements_unique(new_comb):
                                aux_comb.append(new_comb)
                                
                combinations = []
                
                for comb in all_combs:
                    if '<all_variables>' not in comb:
                        combinations.append(comb)
                
                for comb in aux_comb:
                    if comb not in combinations:
                        combinations.append(comb)
                        
                for comb in combinations:
                    new_row = {'Chart': file_tag + '_' + el + '.png', 'Question': row['Template'], 'Id': i}
                    if 'Considering that A=True'in row['Template']:
                        new_row['Question'] = row['Template'].replace('ConditionA', vars[file_tag]['conditions'][0])
                        new_row['Question'] = new_row['Question'].replace('ConditionB', vars[file_tag]['conditions'][1])
                        
                    i += 1
                    aux_row = new_row.copy()
                    for element in range(len(comb)):
                        aux_row['Question'] =  aux_row['Question'].replace('[' + str(element+1) + ']', str(comb[element]))
                    new_data.loc[len(new_data)] = aux_row
                
                if len(combinations) == 0 and '[1]' not in row['Template']:
                    new_row = {'Chart': file_tag + '_' + el + '.png', 'Question': row['Template'], 'Id': i}
                    if 'Considering that A=True'in row['Template']:
                        new_row['Question'] = row['Template'].replace('ConditionA', vars[file_tag]['conditions'][0])
                        new_row['Question'] = new_row['Question'].replace('ConditionB', vars[file_tag]['conditions'][1])
                    i += 1
                    new_data.loc[len(new_data)] = new_row

                    
new_data.to_csv('QuestionsDataset.csv', sep=';',index=False)
            
                
                
                
                
                