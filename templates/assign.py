from pandas import read_csv
import pandas as pd

file_tag = 'WineQT'
questions = read_csv('Questions/' + file_tag + '_questions.csv', sep=';')

descriptions = read_csv('descriptions.csv', sep=';')

questions_answer_data = pd.DataFrame(columns=['Question', 'Charts_id'])

description = descriptions[descriptions['Dataset'] == file_tag]['Description'].iloc[0]

for index, row in questions.iterrows():
    charts = []
    row['Charts'] = row['Charts'][1:-1]
    aux = row['Charts'].split(',')
    for chart in aux:
        if chart == 'Nr records x nr variables':
            charts.append(f'{file_tag}_nr_records_nr_variables')
        if chart == 'Correlation heatmap':
            charts.append(f'{file_tag}_correlation_heatmap')
        if chart == 'Histograms + MV':
            charts.append(f'{file_tag}_histograms_numeric')
            charts.append(f'{file_tag}_mv')
        if chart == 'Scatter-plots':
            charts.append(f'{file_tag}_scatter-plots')
        if chart == 'Histograms':
            charts.append(f'{file_tag}_histograms_numeric')
        if chart == 'All_Histograms':
            charts.append(f'{file_tag}_histograms_numeric')
            charts.append(f'{file_tag}_histograms_symbolic')
        if chart == 'Boxplots':
            charts.append(f'{file_tag}_boxplots')
        if chart == 'Single boxplots':
            charts.append(f'{file_tag}_boxplots')
        if chart == 'Description':
            charts.append(f'{file_tag}_description:{description}')
        if chart == 'Decision tree':
            charts.append(f'{file_tag}_decision_tree')
        if chart == 'Class histogram':
            charts.append(f'{file_tag}_class_histogram')
        if chart == 'Missing values':
            charts.append(f'{file_tag}_mv')
        if chart == 'PCA':
            charts.append(f'{file_tag}_pca')
        if chart == 'Overfitting Decision Tree':
            charts.append(f'{file_tag}_overfitting_decision_tree')
        if chart == 'Overfitting Decision Tree + Overfitting KNN':
            charts.append(f'{file_tag}_overfitting_decision_tree')
            charts.append(f'{file_tag}_overfitting_knn')
        if chart == 'Overfitting KNN':
            charts.append(f'{file_tag}_overfitting_knn')
        if chart == 'Overfitting RF':
            charts.append(f'{file_tag}_overfitting_rf')
        if chart == 'Overfitting GB':
            charts.append(f'{file_tag}_overfitting_gb')
        if chart == 'Overfitting MLP':
            charts.append(f'{file_tag}_overfitting_mlp')
        if chart == 'Overfitting Accuracy + Recall':
            charts.append(f'{file_tag}_overfitting_dt_acc_rec')
    
    new_row = {'Question': row['Question'], 'Charts_id': charts}
    questions_answer_data.loc[len(questions_answer_data)] = new_row

questions_answer_data.to_csv('Questions_Chart/' + file_tag + '_Questions_Chart.csv', sep=';', index=False)
