from pandas import read_csv
import pandas as pd

import random

dataset = read_csv('Final.csv', sep=';')

one_dataset = pd.DataFrame(columns=['Question', 'Chart'])
        
questions = []
for index, row in dataset.iterrows():
    if '[]' in row['Charts_id']:
        continue
    row['Charts_id'] = row['Charts_id'][1:-1]
    current_options = row['Charts_id'].split(',')
    
    if 'The figure doesn’t show any missing values for' in row['Question']:
        #Two obligatory charts
        continue
    elif 'According to the charts, KNN and Decision Trees present a similar behaviour.' in row['Question']:
        #Two obligatory charts
        continue
    elif 'Decision trees and KNN show similar behaviours.' in row['Question']:
        #Two obligatory charts
        continue
    elif 'KNN and Decision Trees show a different trend in the majority of hyperparameters tested.' in row['Question']:
        #Two obligatory charts
        continue
    elif len(current_options) == 1 and 'description' in current_options[0]:
        continue
    elif len(current_options) > 1:
        if 'description' in current_options[0]:
            new_row = {'Question': row['Question'], 'Chart': current_options[1]}
        elif 'scatter-plots' in current_options[0] and 'histograms' in current_options[1]:
            new_row = {'Question': row['Question'], 'Chart': current_options[1]}
        elif 'decision_tree' in current_options[0] and 'correlation_heatmap' in current_options[1]:
            new_row = {'Question': row['Question'], 'Chart': current_options[1]}
        elif ('boxplots' in current_options[0] and 'histograms' in current_options[1]) or ('boxplots' in current_options[1] and 'histograms' in current_options[0]):
            ind = random.randint(0, 1)
            new_row = {'Question': row['Question'], 'Chart': current_options[ind]}
        elif 'mv' in current_options[0] and 'histograms' in current_options[1]:
            new_row = {'Question': row['Question'], 'Chart': current_options[1]}
    else:
        new_row = {'Question': row['Question'], 'Chart': current_options[0]}
    questions.append(new_row)
    
while len(questions) > 0:
    new_row = questions.pop()
    one_dataset.loc[len(one_dataset)] = new_row
    
one_dataset.to_csv('OneImageDataset.csv', sep=';', index=False)