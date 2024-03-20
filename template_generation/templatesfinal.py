import pandas as pd

df = pd.read_csv('TemplatesAssigned.csv',sep=';')

new_dataset = pd.DataFrame(columns=['Chart','Templates'])

aux = []
for index,row in df.iterrows():
    charts = row['Charts'][1:-1].split(',')
    for chart in charts:
        if chart not in aux:
            aux.append(chart)

for chart in aux:
    new_row = {'Chart': chart, 'Templates': []}
    new_dataset.loc[len(new_dataset)] = new_row
    
for index,row in df.iterrows():
    charts = row['Charts'][1:-1].split(',')
    for chart in charts:
        new_dataset.at[new_dataset[new_dataset['Chart'] == chart].index[0],'Templates'].append(row['Template'])
        
new_dataset.to_csv('TemplatesFinal.csv', sep=';',index=False)
    
        