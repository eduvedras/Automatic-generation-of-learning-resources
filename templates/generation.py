from pandas import read_csv
import random
import pandas as pd
from dslabs_functions import get_variable_types

def getnewindex(already_used,min,max):
    ind = random.randint(min, max)
    while ind in already_used and len(already_used) < max-min+1:
        ind = random.randint(min, max)
    already_used.append(ind)
    return ind

file_tag = 'Breast_Cancer'
target = 'diagnosis'

data = read_csv('datasets/' + file_tag + '.csv', index_col='id', sep=',', decimal='.')

templates = read_csv('Templates.csv', sep=';')

print(templates['Space4'].unique())

questions_data = pd.DataFrame(columns=['Question', 'Charts'])

special_cases = ['The number of [1] is [2] than the number of [3] for the presented tree.','The [1] for the presented tree is [2] than its [3].']

variables_types: dict[str, list] = get_variable_types(data)

if target not in variables_types["binary"]:
    templates = templates.loc[templates["Template"] != 'The difference between recall and accuracy becomes smaller with the depth due to the overfitting phenomenon.']

for index, row in templates.iterrows():
    j=0
    new_row = {'Question': row['Template'], 'Charts': row['Charts']}
    current_templates = []
    while j < 4:
        j+=1
        if type(row['Space'+str(j)]) != float:
            row['Space'+str(j)] = row['Space'+str(j)][1:-1]
            current_options = row['Space'+str(j)].split(',')
            if '..' in current_options[0]:
                current_options = list(range(int(current_options[0].split('..')[0]), int(current_options[0].split('..')[1])+1))
            
            if len(current_options) == 2 and current_options[1] == '<class>':
                special_cases.append(new_row['Question'])
                current_options = list(data.columns)
            elif current_options[0] == '<variables>':
                special_cases.append(new_row['Question'])
                current_options = list(data.drop(target, axis=1).columns)
                
            if len(current_options) > 10 and isinstance(current_options[0], int):
                i=0
                already_used = []
                tmp_templates = []
                while i < 5:
                    if j == 1:
                        aux_row = new_row.copy()
                        index = getnewindex(already_used, current_options[0], current_options[-1])
                        aux_row['Question'] =  new_row['Question'].replace('[' + str(j) + ']', str(index))
                        tmp_templates.append(aux_row)
                    else:
                        for template in current_templates:
                            aux_tmp = template.copy()
                            index = getnewindex(already_used, current_options[0], current_options[-1])
                            aux_tmp['Question'] = aux_tmp['Question'].replace('[' + str(j) + ']', str(index))
                            tmp_templates.append(aux_tmp)
                    i+=1
                current_templates = tmp_templates
            else:
                i = 0
                tmp_templates = []
                while i < len(current_options):
                    if j == 1:
                        aux_row = new_row.copy()
                        aux_row['Question'] =  new_row['Question'].replace('[' + str(j) + ']', str(current_options[i]))
                        tmp_templates.append(aux_row)
                    else:
                        for template in current_templates:
                            aux_tmp = template.copy()
                            if new_row['Question'] not in special_cases or current_options[i] not in aux_tmp['Question']:
                                aux_tmp['Question'] = aux_tmp['Question'].replace('[' + str(j) + ']', str(current_options[i]))      
                                tmp_templates.append(aux_tmp)
                    i += 1
                current_templates = tmp_templates
        else:
            if j == 1:
                current_templates.append(new_row)
            break
    while len(current_templates) > 0:
        new_row = current_templates.pop()
        questions_data.loc[len(questions_data)] = new_row

questions_data.to_csv('Questions/' + file_tag + '_questions.csv', sep=';', index=False)
