# iterar refs_dataset adicionar nova coluna e escolher uma questão por template para por lá compparando com o TemplatesAssociation
import pandas as pd
import random
from tqdm import tqdm

questions = pd.read_csv('UnreducedDataset.csv',sep=';')

templates = pd.read_csv('TemplatesAssociation.csv',sep=';')

df = pd.read_csv('refs_dataset.csv',sep=';')

new_data = pd.DataFrame(columns=['Chart','description','Questions'])

qa_dataset = pd.DataFrame(columns=['Chart','Question','Id'])

def getfst(str):
    res = ''
    for i in str:
        if i == '[':
            break
        res += i
    return res

def getlst(str):
    res = ''
    for i in range(len(str)-1,-1,-1):
        if str[i] == ']':
            break
        res = str[i] + res
    return res

for index,row in tqdm(df.iterrows()):
    for indext,rowt in templates.iterrows():
        exception = 'overfitting_' + rowt['Chart'][1:-1]
        if rowt['Chart'][1:-1] in row['Chart'] and exception not in row['Chart']:
            temps = rowt['Templates'][1:-1].split('\', ')
            final_temps = []
            for temp in temps:
                temp = temp.replace('\'','')
                final_temps.append(temp)
            
            q_list = []
            for temp in final_temps:
                fst = getfst(temp)
                lst = getlst(temp)
                
                possible_questions = []
                for indexq,rowq in questions.iterrows():
                    if rowq['Chart'] == row['Chart'] and fst in rowq['Question'] and lst in rowq['Question']:
                        possible_questions.append(rowq['Question'])
                #print(possible_questions)
                random_number = random.randint(0, len(possible_questions) - 1)
                item = possible_questions[random_number]
                q_list.append(item)
                
                new_row_qa = {'Chart': row['Chart'], 'Question': item, 'Id': rowq['Id']}
                qa_dataset.loc[len(qa_dataset)] = new_row_qa
                
            new_row = {'Chart': row['Chart'], 'description': row['description'], 'Questions': q_list}
            new_data.loc[len(new_data)] = new_row
            break
        
new_data.to_csv('QuestionsListForEachChart.csv', sep=";",index=False)
qa_dataset.to_csv('ReducedDataset.csv', sep=";",index=False)

