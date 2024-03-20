from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import gca,figure, savefig, show, subplots

data = read_csv("metadata1.csv", sep=',')

templates = read_csv('Templates.csv', sep=';')
lst = []
dic = {}
for index, row in templates.iterrows():
    first_str = row['Template'].split('[')[0]
    lst_str = row['Template'].split(']')[-1]
    lst.append([first_str,lst_str])
    if first_str == lst_str:
        first_str = first_str[:int(len(first_str)/2)]
        lst_str = lst_str[int(len(lst_str)/2):]
    dic[first_str+"|"+lst_str] = []

data_balanced = DataFrame(columns=['Chart','Question','Id'])

def getkey(question):
    for key in dic:
        if key.split('|')[0] in question and key.split('|')[1] in question:
            break
    return key

for index, row in data.iterrows():
    key = getkey(row['Question'])
    dic[key].append(row['Id'])
    
list_aux = []

for key in dic:
    if len(dic[key]) == 0:
        print(key + "\n")
    
#list_aux.sort()
#print(list_aux)
#print(dic.keys())

import random
'''
def add_questions_to_list(list_to_add, list_of_questions, number_of_questions):
    for i in range(number_of_questions):
        index = random.randint(0, len(list_of_questions)-1)
        if list_of_questions[index] not in list_to_add:
            list_to_add.append(list_of_questions[index])
        else:
            while list_of_questions[index] in list_to_add:
                index = random.randint(0, len(list_of_questions)-1)
                if list_of_questions[index] not in list_to_add:
                    list_to_add.append(list_of_questions[index])
                    break

questions_for_new_dataset = []

for key in dic:
    if len(dic[key]) > 0:
        number_of_questions = len(dic[key])
        if number_of_questions > 290:
            number_of_questions = 290
        if number_of_questions == 0:
            number_of_questions = 1
        add_questions_to_list(questions_for_new_dataset, dic[key], number_of_questions)

def suma(list):
    sum = 0
    for i in list:
        sum += i
    return sum

def list_of_unique_values(list):
    unique = []
    for i in list:
        if i not in unique:
            unique.append(i)
    return unique

def all_elements_in_lists_are_different(list1, list2):
    for i in list1:
        if i in list2:
            return False
    return True

print(len(questions_for_new_dataset))

for i in questions_for_new_dataset:
    data_balanced.loc[len(data_balanced)] = {'Chart': data.loc[data['Id'] == i]['Chart'].values[0], 'Question': data.loc[data['Id'] == i]['Question'].values[0], 'Id': i}
    
data_balanced.to_csv('metadata2.csv', index=False)'''