# import module 
import codecs 
from pandas import read_csv
import pandas as pd
  
# to open/create a new html file in the write mode 
f = open('dataset.html', 'w') 
  
# the html code which will go in the file GFG.html 
html_template = """ 
<html> 
<head></head> 
<body> """

#answers_dataset = pd.DataFrame(columns=['QuestionId', 'Answer'])
#data = read_csv('../template_generation/qa_dataset.csv', sep=';')

answers_dataset = pd.DataFrame(columns=['QuestionId', 'Score'])
directory = '/home/eduvedras/tese/model/results-final/results'
import os
for file in os.scandir(directory):
    if file.is_file() and "csv" in file.name:
        data = pd.read_csv(file.path)
        file_tag = file.name[:-4]
        html_template += f"""
            <p></p>
            <h3>-----------------------------{file_tag}-----------------------------</h3>"""
        for index, row in data.iterrows():
            answers_dataset.loc[len(answers_dataset)] = {'QuestionId': row['Id'], 'Score':''}
            html_template += f"""
            <img src="images/{row['Image']}" width="auto" height = "600"/> 
            <p>{row['Id']}: {row['Prediction']}</p>"""

html_template += """
</body> 
</html> 
"""
  
# writing the code into the file 
f.write(html_template) 
  
# close the file 
f.close() 

answers_dataset.to_csv('answers.csv', sep=';', index=False)
  
# viewing html files 
# below code creates a  
# codecs.StreamReaderWriter object 
#file = codecs.open("dataset.html", 'r', "utf-8") 
  
# using .read method to view the html  
# code from our object 
#print(file.read())