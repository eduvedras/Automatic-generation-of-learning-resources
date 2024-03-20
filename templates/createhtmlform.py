# import module 
import codecs 
from pandas import read_csv
import pandas as pd
  
templates = read_csv('Templates.csv', sep=';')
# to open/create a new html file in the write mode 
f = open('dataset.html', 'w') 
  
# the html code which will go in the file GFG.html 
html_template = """ 
<html> 
<head></head> 
<body> """

answers_dataset = pd.DataFrame(columns=['QuestionId', 'Answer'])
data = read_csv('qa_dataset.csv', sep=';')

for index, row in data.iterrows():
    answers_dataset.loc[len(answers_dataset)] = {'QuestionId': row['Id'], 'Answer':''}
    html_template += f"""
    <img src="images/{row['Chart']}" width="auto" height = "600"/> 
    <p>{row['Id']}: {row['Question']}</p>"""

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