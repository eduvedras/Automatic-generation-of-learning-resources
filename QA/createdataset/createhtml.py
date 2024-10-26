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

from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm


dataset = load_dataset('eduvedras/QA',split='test', trust_remote_code=True)

answers_dataset = pd.DataFrame(columns=['QuestionId', 'Score'])
directory = '/home/eduvedras/tese/model/results-final/results'
import os



for i in tqdm(range(len(dataset))):
    html_template += f"""
    <img src="../images/{dataset[i]['Chart_name']}" width="auto" height = "600"/> 
    <p>{dataset[i]['Id']}: {dataset[i]['Question']}</p>"""

html_template += """
</body> 
</html> 
"""
  
# writing the code into the file 
f.write(html_template) 
  
# close the file 
f.close() 
  
# viewing html files 
# below code creates a  
# codecs.StreamReaderWriter object 
#file = codecs.open("dataset.html", 'r', "utf-8") 
  
# using .read method to view the html  
# code from our object 
#print(file.read())