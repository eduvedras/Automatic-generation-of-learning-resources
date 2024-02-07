from datasets import load_dataset
import pandas as pd
from pandas import read_csv
  
data = read_csv('OriginalDataset.csv', sep=';',index_col="Id")

from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = 'openai/clip-vit-base-patch32'

model = CLIPModel.from_pretrained(model_id).to(device)
tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

prompt = "a dog in the snow"

#tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to(device)
print(inputs)