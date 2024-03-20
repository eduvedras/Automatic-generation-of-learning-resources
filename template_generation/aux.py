from PIL import Image
import requests
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_dataset('eduvedras/Desc_Questions',split='test',trust_remote_code=True)

questions = dataset[0]['Questions'][2:-2].split("', '")

rand_int = random.randint(0,len(questions) - 1)

print(questions[rand_int])