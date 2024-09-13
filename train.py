import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from data_loader import create_dataset


DATASET_PATH = "./data/text.csv"
dataset = create_dataset(DATASET_PATH)
print(dataset)