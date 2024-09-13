import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from data_loader import create_dataset, tokenize_inputs

DATASET_PATH = "./data/text.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load model and tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

# add special tokens 

tokenizer.add_special_tokens({
  "pad_token": "<pad>",
  "additional_special_tokens": ["<emotion>", "</emotion>"]
})

gpt.resize_token_embeddings(len(tokenizer))

# Build dataset

dataset = create_dataset(DATASET_PATH)

# tokenize the dataste

tokenized_dataset = dataset.map(lambda x: tokenize_inputs(tokenizer, x), batched=True)

# setup trainer

training_args = TrainingArguments(
  output_dir='./results_2',
  overwrite_output_dir=True,
  num_train_epochs=3,
  per_device_train_batch_size=4,
  save_steps=10_000,
  save_total_limit=2,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the Trainer
trainer = Trainer(
  model=gpt,
  args=training_args,
  data_collator=data_collator,
  train_dataset=tokenized_dataset,
)


trainer.train()