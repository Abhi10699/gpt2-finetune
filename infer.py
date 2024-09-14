import torch
import warnings

from peft import PeftConfig,PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# ignore warnings

warnings.filterwarnings('ignore')

BASE_MODEL_ID = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PEFT_MODEL_PATH = './full_train/checkpoint-7500'


# initialise peft config

peft_config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)

# setup tokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.add_special_tokens({
  "pad_token": "<pad>",
  "additional_special_tokens": ["<emotion>", "</emotion>"]
})

# setup model and load the fine tuned checkpoint

gpt_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID).to(DEVICE)
gpt_base.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(gpt_base, PEFT_MODEL_PATH)

# Time to Generate 🤤


for mood in ["sadness", "love", "joy", "surprise", "anger", "fear"]:
  
  user_input = f"<emotion>{mood}</emotion>ive"
  inputs = tokenizer(user_input, return_tensors="pt").to(DEVICE)
  
  outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
  print(tokenizer.batch_decode(outputs.detach().cpu().numpy())[0])

  print("\n")