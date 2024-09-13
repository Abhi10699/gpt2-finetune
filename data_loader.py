import pandas as pd
from tqdm import tqdm 
from datasets import Dataset

def map_labels_to_emotion(label_int):
  if label_int == 0:
    return 'sadness'
  elif label_int == 1:
    return 'joy'
  elif label_int == 2:
    return 'love'
  elif label_int == '3':
    return 'anger'
  elif label_int == '4':
    return 'fear'
  else:
    return 'surprise'

def generate_input_sequences(df: pd.DataFrame):

  """
  create input sequences
  foreg: <emotion>joy</emotion>I feel so good today. Thank you god!<|endoftext|>
  """

  train_seqs = []
  
  for row in tqdm(df.values):
    text = row[0]
    emotion = row[-1]
    input_text = f"<emotion>{emotion}</emotion>{text}<|endoftext|>"
    train_seqs.append(input_text)
  
  # generate new df of just those texts
  return train_seqs


def tokenize_inputs(tokenizer, sequences):
  return tokenizer(sequences['texts'], truncation=True, padding='max_length', max_length=512)


def create_dataset(csv_path: str):
  df = pd.read_csv(csv_path)
  df = df.drop(['Unnamed: 0'], axis=1)
  
  # for testing purpose only
  

  # map emotions to labels
  df['emotion'] = df['label'].apply(lambda x: map_labels_to_emotion(x))

  train_df = pd.DataFrame({"texts": generate_input_sequences(df)})
  
  # generate hf dataset

  emotion_dataset = Dataset.from_pandas(train_df)
  return emotion_dataset