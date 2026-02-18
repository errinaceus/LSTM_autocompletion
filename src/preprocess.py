import pandas as pd
import re
import os
import zipfile
import requests
from sklearn.model_selection import train_test_split

RANDOM_STATE = 12345


 
def add(a, b):
    return a + b

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r':.', '', text)                  # эмодзи
    text = re.sub(r'http[s]?://[^\s]+', '', text)   # ссылки
    text = re.sub(r'@[^\s]+', '', text)             # упоминания
    text = re.sub(r'[^a-z0-9\s]', ' ', text)        # спецсимволы
    text = re.sub(r'\s+', ' ', text).strip()        # лишние пробелы
    return text

def prepare_dataset():
    df = pd.read_csv("data/tweets.txt", encoding='latin1',
                     names=['target', 'id', 'date', 'flag', 'user', 'text'])
    df['text'] = df['text'].astype(str).apply(clean_text)
    df = df[df['text'].str.len() > 1]
    df[['text']].to_csv("data/dataset_processed.csv", index=False)

def split_dataset():
    df = pd.read_csv("data/dataset_processed.csv")
    train, eval = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    val, test = train_test_split(eval, test_size=0.5, random_state=RANDOM_STATE)

    train[['text']].to_csv("data/train.csv", index=False)
    val[['text']].to_csv("data/val.csv", index=False)
    test[['text']].to_csv("data/test.csv", index=False)

    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")