import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig
from load_data import *
from sklearn.model_selection import train_test_split

# from transformers import get_cosine_with_hard_restarts_schedule_with_warmup, EarlyStoppingCallback
from torch.optim import Adam
import wandb

import random
import numpy as np

from sklearn.model_selection import StratifiedKFold

#seed 고정
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train(train_dataset,val_dataset,fold):
  # load model and tokenizer
  MODEL_NAME = "xlm-roberta-large"
  tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

  train_label=train_dataset['topic_idx'].values
  val_label=val_dataset['topic_idx'].values

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_val = tokenized_dataset(val_dataset, tokenizer)

  # make dataset for pytorch.
  news_train_dataset = news_Dataset(tokenized_train, train_label)
  news_val_dataset = news_Dataset(tokenized_val, val_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  bert_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
  bert_config.num_labels = 7
  model = XLMRobertaForSequenceClassification.from_pretrained("./results/roberta4/checkpoint-15000",config=bert_config)
  # model.__call__().attention_mask = 0
  # model.__call__().input_ids = None
  # model.__call__().inputs_embeds = None
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results/roberta'+str(fold),
    save_total_limit=5,
    save_steps=3000,
    num_train_epochs=70,
    learning_rate=1e-7,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    warmup_steps=3000,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=1000,
    evaluation_strategy='steps',
    eval_steps = 2000,
    dataloader_num_workers=4,
    # lr_scheduler_type='constant_with_warmup',
    label_smoothing_factor=0.2
  )

  wandb.config.update(training_args,allow_val_change=True)

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=news_train_dataset,
    eval_dataset=news_val_dataset,
    compute_metrics=compute_metrics
  )
  # early_stopping = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)

  # train model
  trainer.train()

def main():
  seed_everything(42)
  
  # load dataset
  data = pd.read_csv('./Data/train_data.csv').drop(['index'],axis=1)
  data,val_dataset = train_test_split(data,test_size=0.1,stratify=data['topic_idx'],random_state=42)
  # train_dataset aug
  additional_dataset = pd.read_csv('./Data/aug_backtrans_1.csv').drop(['Unnamed: 0'],axis=1).iloc[45654:,:]

  data = pd.concat([data,additional_dataset])

  # kfold
  kfold=[]

  splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  for train_idx, val_idx in splitter.split(data['title'],data['topic_idx']):
      kfold.append((data.iloc[train_idx,:],data.iloc[val_idx,:]))

  for fold,(train_dataset, valid_dataset) in enumerate(kfold):
    if fold == 4:
      wandb.init()
      wandb.run.name = 'roberta'+str(fold)
    
      wandb.run.save()
      print(f'fold{fold} 학습중...')

      train(train_dataset=train_dataset,val_dataset=val_dataset,fold=fold)
    else:
      continue

if __name__ == '__main__':
  main()
