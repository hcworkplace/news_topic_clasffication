from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, XLMRobertaConfig
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []

  for i, data in enumerate(dataloader):
      with torch.no_grad():
          outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            #token_type_ids=data['token_type_ids'].to(device)
            )
          logits = outputs[0]
          logits = logits.detach().cpu().numpy()
          result = np.argmax(logits, axis=-1)

          output_pred.extend(result)
  return output_pred
  #return np.array(output_pred).flatten()

# def load_test_dataset(dataset_dir, tokenizer):
#   # test_dataset = load_data(dataset_dir)
#   test_label=submission['topic_idx'].values
#   # tokenizing dataset
#   tokenized_test = tokenized_dataset(test_data, tokenizer)
#   news_test_dataset = news_Dataset(tokenized_test, test_label)
#   return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = "xlm-roberta-large"  
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  # load my model
  MODEL_NAME='./results/roberta_label02_aug_all_1_lr_1e-7/checkpoint-403000' # model dir.
  model = XLMRobertaForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  # load test datset
  test_data=pd.read_csv('./Data/test_data.csv')
  submission=pd.read_csv('./Data/sample_submission.csv')

  # test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  # test_dataset = RE_Dataset(test_dataset ,test_label)
  test_label=submission['topic_idx'].values

  tokenized_test = tokenized_dataset(test_data, tokenizer)
  news_test_dataset = news_Dataset(tokenized_test, test_label)
  # predict answer
  # pred_answer = inference(model, test_dataset, device)
  pred_answer = inference(model, news_test_dataset, device)

  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  # output = pd.DataFrame(pred_answer, columns=['pred'])
  # output.to_csv('./prediction/submission.csv', index=False)

  submission['topic_idx']=pred_answer
  submission.to_csv('./roberta_label02_aug_all_1_lr_1e-7_checkpoint-403000.csv',index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/roberta_label02_aug_all_1_lr_1e-7/checkpoint-403000")
  args = parser.parse_args()
  print(args)
  main(args)
  
