from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertModel
from tokenizers import BertWordPieceTokenizer

import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

device = None

device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print (device)
else:
    print ("MPS device not found.")


# 모델 및 토크나이저 로드
model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# test 데이터 가져오기
test_data = "./yahoo_answers_csv/test.csv"
test_df = pd.read_csv(test_data)

sentence = test_df.iloc[0, 1] + " " + test_df.iloc[0, 2] + " " + test_df.iloc[0, 3]
true_label = test_df.iloc[0, 0]
# print(f"sentence: {sentence}")

inputs = tokenizer(sentence, return_tensors="pt").to(device)

token_length = inputs.input_ids.shape[1]
# print(f"token length: {token_length}\n")


# 모델에 입력값 넣기
outputs = model(**inputs)

predictions = outputs.logits.argmax(dim=-1)
# print(f"pred output: {predictions.item() + 1}")
# print(f"true label: {true_label}")
# print(outputs.logits)

class TestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.sentences = df.iloc[:, 1:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).values
        self.labels = df.iloc[:, 0].values
        self.tokenizer = tokenizer
  
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer(sentence, truncation=True, max_length=512, padding='max_length', return_tensors="pt")
        label = torch.tensor(self.labels[idx]) # label은 1부터 시작하기 때문에, 나중에 inference할 때에 예측값에 1을 더해줘야 합니다.
        
        return inputs, label

test_data = "./yahoo_answers_csv/test.csv"
test_df = pd.read_csv(test_data, header=None)

test_dataset = TestDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for layer_index in range(6, 7): # 6
    for head_index in range(12):
        
        print('\n====================================================')
        model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
        model = model.to(device)

        # print(model.bert.encoder.layer[layer_index].attention.self.value.weight.data)
        model.bert.encoder.layer[layer_index].attention.prune_heads([head_index])
        # print(model.bert.encoder.layer[layer_index].attention.self.value.weight.data)

        preds = []
        true_labels = []

        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = batch
            inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()} 
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            prediction = outputs.logits.argmax(dim=-1) + 1
            
            preds.extend(prediction.tolist())
            true_labels.extend(labels.tolist())

        print(f"off attention head index // layer: {layer_index+1}, head: {head_index+1}")
            
        print('classification report')
        print(classification_report(true_labels, preds, digits=5))

        print('confusion matrix')
        print(confusion_matrix(true_labels, preds))
        
        con = confusion_matrix(true_labels, preds).diagonal()
        for i in range(10):
            print(con[i])