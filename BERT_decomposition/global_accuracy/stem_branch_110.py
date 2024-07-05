from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertModel
from tokenizers import BertWordPieceTokenizer

import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
    print (device)
else:
    print ("cuda device not found.")


# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

test_data = "b.csv"
test_df = pd.read_csv(test_data, header=None)

test_dataset = TestDataset(test_df, tokenizer)



#####======####======####======####======####======####======####======####======
#####======####======####======####======####======####======####======####======
for ite in range(32, 25, -1):
    print(ite)

    class_1_order_recall= [(4, 0), (3, 0), (7, 8), (7, 0), (8, 11), (2, 10), (1, 10), (8, 9), (2, 6), (8, 10), (4, 3), (1, 7), (8, 1), (6, 5), (5, 8), (0, 8), (6, 0), (3, 3), (3, 10), (5, 9), (5, 10), (0, 10), (3, 8), (1, 11), (6, 9), (0, 3), (1, 1), (4, 5), (0, 0), (5, 3), (10, 4), (10, 7), (1, 4), (0, 4), (2, 0), (3, 5), (2, 9), (8, 8), (2, 1), (5, 4), (9, 8), (11, 0), (10, 6), (10, 0), (7, 5), (6, 8), (11, 11), (9, 5)]
    class_1_order_precision= [(9, 5), (11, 11), (11, 6), (10, 6), (10, 11), (7, 5), (11, 0), (6, 8), (8, 8), (6, 7), (4, 4), (0, 6), (0, 1), (10, 0), (9, 10), (1, 5), (2, 0), (5, 5), (5, 6), (10, 8), (3, 2), (7, 4), (2, 1), (2, 9), (6, 2), (2, 5), (0, 7), (0, 3), (10, 3), (9, 1), (1, 0), (10, 2), (1, 3), (2, 11), (8, 7), (5, 4), (9, 9), (1, 4), (8, 2), (5, 2), (7, 2), (1, 1), (6, 1), (9, 6), (4, 6), (4, 3), (2, 7), (1, 7), (7, 9), (7, 1), (0, 4), (5, 10), (6, 6), (6, 9), (6, 10), (7, 3), (9, 7), (3, 11), (1, 8), (0, 2), (3, 1), (3, 5), (10, 9), (1, 6), (10, 10), (1, 2), (0, 11), (0, 10), (11, 9), (4, 5), (3, 10), (1, 9), (11, 5), (2, 3), (7, 11), (4, 7), (10, 7), (4, 8), (8, 3), (8, 5), (3, 7), (3, 9), (5, 3), (0, 5), (5, 1), (8, 0), (3, 8), (10, 5), (7, 7), (9, 2), (0, 0), (3, 3), (2, 2), (7, 6), (2, 8), (6, 0), (1, 10), (5, 0), (9, 8), (0, 9), (5, 8), (9, 11), (4, 11), (6, 3), (6, 4), (2, 4), (7, 10), (0, 8), (4, 9), (4, 10), (5, 11), (3, 6), (9, 4), (5, 7), (8, 11), (8, 9), (5, 9), (6, 5), (3, 4), (11, 1), (11, 3), (10, 4), (8, 4), (2, 10), (9, 0), (6, 11), (2, 6), (1, 11), (9, 3), (11, 10), (10, 1), (8, 6), (11, 7), (11, 8), (11, 2), (7, 0), (4, 1), (4, 0), (4, 2), (11, 4), (8, 1), (8, 10), (7, 8), (3, 0)]
    class_2_order_recall= [(11, 0), (9, 5), (10, 6), (0, 8), (6, 8), (9, 8), (3, 5), (10, 7), (11, 11), (3, 8), (4, 3), (1, 11), (3, 10), (0, 3), (6, 9), (5, 8), (1, 7), (4, 0), (0, 10), (2, 0), (1, 4), (6, 0), (6, 5), (7, 5), (8, 1), (2, 6), (2, 1), (0, 0), (2, 10), (1, 10), (8, 8), (2, 9), (3, 3), (10, 4), (8, 11), (8, 9), (10, 0), (7, 8), (4, 5), (1, 1), (5, 3), (8, 10), (5, 4), (5, 9), (7, 0), (5, 10), (3, 0), (0, 4)]
    class_2_order_precision= [(11, 2), (11, 8), (11, 9), (0, 9), (10, 3), (0, 11), (3, 0), (0, 4), (7, 2), (8, 8), (3, 2), (2, 5), (5, 2), (9, 4), (1, 3), (5, 8), (10, 2), (3, 6), (5, 4), (4, 10), (5, 3), (0, 1), (5, 9), (3, 1), (7, 0), (3, 4), (5, 5), (3, 9), (3, 10), (6, 4), (4, 2), (7, 1), (5, 10), (6, 11), (6, 1), (5, 1), (7, 4), (9, 2), (1, 0), (2, 9), (8, 10), (1, 10), (10, 4), (11, 3), (2, 1), (7, 11), (7, 9), (7, 8), (0, 2), (0, 6), (9, 10), (5, 6), (10, 9), (6, 3), (6, 2), (9, 0), (6, 7), (6, 9), (8, 3), (7, 5), (9, 7), (7, 7), (5, 11), (10, 10), (2, 4), (5, 0), (2, 0), (3, 7), (2, 8), (1, 8), (4, 1), (1, 6), (3, 11), (4, 4), (4, 9), (4, 8), (1, 5), (9, 3), (2, 3), (2, 2), (8, 1), (0, 10), (10, 0), (8, 6), (8, 9), (8, 11), (1, 2), (1, 4), (8, 5), (2, 6), (4, 11), (8, 0), (3, 3), (5, 7), (6, 10), (6, 5), (6, 6), (0, 5), (10, 5), (4, 7), (1, 1), (11, 10), (11, 4), (7, 6), (11, 7), (8, 7), (7, 3), (2, 10), (11, 5), (7, 10), (2, 11), (8, 4), (8, 2), (11, 1), (0, 7), (3, 5), (9, 9), (9, 6), (6, 0), (4, 3), (4, 0), (4, 5), (2, 7), (4, 6), (9, 11), (10, 1), (9, 1), (3, 8), (1, 9), (10, 8), (11, 6), (0, 0), (6, 8), (1, 11), (0, 3), (11, 11), (10, 7), (10, 6), (9, 8), (1, 7), (0, 8), (9, 5), (10, 11), (11, 0)]
    class_3_order_recall= [(10, 6), (11, 11), (11, 0), (9, 8), (8, 10), (4, 5), (10, 4), (7, 0), (9, 5), (3, 0), (7, 5), (2, 9), (5, 10), (8, 1), (8, 8), (2, 0), (5, 3), (2, 1), (0, 3), (6, 9), (3, 10), (1, 10), (6, 5), (8, 11), (7, 8), (1, 1), (4, 0), (3, 8), (6, 8), (8, 9), (3, 3), (0, 10), (1, 4), (2, 10), (5, 9), (6, 0), (1, 11), (1, 7), (0, 4), (3, 5), (5, 4), (5, 8), (2, 6), (4, 3), (10, 0), (0, 0), (0, 8), (10, 7)]
    class_3_order_precision= [(11, 6), (10, 7), (11, 5), (0, 0), (3, 4), (9, 3), (9, 0), (0, 1), (11, 9), (11, 3), (0, 7), (3, 5), (10, 0), (4, 3), (5, 4), (9, 11), (10, 3), (6, 6), (7, 9), (8, 6), (4, 1), (2, 2), (5, 11), (0, 10), (1, 6), (0, 8), (0, 11), (1, 2), (0, 2), (10, 1), (8, 7), (4, 4), (7, 11), (1, 4), (5, 7), (7, 7), (11, 2), (7, 4), (11, 1), (3, 10), (1, 10), (0, 6), (2, 6), (1, 0), (1, 9), (10, 8), (7, 6), (1, 8), (5, 1), (5, 0), (9, 4), (8, 0), (4, 9), (4, 8), (11, 10), (5, 2), (2, 11), (1, 1), (8, 11), (10, 9), (0, 3), (1, 3), (7, 3), (10, 5), (9, 10), (6, 3), (6, 4), (2, 4), (10, 10), (2, 8), (8, 5), (8, 8), (3, 9), (3, 8), (3, 7), (3, 6), (5, 8), (7, 8), (8, 2), (8, 3), (7, 0), (7, 2), (7, 1), (8, 4), (1, 11), (2, 7), (3, 3), (2, 5), (1, 5), (4, 5), (2, 9), (3, 0), (4, 10), (3, 11), (1, 7), (5, 5), (5, 6), (5, 10), (2, 10), (2, 0), (6, 7), (6, 8), (2, 3), (9, 2), (3, 1), (9, 1), (4, 0), (4, 7), (8, 1), (4, 11), (2, 1), (9, 6), (4, 6), (9, 9), (0, 4), (4, 2), (5, 3), (11, 7), (6, 2), (6, 10), (6, 0), (11, 8), (6, 11), (3, 2), (9, 7), (6, 9), (7, 10), (5, 9), (6, 1), (8, 9), (10, 11), (10, 4), (6, 5), (10, 2), (7, 5), (11, 4), (0, 9), (9, 8), (9, 5), (8, 10), (0, 5), (11, 0), (10, 6), (11, 11)]
    class_4_order_recall= [(9, 5), (10, 0), (7, 5), (2, 1), (8, 8), (0, 10), (1, 10), (3, 10), (3, 8), (5, 4), (4, 0), (5, 8), (0, 8), (0, 3), (8, 9), (2, 9), (8, 11), (1, 4), (0, 4), (1, 1), (4, 3), (6, 5), (9, 8), (5, 10), (5, 3), (2, 6), (2, 0), (2, 10), (4, 5), (7, 8), (8, 10), (5, 9), (6, 9), (7, 0), (3, 5), (6, 0), (1, 7), (3, 3), (1, 11), (10, 4), (6, 8), (0, 0), (8, 1), (3, 0), (11, 11), (10, 7), (10, 6), (11, 0)]
    class_4_order_precision= [(11, 0), (10, 6), (10, 7), (11, 11), (11, 6), (10, 3), (11, 4), (3, 0), (0, 5), (4, 0), (10, 1), (11, 8), (1, 11), (9, 1), (9, 10), (10, 4), (11, 1), (9, 8), (4, 7), (0, 6), (2, 10), (8, 3), (6, 9), (6, 1), (5, 10), (3, 10), (9, 9), (2, 9), (0, 0), (2, 0), (1, 7), (8, 6), (9, 0), (7, 8), (9, 3), (7, 11), (3, 3), (3, 2), (4, 5), (4, 11), (8, 4), (0, 9), (6, 5), (8, 1), (0, 1), (3, 9), (6, 0), (5, 11), (4, 10), (7, 6), (6, 2), (4, 8), (4, 9), (1, 8), (4, 3), (3, 11), (3, 4), (9, 6), (10, 11), (11, 5), (1, 4), (1, 3), (7, 7), (0, 10), (6, 8), (6, 3), (11, 7), (8, 0), (8, 5), (2, 5), (4, 1), (2, 6), (5, 7), (3, 5), (5, 1), (1, 10), (5, 3), (4, 2), (4, 4), (7, 9), (7, 10), (2, 3), (8, 10), (0, 7), (10, 5), (3, 7), (1, 5), (7, 4), (2, 1), (2, 4), (5, 5), (7, 3), (6, 6), (2, 8), (5, 2), (1, 6), (6, 4), (7, 0), (6, 10), (6, 11), (2, 7), (2, 11), (5, 9), (0, 2), (3, 8), (5, 0), (10, 10), (10, 9), (1, 9), (0, 8), (1, 2), (0, 3), (0, 11), (10, 2), (1, 1), (5, 8), (3, 1), (11, 10), (5, 6), (5, 4), (7, 1), (4, 6), (8, 9), (7, 2), (3, 6), (8, 11), (11, 2), (1, 0), (8, 2), (0, 4), (10, 8), (2, 2), (9, 2), (8, 7), (9, 4), (7, 5), (11, 3), (6, 7), (9, 11), (9, 7), (8, 8), (11, 9), (9, 5), (10, 0)]
    class_5_order_recall= [(9, 5), (10, 7), (10, 6), (11, 0), (10, 4), (3, 0), (7, 5), (4, 0), (2, 0), (2, 9), (1, 7), (1, 4), (0, 0), (3, 3), (1, 11), (5, 3), (2, 6), (5, 10), (4, 3), (0, 4), (7, 8), (7, 0), (0, 10), (1, 1), (2, 1), (3, 5), (5, 9), (8, 9), (0, 8), (6, 9), (5, 4), (8, 1), (5, 8), (2, 10), (4, 5), (6, 0), (3, 8), (8, 8), (3, 10), (1, 10), (6, 5), (6, 8), (0, 3), (8, 10), (8, 11), (9, 8), (10, 0), (11, 11)]
    class_5_order_precision= [(11, 11), (10, 0), (9, 1), (8, 11), (0, 5), (9, 8), (7, 8), (11, 9), (11, 3), (0, 8), (8, 8), (1, 10), (7, 10), (3, 6), (0, 3), (1, 2), (3, 10), (6, 5), (2, 6), (5, 8), (4, 1), (3, 4), (7, 0), (6, 9), (8, 2), (4, 6), (5, 5), (3, 8), (6, 1), (8, 9), (2, 3), (0, 4), (10, 10), (0, 10), (0, 11), (1, 0), (9, 7), (5, 4), (9, 3), (2, 2), (9, 2), (8, 0), (11, 5), (5, 9), (6, 0), (6, 3), (6, 4), (11, 4), (6, 6), (8, 7), (8, 4), (8, 3), (7, 1), (7, 2), (8, 10), (7, 3), (10, 8), (10, 5), (9, 9), (7, 9), (9, 6), (8, 1), (10, 9), (5, 11), (5, 2), (1, 4), (3, 7), (3, 5), (1, 1), (4, 3), (1, 5), (2, 10), (2, 11), (4, 11), (2, 4), (2, 7), (0, 6), (5, 0), (1, 6), (0, 7), (3, 2), (1, 8), (5, 3), (1, 9), (2, 8), (8, 5), (7, 7), (7, 6), (3, 1), (8, 6), (10, 2), (4, 8), (5, 1), (5, 10), (11, 2), (11, 1), (6, 2), (10, 1), (4, 7), (2, 1), (0, 9), (6, 7), (6, 8), (6, 11), (4, 2), (10, 11), (1, 7), (11, 8), (9, 4), (1, 11), (2, 9), (4, 10), (4, 4), (4, 5), (4, 9), (3, 11), (3, 9), (7, 4), (2, 0), (6, 10), (3, 3), (7, 11), (3, 0), (5, 6), (2, 5), (5, 7), (0, 2), (11, 0), (9, 0), (0, 0), (4, 0), (10, 4), (9, 11), (7, 5), (1, 3), (11, 10), (0, 1), (10, 3), (9, 10), (11, 6), (11, 7), (10, 7), (10, 6), (9, 5)]
    class_6_order_recall= [(11, 11), (10, 6), (9, 5), (1, 11), (7, 5), (6, 8), (3, 8), (10, 7), (1, 10), (5, 9), (2, 6), (9, 8), (11, 0), (5, 8), (3, 10), (1, 7), (3, 3), (3, 0), (8, 10), (8, 9), (8, 8), (6, 5), (7, 8), (1, 4), (2, 10), (10, 0), (1, 1), (6, 9), (5, 3), (5, 4), (6, 0), (0, 3), (4, 3), (4, 0), (8, 1), (5, 10), (4, 5), (2, 9), (0, 10), (8, 11), (7, 0), (2, 1), (3, 5), (2, 0), (0, 4), (0, 0), (10, 4), (0, 8)]
    class_6_order_precision= [(10, 4), (0, 5), (4, 0), (0, 10), (0, 2), (2, 7), (9, 11), (5, 8), (11, 4), (0, 8), (0, 9), (1, 8), (4, 2), (3, 9), (9, 4), (0, 0), (8, 11), (7, 10), (7, 8), (6, 5), (3, 11), (2, 2), (3, 5), (1, 10), (11, 5), (1, 11), (1, 0), (11, 3), (4, 7), (8, 8), (4, 10), (8, 6), (8, 5), (8, 4), (5, 2), (5, 4), (8, 1), (5, 10), (11, 10), (11, 6), (0, 3), (7, 7), (11, 8), (7, 2), (6, 7), (4, 1), (1, 2), (9, 2), (3, 3), (1, 4), (2, 6), (2, 9), (9, 7), (1, 3), (1, 9), (1, 7), (2, 3), (2, 4), (2, 8), (10, 10), (10, 1), (3, 0), (10, 2), (11, 1), (7, 6), (10, 5), (10, 7), (7, 1), (7, 9), (10, 9), (8, 2), (8, 0), (10, 3), (8, 3), (8, 7), (9, 10), (7, 0), (8, 9), (9, 9), (9, 3), (9, 6), (7, 11), (6, 11), (5, 11), (6, 9), (3, 2), (3, 7), (6, 10), (3, 1), (2, 11), (4, 4), (2, 10), (2, 5), (4, 5), (2, 1), (4, 6), (4, 9), (2, 0), (3, 6), (5, 1), (5, 0), (5, 5), (0, 4), (0, 11), (1, 1), (6, 4), (5, 3), (6, 2), (6, 3), (1, 5), (5, 9), (1, 6), (5, 6), (6, 1), (3, 4), (11, 9), (0, 7), (10, 0), (0, 6), (3, 10), (3, 8), (6, 6), (7, 3), (7, 4), (7, 5), (5, 7), (4, 11), (6, 0), (9, 0), (11, 7), (4, 8), (0, 1), (10, 8), (8, 10), (10, 11), (11, 0), (11, 2), (9, 1), (4, 3), (9, 8), (6, 8), (9, 5), (11, 11), (10, 6)]
    class_7_order_recall= [(11, 11), (9, 8), (8, 11), (8, 9), (3, 8), (7, 0), (4, 0), (2, 0), (6, 8), (2, 10), (0, 8), (3, 5), (6, 5), (0, 3), (5, 10), (8, 10), (2, 9), (8, 8), (3, 10), (5, 4), (1, 1), (1, 10), (7, 5), (3, 0), (4, 3), (5, 3), (2, 6), (6, 0), (4, 5), (8, 1), (5, 9), (0, 4), (10, 0), (2, 1), (3, 3), (5, 8), (6, 9), (0, 10), (7, 8), (1, 7), (1, 4), (11, 0), (0, 0), (1, 11), (10, 6), (10, 4), (9, 5), (10, 7)]
    class_7_order_precision= [(11, 6), (10, 7), (11, 7), (9, 5), (9, 11), (11, 0), (8, 7), (11, 3), (11, 10), (10, 4), (5, 0), (7, 8), (9, 6), (1, 11), (5, 5), (6, 4), (1, 1), (10, 6), (6, 7), (11, 1), (10, 3), (4, 11), (3, 1), (8, 1), (3, 3), (8, 2), (3, 11), (2, 5), (0, 0), (5, 8), (5, 7), (1, 10), (7, 1), (9, 10), (1, 4), (10, 2), (0, 10), (5, 1), (10, 11), (5, 9), (3, 5), (5, 10), (8, 6), (7, 6), (7, 9), (7, 10), (8, 0), (8, 4), (8, 5), (9, 0), (9, 4), (9, 9), (10, 10), (11, 2), (6, 10), (4, 8), (5, 11), (1, 0), (3, 4), (4, 6), (4, 5), (4, 4), (2, 3), (2, 4), (4, 7), (1, 5), (1, 6), (7, 2), (7, 3), (10, 0), (7, 5), (2, 11), (4, 9), (7, 11), (6, 11), (2, 8), (2, 7), (8, 3), (1, 9), (8, 8), (1, 8), (2, 9), (3, 6), (6, 8), (4, 10), (5, 2), (5, 3), (5, 4), (4, 3), (5, 6), (11, 9), (4, 1), (11, 8), (9, 3), (9, 1), (0, 4), (6, 2), (6, 3), (0, 6), (10, 5), (3, 7), (0, 3), (1, 7), (0, 11), (1, 3), (2, 0), (11, 4), (0, 2), (2, 6), (6, 1), (3, 9), (4, 2), (6, 6), (6, 9), (3, 8), (7, 7), (3, 2), (6, 0), (0, 8), (10, 8), (3, 10), (2, 2), (8, 10), (9, 2), (3, 0), (11, 5), (1, 2), (0, 1), (4, 0), (2, 1), (8, 9), (10, 9), (7, 4), (8, 11), (6, 5), (0, 7), (10, 1), (2, 10), (0, 5), (7, 0), (0, 9), (9, 7), (9, 8), (11, 11)]
    class_8_order_recall= [(10, 6), (11, 0), (0, 10), (3, 5), (4, 3), (10, 0), (5, 10), (1, 4), (5, 4), (6, 0), (6, 9), (3, 10), (0, 4), (0, 8), (8, 8), (2, 0), (7, 0), (1, 10), (2, 1), (1, 1), (2, 9), (0, 3), (5, 9), (0, 0), (1, 7), (3, 3), (5, 8), (10, 7), (6, 5), (4, 5), (8, 1), (8, 9), (8, 11), (5, 3), (6, 8), (2, 10), (7, 8), (7, 5), (8, 10), (4, 0), (3, 8), (2, 6), (3, 0), (1, 11), (9, 8), (9, 5), (10, 4), (11, 11)]
    class_8_order_precision= [(10, 4), (9, 5), (11, 11), (7, 5), (4, 0), (11, 4), (9, 8), (9, 7), (2, 10), (3, 0), (10, 1), (0, 9), (6, 5), (4, 2), (0, 5), (2, 6), (9, 1), (3, 8), (3, 10), (1, 7), (11, 7), (9, 9), (9, 11), (11, 5), (8, 6), (2, 9), (2, 8), (2, 7), (8, 9), (8, 11), (1, 9), (8, 1), (7, 10), (3, 6), (6, 4), (6, 1), (3, 7), (8, 4), (9, 4), (0, 3), (10, 7), (8, 10), (1, 2), (5, 7), (2, 0), (7, 6), (7, 8), (1, 11), (9, 2), (7, 3), (9, 3), (7, 1), (8, 2), (6, 10), (6, 8), (6, 7), (10, 8), (6, 2), (5, 9), (5, 8), (5, 11), (4, 7), (3, 1), (0, 7), (4, 5), (7, 4), (7, 2), (3, 9), (10, 9), (0, 6), (10, 5), (3, 2), (7, 11), (8, 5), (8, 8), (9, 0), (1, 5), (3, 3), (6, 11), (1, 3), (0, 1), (11, 1), (5, 2), (6, 3), (0, 2), (6, 0), (1, 10), (0, 8), (8, 0), (4, 10), (4, 9), (0, 10), (4, 6), (7, 0), (6, 6), (6, 9), (1, 8), (1, 6), (5, 3), (2, 1), (2, 2), (1, 4), (0, 11), (10, 11), (5, 6), (10, 2), (8, 3), (2, 11), (4, 11), (4, 8), (1, 1), (10, 10), (4, 1), (2, 4), (0, 4), (11, 8), (5, 1), (5, 4), (2, 3), (5, 5), (8, 7), (5, 0), (11, 10), (7, 9), (7, 7), (1, 0), (11, 3), (3, 5), (3, 11), (3, 4), (0, 0), (9, 6), (11, 0), (2, 5), (5, 10), (10, 0), (4, 3), (4, 4), (11, 2), (10, 3), (9, 10), (10, 6), (11, 9), (11, 6)]
    class_9_order_recall= [(11, 11), (10, 4), (9, 5), (6, 8), (0, 0), (2, 9), (10, 7), (8, 8), (7, 5), (1, 10), (3, 10), (4, 5), (5, 8), (5, 4), (6, 5), (2, 0), (5, 10), (0, 10), (1, 4), (10, 0), (2, 10), (8, 11), (3, 5), (0, 3), (1, 11), (5, 9), (5, 3), (7, 8), (9, 8), (2, 6), (1, 1), (0, 4), (3, 0), (3, 3), (8, 1), (6, 9), (4, 3), (2, 1), (7, 0), (3, 8), (1, 7), (0, 8), (6, 0), (8, 9), (4, 0), (8, 10), (11, 0), (10, 6)]
    class_9_order_precision= [(8, 10), (10, 6), (11, 2), (11, 0), (11, 7), (0, 8), (4, 3), (7, 0), (9, 8), (4, 0), (4, 6), (8, 9), (4, 10), (11, 10), (5, 9), (6, 0), (1, 4), (11, 9), (0, 3), (4, 1), (9, 6), (3, 11), (5, 3), (2, 3), (5, 1), (3, 0), (8, 4), (5, 10), (1, 0), (4, 5), (4, 7), (3, 8), (7, 10), (5, 0), (7, 6), (8, 1), (7, 7), (7, 4), (2, 11), (10, 11), (10, 0), (0, 10), (1, 1), (1, 6), (1, 7), (9, 4), (1, 10), (1, 11), (2, 2), (2, 6), (8, 6), (5, 5), (7, 1), (5, 7), (7, 3), (6, 9), (4, 11), (10, 9), (6, 3), (7, 8), (10, 5), (0, 11), (9, 7), (4, 8), (2, 10), (2, 1), (8, 0), (3, 6), (8, 11), (8, 2), (6, 10), (9, 1), (9, 10), (7, 9), (11, 4), (8, 5), (5, 11), (3, 9), (3, 7), (2, 0), (3, 2), (1, 2), (1, 5), (6, 4), (11, 8), (2, 8), (3, 4), (10, 8), (5, 6), (6, 1), (6, 2), (0, 2), (2, 5), (11, 1), (9, 0), (8, 3), (2, 7), (6, 5), (7, 11), (3, 10), (4, 2), (10, 2), (5, 2), (7, 2), (6, 11), (0, 4), (6, 6), (3, 5), (1, 9), (0, 7), (10, 1), (5, 4), (9, 2), (2, 4), (2, 9), (5, 8), (4, 9), (3, 1), (4, 4), (0, 1), (7, 5), (3, 3), (9, 11), (1, 8), (8, 7), (9, 9), (1, 3), (6, 7), (10, 10), (9, 3), (0, 0), (6, 8), (8, 8), (0, 9), (10, 7), (11, 3), (0, 6), (0, 5), (9, 5), (11, 5), (10, 3), (10, 4), (11, 6), (11, 11)]
    class_10_order_recall= [(10, 7), (9, 5), (11, 0), (10, 4), (7, 5), (1, 7), (2, 0), (1, 4), (2, 6), (4, 5), (4, 0), (6, 8), (5, 4), (10, 0), (0, 0), (1, 1), (1, 10), (0, 10), (6, 9), (5, 8), (5, 3), (2, 9), (3, 10), (1, 11), (0, 4), (3, 0), (5, 9), (3, 5), (7, 8), (3, 3), (4, 3), (8, 1), (5, 10), (0, 3), (3, 8), (6, 5), (2, 1), (8, 11), (2, 10), (0, 8), (6, 0), (8, 9), (8, 10), (11, 11), (9, 8), (8, 8), (7, 0), (10, 6)]
    class_10_order_precision= [(10, 6), (0, 9), (11, 11), (8, 9), (0, 5), (7, 0), (8, 11), (11, 8), (11, 5), (9, 2), (6, 5), (10, 2), (2, 5), (1, 10), (9, 7), (9, 8), (1, 5), (10, 1), (0, 7), (4, 0), (10, 3), (11, 10), (3, 11), (0, 2), (8, 8), (7, 8), (6, 2), (3, 3), (7, 4), (4, 6), (3, 5), (7, 1), (3, 7), (4, 1), (7, 11), (2, 1), (2, 9), (4, 10), (5, 7), (8, 6), (8, 10), (6, 6), (2, 0), (9, 10), (10, 5), (0, 11), (5, 10), (10, 10), (6, 0), (4, 3), (6, 1), (5, 8), (8, 5), (6, 11), (7, 3), (7, 7), (7, 9), (8, 0), (8, 3), (9, 3), (10, 9), (11, 2), (6, 10), (5, 6), (0, 0), (4, 4), (1, 7), (2, 3), (2, 7), (4, 5), (1, 1), (4, 7), (3, 8), (4, 8), (0, 6), (0, 4), (5, 1), (5, 3), (0, 8), (3, 9), (8, 4), (2, 8), (8, 2), (2, 4), (9, 4), (9, 6), (1, 6), (2, 10), (10, 8), (11, 3), (11, 4), (1, 9), (7, 10), (5, 11), (3, 0), (7, 6), (5, 2), (6, 3), (7, 2), (1, 3), (6, 9), (1, 0), (0, 10), (1, 8), (3, 10), (4, 11), (1, 11), (2, 2), (10, 11), (8, 7), (11, 1), (5, 0), (5, 9), (0, 3), (3, 2), (11, 9), (6, 4), (10, 0), (3, 1), (1, 4), (9, 9), (8, 1), (2, 6), (9, 0), (4, 2), (5, 4), (5, 5), (6, 7), (2, 11), (0, 1), (3, 4), (3, 6), (6, 8), (4, 9), (9, 11), (1, 2), (11, 7), (9, 5), (11, 6), (7, 5), (11, 0), (9, 1), (10, 4), (10, 7)]

    class_order_recall = [class_1_order_recall,
                            class_2_order_recall,
                            class_3_order_recall,
                            class_4_order_recall,
                            class_5_order_recall,
                            class_6_order_recall,
                            class_7_order_recall,
                            class_8_order_recall,
                            class_9_order_recall,
                            class_10_order_recall]


    class_order_precision = [class_1_order_precision, 
                            class_2_order_precision, 
                            class_3_order_precision, 
                            class_4_order_precision, 
                            class_5_order_precision, 
                            class_6_order_precision, 
                            class_7_order_precision, 
                            class_8_order_precision, 
                            class_9_order_precision, 
                            class_10_order_precision]

    class_1_head = []
    class_2_head = []
    class_3_head = []
    class_4_head = []
    class_5_head = []
    class_6_head = []
    class_7_head = []
    class_8_head = []
    class_9_head = []
    class_10_head = []

    class_head = [class_1_head, 
                class_2_head, 
                class_3_head, 
                class_4_head, 
                class_5_head, 
                class_6_head, 
                class_7_head, 
                class_8_head, 
                class_9_head, 
                class_10_head]

    for cl in range(10):
        for i in range(ite):
            class_head[cl].append(class_order_recall[cl][i])

        for i in range(5):
            class_head[cl].append(class_order_precision[cl][i])

    should_be_remain_head = []
    for j in range(10):
        # print(f'\nclass {j + 1}')
        # print('original_head: ', class_head[j])
        # print('original_head_num: ', len(class_head[j]))


        check = [0 for _ in range(12)]
        for x, y in class_head[j]:
            check[x] += 1
        # print(check)
        # print(len(class_head[j]))


        for i in range(12):
            if check[i] == 0: # 만약에 해당 층에 아무것도 없으면 여기에서 걸림

                cc = True
                q = 0
                while cc:
                    if class_order_recall[j][q] not in class_head[j] and class_order_recall[j][q][0] == i:
                        class_head[j].append(class_order_recall[j][q])
                        check[i] += 1
                        cc = False # 해당 층에 헤드 추가하면 루프 탈출
                    q += 1 # 추가 되든 말든 계속해서 다음 헤드로 넘어감



        # print(check)

        should_be_remain_head.append(class_head[j])
        
        # print('should_be_ramain_head: ', class_head[j])
        # print(f'num of head for class {j+1}: ', len(class_head[j]))



    all_tuples = {(i, j) for i in range(12) for j in range(12)}
    should_be_removed_head_10_0 = []
    for i in range(10):
        should_be_removed_head_10_0.append(list(all_tuples - set(should_be_remain_head[i])))
        # print(f'num head that should_be_removed_head_class_{i+1}: ', len(should_be_removed_head_10_0[i]))

    # print('\n')
    # for i in range(10):
    #     print(f'should_be_removed_head_class_{i+1}=', should_be_removed_head_10_0[i])
#####======####======####======####======####======####======####======####======
#####======####======####======####======####======####======####======####======




    # ================================================================================================
    # ================================================================================================
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_1 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_2 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_3 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_4 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_5 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_6 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_7 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_8 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_9 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model_10 = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")

    model_1 = model_1.to(device)
    model_2 = model_2.to(device)
    model_3 = model_3.to(device)
    model_4 = model_4.to(device)
    model_5 = model_5.to(device)
    model_6 = model_6.to(device)
    model_7 = model_7.to(device)
    model_8 = model_8.to(device)
    model_9 = model_9.to(device)
    model_10 = model_10.to(device)

    models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8, model_9, model_10]

    rkrk = 0
    for i in range(10): # pruning
        pruned_head_num = 0
        for layer_index, head_index in should_be_removed_head_10_0[i]:
            models[i].bert.encoder.layer[layer_index].attention.prune_heads([head_index])
            pruned_head_num += 1

        print('class ', i + 1, ' pruned_head_num: ', pruned_head_num, 'remaining head num: ', 12 * 12 - pruned_head_num)
        rkrk += 12 * 12 - pruned_head_num
    print(rkrk/10/144 * 100, '%')

    def find_index_per_column(logits_matrix, true_table_matrix):
        # 결과를 저장할 리스트 초기화
        result_indices = []
        
        # 열(column) 별로 반복
        for column in range(logits_matrix.shape[1]):
            # 현재 열에 대한 logits와 true_table 추출
            logits = logits_matrix[:, column]
            true_table = true_table_matrix[:, column]
            
            true_indices = [i for i, value in enumerate(true_table) if value]
            # 3. 모두 다 False인 경우
            if not true_indices:
                result_indices.append(np.argmax(logits) + 1)
                continue

            # 1. True인 인덱스가 한 개만 존재할 경우
            if len(true_indices) == 1:
                result_indices.append(true_indices[0] + 1)
                continue

            # 2. True인 인덱스가 2개 이상 존재할 경우
            max_value = float('-inf')
            max_index = None
            for index in true_indices:
                if logits[index] > max_value:
                    max_value = logits[index]
                    max_index = index
            result_indices.append(max_index + 1)

        return result_indices

    # ================================================================================================
    # 모델 평가

    correct_num = []
    correct_num_neg = []
    accs = [[], [], [], [], [], [], [], [], [], []]
    preds = []
    true_labels = []
    preds_rp =[[], [], [], [], [], [], [], [], [], []]


    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = batch
        inputs = {k: v.squeeze(1).to(device) for k, v in inputs.items()} 
        labels = labels.to(device)

        with torch.no_grad():
            outputs_1 = model_1(**inputs)
            outputs_2 = model_2(**inputs)
            outputs_3 = model_3(**inputs)
            outputs_4 = model_4(**inputs)
            outputs_5 = model_5(**inputs)
            outputs_6 = model_6(**inputs)
            outputs_7 = model_7(**inputs)
            outputs_8 = model_8(**inputs)
            outputs_9 = model_9(**inputs)
            outputs_10 = model_10(**inputs)

        logit1 = outputs_1.logits[:, 0].tolist() # 1번 클래스 이진분류기의, 1번 클래스에 대한 로짓값 (batch_size만큼 나옴)
        logit2 = outputs_2.logits[:, 1].tolist() # 2번 클래스 이진분류기의, 2번 클래스에 대한 로짓값
        logit3 = outputs_3.logits[:, 2].tolist() # 3번 클래스 이진분류기의, 3번 클래스에 대한 로짓값
        logit4 = outputs_4.logits[:, 3].tolist() # 4번 클래스 이진분류기의, 4번 클래스에 대한 로짓값
        logit5 = outputs_5.logits[:, 4].tolist() # 5번 클래스 이진분류기의, 5번 클래스에 대한 로짓값
        logit6 = outputs_6.logits[:, 5].tolist() # 6번 클래스 이진분류기의, 6번 클래스에 대한 로짓값
        logit7 = outputs_7.logits[:, 6].tolist() # 7번 클래스 이진분류기의, 7번 클래스에 대한 로짓값
        logit8 = outputs_8.logits[:, 7].tolist() # 8번 클래스 이진분류기의, 8번 클래스에 대한 로짓값
        logit9 = outputs_9.logits[:, 8].tolist() # 9번 클래스 이진분류기의, 9번 클래스에 대한 로짓값
        logit10 = outputs_10.logits[:, 9].tolist() # 10번 클래스 이진분류기의, 10번 클래스에 대한 로짓값

        prediction_1 = (outputs_1.logits.argmax(dim=-1) + 1) == 1
        prediction_2 = (outputs_2.logits.argmax(dim=-1) + 1) == 2
        prediction_3 = (outputs_3.logits.argmax(dim=-1) + 1) == 3
        prediction_4 = (outputs_4.logits.argmax(dim=-1) + 1) == 4
        prediction_5 = (outputs_5.logits.argmax(dim=-1) + 1) == 5
        prediction_6 = (outputs_6.logits.argmax(dim=-1) + 1) == 6
        prediction_7 = (outputs_7.logits.argmax(dim=-1) + 1) == 7
        prediction_8 = (outputs_8.logits.argmax(dim=-1) + 1) == 8
        prediction_9 = (outputs_9.logits.argmax(dim=-1) + 1) == 9
        prediction_10 = (outputs_10.logits.argmax(dim=-1) + 1) == 10

        preds_rp[0].extend((outputs_1.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[1].extend((outputs_2.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[2].extend((outputs_3.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[3].extend((outputs_4.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[4].extend((outputs_5.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[5].extend((outputs_6.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[6].extend((outputs_7.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[7].extend((outputs_8.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[8].extend((outputs_9.logits.argmax(dim=-1) + 1).tolist())
        preds_rp[9].extend((outputs_10.logits.argmax(dim=-1) + 1).tolist())

        logit_matrix = np.array([logit1, logit2, logit3, logit4, logit5, logit6, logit7, logit8, logit9, logit10])
        prediction_matrix = np.array([prediction_1.tolist(), prediction_2.tolist(), prediction_3.tolist(), prediction_4.tolist(), prediction_5.tolist(), prediction_6.tolist(), prediction_7.tolist(), prediction_8.tolist(), prediction_9.tolist(), prediction_10.tolist()])

        prediction = find_index_per_column(logit_matrix, prediction_matrix)

        preds.extend(prediction)
        true_labels.extend(labels.tolist())


        # print(preds)
        # print(true_labels)
        
    # print('classification report')
    # print(classification_report(true_labels, preds, digits=5))

    # print('confusion matrix')
    # print(confusion_matrix(true_labels, preds))

    con = confusion_matrix(true_labels, preds)
    con_diag = np.diag(con) # numpy array
    con_vertical = np.sum(con, axis=0) # numpy array

    for i in range(10):
        accs[i].append(con_diag[i] / 30)

    for i in range(10): # 10개의 클래스에 대해서 recall과 precision을 출력
        print('class:', i + 1)
        con_i = np.array(confusion_matrix(true_labels, preds_rp[i]))

        print(f'recall: {con_i[i, i] / np.sum(con_i[i, :]) * 100 :.2f}')
        print(f'precision: {con_i[i, i] / np.sum(con_i[:, i]) * 100 :.2f}')
        print('')


    print('global accuracy')
    print(np.array(accs))
    # ================================================================================================
    # ================================================================================================
