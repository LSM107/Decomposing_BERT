from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertModel
from tokenizers import BertWordPieceTokenizer


import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support



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

all_heads = [(i, j) for i in range(12) for j in range(12)]

Importance_heads = [(0, 8), (4, 5), (0, 3), (6, 5), (5, 8), (0, 4), (0, 10), (5, 3), (3, 8), (6, 0), (3, 0), (3, 10), (2, 1), (2, 6), (1, 7), (1, 1), (10, 0), (4, 3), (10, 7), (3, 3), (5, 10), (2, 10), (8, 8), (6, 9), (0, 0), (8, 9), (8, 1), (1, 11), (5, 4), (9, 8), (2, 0), (8, 10), (3, 5), (5, 9), (2, 9), (7, 5), (7, 8), (1, 10), (11, 0), (1, 4), (10, 4), (6, 8), (8, 11), (4, 0), (10, 6), (7, 0), (11, 11), (9, 5)]
Importance_heads = sorted(Importance_heads)
print(len(Importance_heads))
pruning_heads = sorted(list(set(all_heads) - set(Importance_heads)))

# 결과를 저장할 딕셔너리, 리스트 생성
class_recall = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
class_precision = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

# 공통 헤드만 제거했을 때의 recall, precision을 계산
model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
model = model.to(device)

# 공통적으로 제거해야 하는 헤드들 제거하기
for l, h in pruning_heads:
    model.bert.encoder.layer[l].attention.prune_heads([h])


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


precision, recall, _, _ = precision_recall_fscore_support(true_labels, preds, labels=range(1, 11), average=None)


# 리스트에 프리시전과 리콜 값 저장
precision_original = precision.tolist()
recall_original = recall.tolist()



for layer_index, head_index in Importance_heads:
    print('\n====================================================')
    print(f'Layer: {layer_index}, Head: {head_index}')
    print('----------------------------------------------------\n')

    # 모델을 새롭게 정의함
    model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-yahoo_answers_topics")
    model = model.to(device)


    # 공통적으로 제거해야 하는 헤드들 제거하기
    for l, h in pruning_heads:
        model.bert.encoder.layer[l].attention.prune_heads([h])

    # 조사할 헤드 제거하기
    model.bert.encoder.layer[layer_index].attention.prune_heads([head_index])

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

    # preds와 true_labels를 사용하여 각 클래스에 대한 프리시전, 리콜 계산
    precision, recall, _, _ = precision_recall_fscore_support(true_labels, preds, labels=range(1, 11), average=None)

    # 리스트에 프리시전과 리콜 값 저장
    precision_per_class = precision.tolist()
    recall_per_class = recall.tolist()

    for k in range(10):
        class_recall[k][(layer_index, head_index)] = recall_per_class[k] - recall_original[k]
        class_precision[k][(layer_index, head_index)] = precision_per_class[k] - precision_original[k]



for t in range(10):
    class_recall[t] = sorted(class_recall[t].items(), key=lambda x: x[1], reverse=True)
    class_precision[t] = sorted(class_precision[t].items(), key=lambda x: x[1], reverse=True)

    print(f'\nClass {t+1} Recall')
    print(class_recall[t])

    print(f'\nClass {t+1} Precision')
    print(class_precision[t])
