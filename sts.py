import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# 1. 데이터 로드 및 전처리
class KoreanSTSDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1 = self.data.iloc[idx]['sentence1']
        sentence2 = self.data.iloc[idx]['sentence2']
        score = self.data.iloc[idx]['score']

        inputs = self.tokenizer(
            sentence1, sentence2,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': inputs['token_type_ids'].squeeze(0),
            'score': torch.tensor(score, dtype=torch.float)
        }

# 2. EDA 기법 적용
def augment_data(data):
    def swap_words(sentence):
        words = sentence.split()
        if len(words) > 1:
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    augmented_data = []
    for _, row in data.iterrows():
        augmented_data.append(row)
        if np.random.rand() < 0.3:  # 30% 확률로 데이터 증강
            row_copy = row.copy()
            row_copy['sentence1'] = swap_words(row['sentence1'])
            row_copy['sentence2'] = swap_words(row['sentence2'])
            augmented_data.append(row_copy)

    return pd.DataFrame(augmented_data)

# 3. STS Task를 위한 모델 및 Loss 정의
class STSModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(STSModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        score = self.regressor(cls_output)
        return score

# 4. 학습 및 검증
def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            scores = batch['score'].to(device)

            predictions = model(input_ids, attention_mask, token_type_ids).squeeze(-1)
            loss = loss_fn(predictions, scores)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader)}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                scores = batch['score'].to(device)

                predictions = model(input_ids, attention_mask, token_type_ids).squeeze(-1)
                loss = loss_fn(predictions, scores)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}")

# 실행 부분
if __name__ == "__main__":
    # 데이터 로드
    data = pd.read_csv('korean_sts_data.csv')  # sentence1, sentence2, score 컬럼 포함

    # 데이터 증강
    data = augment_data(data)

    # 데이터 분리
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Tokenizer 및 Dataset 준비
    tokenizer = BertTokenizer.from_pretrained('beomi/KcELECTRA-base')
    max_len = 128
    train_dataset = KoreanSTSDataset(train_data, tokenizer, max_len)
    val_dataset = KoreanSTSDataset(val_data, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STSModel(pretrained_model_name='beomi/KcELECTRA-base')

    # Optimizer, Scheduler, Loss
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 5  # 5 epochs
    scheduler = get_scheduler('linear', optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    loss_fn = nn.MSELoss()

    # 학습 및 평가
    train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, epochs=5)
