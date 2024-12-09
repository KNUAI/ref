import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 1. 기본적인 한국어 처리: 데이터 로드 및 전처리
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df.dropna(subset=["sentence1", "sentence2", "label"])
    df = df[df['label'] <= 5]  # STS label은 0-5 사이
    return df

# 2. EDA 기법을 적용하여 데이터 증강
def augment_data(df):
    augmented = []
    for idx, row in df.iterrows():
        # Swap sentence1 and sentence2
        swapped_row = {"sentence1": row["sentence2"], "sentence2": row["sentence1"], "label": row["label"]}
        augmented.append(swapped_row)
    augmented_df = pd.DataFrame(augmented)
    return pd.concat([df, augmented_df], ignore_index=True)

# 3. Dataset 클래스 정의
class STSDataset(Dataset):
    def __init__(self, sentences1, sentences2, labels, tokenizer, max_len):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences1)

    def __getitem__(self, index):
        sentence1 = str(self.sentences1[index])
        sentence2 = str(self.sentences2[index])
        label = self.labels[index]

        inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float)
        }

# 4. STS task 수행을 위한 loss function 정의
def custom_loss(output, target):
    mse_loss = torch.nn.MSELoss()
    return mse_loss(output.squeeze(-1), target)

# 5. Pretrained Model을 사용하여 STSModel 구현
def train_sts_model():
    # Load dataset
    file_path = "klue-sts-v1.1/klue-sts-v1.1_train.tsv"  # Adjust the path
    df = load_and_preprocess_data(file_path)
    df = augment_data(df)

    train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels = train_test_split(
        df['sentence1'], df['sentence2'], df['label'], test_size=0.2, random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    train_dataset = STSDataset(train_texts1.tolist(), train_texts2.tolist(), train_labels.tolist(), tokenizer, max_len=128)
    val_dataset = STSDataset(val_texts1.tolist(), val_texts2.tolist(), val_labels.tolist(), tokenizer, max_len=128)

    # Model setup
    model = AutoModelForSequenceClassification.from_pretrained("klue/roberta-base", num_labels=1)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    def compute_metrics(pred):
        preds = pred.predictions.squeeze(-1)
        labels = pred.label_ids
        pearson_corr = np.corrcoef(preds, labels)[0, 1]
        return {"pearson_corr": pearson_corr}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained("./sts_model")
    tokenizer.save_pretrained("./sts_model")

    print("Model training and saving complete.")

# 실행
if __name__ == "__main__":
    train_sts_model()
