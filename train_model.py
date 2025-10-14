# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# 1) Load cleaned dataset (must be produced by preprocess_dataset.py)
df = pd.read_csv("data_train_clean.csv")   # <- use cleaned file only

if df.empty:
    raise ValueError("data_train_clean.csv is empty. Check preprocessing step.")

# Encode labels
label2id = {"Positive":0, "Negative":1, "Risky":2}
id2label = {v:k for k,v in label2id.items()}
df['label_id'] = df['label'].map(label2id)
if df['label_id'].isnull().any():
    raise ValueError("Some labels could not be mapped to label2id. Check df['label'] unique values.")

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42, stratify=df['label_id']
)

# 2) Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize with explicit settings (max_length, truncation, padding)
max_len = 128
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_len)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_len)

# Convert labels to tensors
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

# 3) Training arguments: use evaluation per epoch for simplicity
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    dataloader_pin_memory=False,
    remove_unused_columns=True,
    warmup_steps=50,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",   # changed from "steps" to "epoch"
    save_strategy="epoch",
    save_total_limit=1
)

# 4) Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
)

# 5) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 6) Train
trainer.train()

# 7) Evaluate
preds = trainer.predict(val_dataset)
pred_labels = preds.predictions.argmax(-1)
print(classification_report(val_labels, pred_labels, target_names=list(label2id.keys())))

# 8) Save model and tokenizer
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")

print("✅ Model training complete and saved in 'trained_model/' folder")
