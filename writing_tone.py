!pip install transformers datasets scikit-learn --quiet

# importing necessary libraries
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)
import random
import os


# Load the dataset
df = pd.read_csv("writing_tone_dataset.csv")

# Basic cleanup
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Shuffle
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# View sample
print(df.head())
print(df['style'].value_counts())

# label encoding
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['style'])

label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

num_labels = len(label2id)
print("Classes:", label2id)

# Checking how many samples remain per class
print(df['label'].value_counts())

# Filter only those classes with >= 50 samples 
min_required = 50
filtered_df = df.groupby('label').filter(lambda x: len(x) >= min_required)

# Train-test split
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    filtered_df,
    test_size=0.2,
    stratify=filtered_df['label'],
    random_state=42
)

# Extracting text and label lists
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

# Final check
print("Train size:", len(train_texts))
print("Test size:", len(test_texts))
print("Overlap after fix:", len(set(train_texts) & set(test_texts)))

# Tokenization
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    encodings['labels'] = labels
    return encodings

train_enc = tokenize(train_texts, train_labels)
test_enc = tokenize(test_texts, test_labels)

# Creating Dataset and DataLoader
import torch
from torch.utils.data import Dataset, DataLoader

class StyleDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        return len(self.encodings["input_ids"])
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

train_dataset = StyleDataset(train_enc)
test_dataset = StyleDataset(test_enc)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Loading the model
from transformers import RobertaForSequenceClassification

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training the model
from transformers import get_scheduler
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
# Training loop
from tqdm.notebook import tqdm

model.train()
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    total_loss = 0
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

# Evaluation
from sklearn.metrics import classification_report

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# prediction function
def predict_style(texts):
    if isinstance(texts, str):
        texts = [texts]  # wrap in list if single example

    # Tokenize
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

    # Decode
    predicted_labels = [id2label[p.item()] for p in preds]
    return predicted_labels

# Test with a single sentence
print(predict_style("I understand how tough this must be for you. Stay strong."))

# Test with a batch of examples
custom_texts = [
    "Please follow the instructions carefully to install the software.",
    "Roses are red, violets are blue, I'm debugging code and so are you.",
    "This analysis is based on the hypothesis that X influences Y.",
    "Try our product free for 30 days and see the difference!",
    "I'm truly sorry you're dealing with that right now.",
    "Heyy I dont know is this correct or not.",
    "heyy I think you should use transformers libarary and roberta model.",
    "I think we can complete this project on time."
]

predicted = predict_style(custom_texts)
for text, label in zip(custom_texts, predicted):
    print(f"\nText: {text}\n→ Predicted Style: {label}")








