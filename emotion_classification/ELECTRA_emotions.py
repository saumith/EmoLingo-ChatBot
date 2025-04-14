# import necessary libraries
import random
import pandas as pd
import numpy as np
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report, hamming_loss
from peft import LoraConfig, get_peft_model, TaskType
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator


# Device setup
accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")

# Globals
model = None
tokenizer = None
emotion_labels = None
# Augmentation Functions (Simple version)
def augment_text(text):
    words = text.split()
    if len(words) < 4:
        return text

    aug_choice = random.choice(['swap', 'delete'])

    if aug_choice == 'swap':
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    elif aug_choice == 'delete' and len(words) > 1:
        del words[random.randint(0, len(words)-1)]

    return ' '.join(words)

# Dataset Preparation
def load_dataset_goemotions():
    return load_dataset("go_emotions")

def process_goemotions(dataset):
    train_df = pd.DataFrame(dataset["train"])
    val_df = pd.DataFrame(dataset["validation"])
    test_df = pd.DataFrame(dataset["test"])
    labels = dataset["train"].features["labels"].feature.names
    counts = Counter()
    for row in train_df["labels"]:
        for label in row:
            counts[labels[label]] += 1
    return train_df, val_df, test_df, labels, counts

def balance_dataset(train_df, emotion_labels, label_counts):
    max_count = max(label_counts.values())
    target = {label: int(max_count * 0.5) for label in label_counts}
    current = {label: 0 for label in label_counts}

    texts, labels = [], []
    for _, row in train_df.iterrows():
        text = row["text"]
        lbls = row["labels"]
        if not lbls:
            continue
        for l in lbls:
            name = emotion_labels[l]
            current[name] += 1
            if current[name] < target[name]:
                for _ in range(2):
                    texts.append(augment_text(text))
                    labels.append(lbls)

    return pd.concat([train_df, pd.DataFrame({"text": texts, "labels": labels})], ignore_index=True)

# Dataset Class
class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding=True, truncation=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label_tensor = torch.zeros(len(emotion_labels))
        label_tensor[self.labels[idx]] = 1
        item['labels'] = label_tensor
        return item

    def __len__(self):
        return len(self.labels)

# Evaluation Function
def evaluate_model(model, data_loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits).cpu().numpy() > 0.5

            all_labels.extend(labels)
            all_preds.extend(preds)

    report = classification_report(all_labels, all_preds, target_names=emotion_labels, zero_division=0)
    print("Classification Report:\n", report)
    acc = accuracy_score(all_labels, all_preds)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    hamming = 1 - hamming_loss(all_labels, all_preds)
    print(f"Accuracy (Subset Accuracy): {acc:.4f}")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Hamming Accuracy: {hamming:.4f}")
    
# Training Function
def train_model(model, train_loader, optimizer, scheduler, device, epochs=3, accumulation_steps=2):
    scaler = GradScaler()
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        progress_bar = tqdm(train_loader, total=len(train_loader), desc="Training")
        optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss.mean() / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=running_loss / (step + 1))

    return model
# Prediction Function
def predict_emotions(texts, model, tokenizer, threshold=0.5):
    model.eval()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()

    predictions = []
    for prob in probs:
        pred_labels = [emotion_labels[i] for i, p in enumerate(prob) if p >= threshold]
        predictions.append(pred_labels if pred_labels else ["neutral"])

    return predictions
# Main Function
def main():
    global model, tokenizer, emotion_labels

    hf_token = "Token"  # Replace with your token

    dataset = load_dataset_goemotions()
    train_df, val_df, test_df, emotion_labels, label_counts = process_goemotions(dataset)
    train_df = balance_dataset(train_df, emotion_labels, label_counts)

    model_name = "google/electra-large-discriminator"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(emotion_labels),
        problem_type="multi_label_classification"
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]  # Electra's attention projections
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    batch_size = 32
    accumulation_steps = 2

    train_dataset = GoEmotionsDataset(train_df['text'].tolist(), train_df['labels'].tolist(), tokenizer)
    val_dataset = GoEmotionsDataset(val_df['text'].tolist(), val_df['labels'].tolist(), tokenizer)
    test_dataset = GoEmotionsDataset(test_df['text'].tolist(), test_df['labels'].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)


    optimizer = AdamW(model.parameters(), lr=1e-4)
    total_steps = len(train_loader) * 3 // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model = train_model(model, train_loader, optimizer, scheduler, device, epochs=3, accumulation_steps=accumulation_steps)

    print("\nEvaluating model before saving...")
    evaluate_model(model, test_loader, device)

    save_directory = "electra-goemotions-lora"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"✅ Model saved at {save_directory}")

    model.push_to_hub("Saumith/electra-goemotions-lora", token=hf_token)
    tokenizer.push_to_hub("Saumith/electra-goemotions-lora", token=hf_token)
    print("✅ Model pushed to Hugging Face!")

if __name__ == "__main__":
    main()
    load_model_and_tokenizer()
    run_manual_tests()
