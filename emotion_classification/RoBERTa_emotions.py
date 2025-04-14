# import necessary libraries
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, jaccard_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

# Set random seed for reproducibility
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

set_seed()

# Load GoEmotions dataset
def load_goemotions():
    dataset = load_dataset("go_emotions")
    return dataset

# Data Augmentation: Random Swap and Deletion
def augment_text(text):
    words = text.split()
    if len(words) > 3 and random.random() > 0.5:
        i1, i2 = random.sample(range(len(words)), 2)
        words[i1], words[i2] = words[i2], words[i1]
    if len(words) > 3 and random.random() > 0.7:
        words.pop(random.randint(0, len(words) - 1))
    return " ".join(words)

# Balance Dataset by Augmenting
def balance_dataset(train_df, emotion_labels):
    label_counts = {label: 0 for label in emotion_labels}
    for labels in train_df['labels']:
        for l in labels:
            label_counts[emotion_labels[l]] += 1

    max_count = max(label_counts.values())
    target = {label: int(max_count * 0.5) for label in label_counts}

    current = {label: 0 for label in label_counts}
    new_texts = []
    new_labels = []

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
                    new_texts.append(augment_text(text))
                    new_labels.append(lbls)

    aug_df = pd.DataFrame({"text": new_texts, "labels": new_labels})
    return pd.concat([train_df, aug_df], ignore_index=True)

# Process GoEmotions dataset
def process_goemotions(dataset):
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    # Get emotion labels
    emotion_labels = dataset['train'].features['labels'].feature.names
    num_labels = len(emotion_labels)

    return train_df, val_df, test_df, emotion_labels, num_labels

# Custom Dataset Class (No Preprocessing, No Augmentation)
class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels_tensor = torch.FloatTensor(labels)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels_tensor
        }

# Evaluate Metrics
def evaluate_metrics(true_labels, predictions, mode="Validation"):
    micro_f1 = f1_score(true_labels, predictions, average='micro')
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    precision = precision_score(true_labels, predictions, average='micro')
    recall = recall_score(true_labels, predictions, average='micro')
    hamming = hamming_loss(true_labels, predictions)
    jaccard = jaccard_score(true_labels, predictions, average='micro')

    print(f"\n----- {mode} Metrics -----")
    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Precision (Micro): {precision:.4f}")
    print(f"Recall (Micro): {recall:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Jaccard Score (Micro): {jaccard:.4f}")
    print("-------------------------\n")

# Training Function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=3):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc="Training")

        for batch in train_progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc="Validation")
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                val_loss += loss.item()

                preds = torch.sigmoid(outputs.logits).cpu().numpy() >= 0.5
                all_preds.extend(preds.astype(int).tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

                val_progress_bar.set_postfix({'loss': loss.item()})

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation Loss: {avg_val_loss}")

        evaluate_metrics(all_labels, all_preds, mode="Validation")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'roberta_goemotions_best.pt')
            print("Model saved!")

    return model

# Test Model
def test_model(model, test_dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs.logits).cpu().numpy() >= 0.5

            all_preds.extend(preds.astype(int).tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    evaluate_metrics(all_labels, all_preds, mode="Test")

# Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_goemotions()
    train_df, val_df, test_df, emotion_labels, num_labels = process_goemotions(dataset)

    # Balance and augment the training set
    train_df = balance_dataset(train_df, emotion_labels)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    train_dataset = GoEmotionsDataset(
        texts=train_df['text'].tolist(),
        labels=[[1 if label in labels else 0 for label in range(num_labels)] for labels in train_df['labels']],
        tokenizer=tokenizer
    )

    val_dataset = GoEmotionsDataset(
        texts=val_df['text'].tolist(),
        labels=[[1 if label in labels else 0 for label in range(num_labels)] for labels in val_df['labels']],
        tokenizer=tokenizer
    )

    test_dataset = GoEmotionsDataset(
        texts=test_df['text'].tolist(),
        labels=[[1 if label in labels else 0 for label in range(num_labels)] for labels in test_df['labels']],
        tokenizer=tokenizer
    )

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print("Training model...")
    model = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=3)

    print("Testing model...")
    model.load_state_dict(torch.load('roberta_goemotions_best.pt'))
    test_model(model, test_dataloader, device)

    # ✅ Save model and tokenizer properly for Hugging Face upload
    save_path = "/content/roberta_goemotions_final"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n✅ Model and tokenizer saved to {save_path}!")
    print(f"You can now push this folder to Hugging Face 🚀")

if __name__ == "__main__":
    main()
