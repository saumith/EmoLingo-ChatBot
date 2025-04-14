# importing necessary libraries
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)

# Detect device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom model (PyTorch)
class ProficiencyDistilBERT(nn.Module):
    def __init__(self, model_name, num_labels_proficiency=3):
        super(ProficiencyDistilBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.proficiency_classifier = nn.Linear(self.bert.config.hidden_size, num_labels_proficiency)

    def forward(self, input_ids, attention_mask=None, labels_proficiency=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits_proficiency = self.proficiency_classifier(pooled_output)

        loss = None
        if labels_proficiency is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits_proficiency, labels_proficiency)

        return {"loss": loss, "logits_proficiency": logits_proficiency}

# Dataset class
class ICNALESentenceDataset(Dataset):
    def __init__(self, texts, labels_proficiency, tokenizer, label2id_proficiency):
        self.texts = texts
        self.labels_proficiency = labels_proficiency
        self.tokenizer = tokenizer
        self.label2id_proficiency = label2id_proficiency

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_proficiency = self.label2id_proficiency[self.labels_proficiency[idx]]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors='pt')
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels_proficiency'] = torch.tensor(label_proficiency)
        return encoding

# Data collator
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels_proficiency = torch.stack([item['labels_proficiency'] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels_proficiency": labels_proficiency}

# Analyzer class
class EnglishProficiencyAnalyzer:
    def __init__(self):
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.label2id_proficiency = {"A2": 0, "B1": 1, "B2": 2}
        self.id2label_proficiency = {v: k for k, v in self.label2id_proficiency.items()}

    def load_icnale_dataset(self, root_dir):
        data = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith(".txt") or fname.endswith(".TXT"):
                    proficiency_level = self.map_proficiency_level(fname)
                    if proficiency_level is None:
                        continue
                    try:
                        with open(os.path.join(dirpath, fname), encoding='utf-8') as f:
                            essay = f.read().strip()
                            sentences = sent_tokenize(essay)
                            for sentence in sentences:
                                tokens = word_tokenize(sentence)
                                if len(tokens) >= 5:
                                    data.append({"text": sentence, "proficiency": proficiency_level})
                    except Exception:
                        continue
        return pd.DataFrame(data)

    def map_proficiency_level(self, filename):
        if "_A2" in filename:
            return "A2"
        elif "_B1" in filename:
            return "B1"
        elif "_B2" in filename:
            return "B2"
        else:
            return None

    def train_model(self, icnale_path):
        proficiency_data = self.load_icnale_dataset(icnale_path)
        print(f"\nOriginal Dataset distribution:\n{proficiency_data['proficiency'].value_counts()}")

        # Balance dataset
        min_count = proficiency_data['proficiency'].value_counts().min()
        proficiency_data = proficiency_data.groupby('proficiency').apply(
            lambda x: x.sample(min_count, random_state=42)
        ).reset_index(drop=True)

        print(f"\nBalanced Dataset distribution:\n{proficiency_data['proficiency'].value_counts()}")

        X_train, X_test, y_train, y_test = train_test_split(
            proficiency_data['text'], proficiency_data['proficiency'], test_size=0.2, random_state=42
        )

        train_dataset = ICNALESentenceDataset(X_train.tolist(), y_train.tolist(), self.tokenizer, self.label2id_proficiency)
        test_dataset = ICNALESentenceDataset(X_test.tolist(), y_test.tolist(), self.tokenizer, self.label2id_proficiency)

        self.model = ProficiencyDistilBERT(self.model_name).to(device)

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            save_strategy="no"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=collate_fn,
            tokenizer=self.tokenizer
        )

        trainer.train()

        # Evaluate manually
        self.evaluate_model(test_dataset)

        # Save model
        os.makedirs("./proficiency_model", exist_ok=True)
        torch.save(self.model.state_dict(), "./proficiency_model/pytorch_model.bin")
        self.tokenizer.save_pretrained("./proficiency_model")
        print("\n✅ Model saved successfully in './proficiency_model'!")

    def evaluate_model(self, test_dataset):
        self.model.eval()

        all_preds = []
        all_labels = []

        test_loader = DataLoader(test_dataset, batch_size=32)

        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels_proficiency'].to(device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs["logits_proficiency"], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        print("\nClassification Report (Test Set):")
        print(classification_report(all_labels, all_preds, target_names=["A2", "B1", "B2"], digits=4))

    def predict(self, text):
        self.model.eval()
        tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**tokens)

        logits = outputs["logits_proficiency"]
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        proficiency_label = self.id2label_proficiency[pred.item()]

        if proficiency_label == "B2" and confidence.item() > 0.83:
            proficiency_label = "Advanced"

        return proficiency_label, confidence.item()

    def interactive_mode(self):
        print("\n=== English Proficiency Analyzer ===")
        print("Type a sentence and get proficiency prediction (type 'exit' to quit):\n")
        while True:
            text = input("Enter a sentence: ")
            if text.lower() == 'exit':
                break
            if not text.strip():
                print("Please enter a valid sentence.")
                continue
            proficiency, confidence = self.predict(text)
            print(f"\nPredicted Proficiency Level: {proficiency} (Confidence: {confidence:.2f})\n")

# Main runner
if __name__ == "__main__":
    analyzer = EnglishProficiencyAnalyzer()
    icnale_path = "ICNALE_WE_2.6"  # Your ICNALE dataset path
    analyzer.train_model(icnale_path)
    analyzer.interactive_mode()
