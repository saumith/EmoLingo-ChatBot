import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, pipeline, T5ForConditionalGeneration, T5Tokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import openai

# 1. Setting up Groq API
client = openai.OpenAI(
    api_key="api_key", # Replace with your Groq API key
    base_url="https://api.groq.com/openai/v1"
)

# 2. Labelling to Emotion Mapping
label2emotion = {
    "LABEL_0": "admiration", "LABEL_1": "amusement", "LABEL_2": "anger", "LABEL_3": "annoyance",
    "LABEL_4": "approval", "LABEL_5": "caring", "LABEL_6": "confusion", "LABEL_7": "curiosity",
    "LABEL_8": "desire", "LABEL_9": "disappointment", "LABEL_10": "disapproval", "LABEL_11": "disgust",
    "LABEL_12": "embarrassment", "LABEL_13": "excitement", "LABEL_14": "fear", "LABEL_15": "gratitude",
    "LABEL_16": "grief", "LABEL_17": "joy", "LABEL_18": "love", "LABEL_19": "nervousness",
    "LABEL_20": "optimism", "LABEL_21": "pride", "LABEL_22": "realization", "LABEL_23": "relief",
    "LABEL_24": "remorse", "LABEL_25": "sadness", "LABEL_26": "surprise", "LABEL_27": "neutral"
}

# 3. Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# 4. Loading RoBERTa Model (Emotion Model 1)
roberta_model_path = "krishnathalapathy/robertafinalaug"
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_path)
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_path)
roberta_pipe = pipeline("text-classification", model=roberta_model, tokenizer=roberta_tokenizer, return_all_scores=True)

# 5. Loading T5 Model (Emotion Model 2)
t5_model_path = "Akshay-Sai/t5base-goemotions-finetuned"
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)

# 6. Loading Electra + LoRA Model (Emotion Model 3)
electra_base_model = AutoModelForSequenceClassification.from_pretrained(
    "google/electra-large-discriminator", 
    num_labels=28
).to(device)

electra_adapter_model = PeftModel.from_pretrained(
    electra_base_model,
    "Saumith/electra-goemotions-lora"
).to(device)

electra_tokenizer = AutoTokenizer.from_pretrained(
    "Saumith/electra-goemotions-lora", 
    use_fast=False
)

print("✅ All 3 Emotion Models Loaded!")

# 7. Loading our Proficiency Model locally

# Defining our custom model
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

# Loading tokenizer and model
proficiency_tokenizer = AutoTokenizer.from_pretrained("./proficiency_model")
proficiency_model = ProficiencyDistilBERT("distilbert-base-uncased")
proficiency_model.load_state_dict(torch.load("./proficiency_model/pytorch_model.bin", map_location=device))
proficiency_model.to(device)
proficiency_model.eval()

print("✅ Proficiency Model Loaded!")

# 7.5 Loading Tone Detection Model
style_model_name = "Akshay-Sai/roberta-style-classifier"
style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
style_model = AutoModelForSequenceClassification.from_pretrained(style_model_name)
style_model.to(device)

def predict_style(texts):
    if isinstance(texts, str):
        texts = [texts]

    inputs = style_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    style_model.eval()
    with torch.no_grad():
        outputs = style_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    predicted_labels = [style_model.config.id2label[p.item()] for p in predictions]
    return predicted_labels

print("✅ Tone Detection Model Loaded!")


# 8. Predicting Emotion (Ensemble of 3 Models)
def predict_emotion(user_input):
    emotion_scores = {}

    print("\n[INFO] Individual Emotion Model Outputs:")

    # --- RoBERTa ---
    roberta_results = roberta_pipe(user_input)[0]
    top1_roberta = max(roberta_results, key=lambda x: x['score'])
    roberta_emotion = label2emotion.get(top1_roberta['label'], top1_roberta['label'])
    print(f"  → RoBERTa predicts: {roberta_emotion} (Confidence: {top1_roberta['score']:.4f})")
    emotion_scores[roberta_emotion] = emotion_scores.get(roberta_emotion, 0) + 1

    # --- T5 ---
    prompt = f"classify: {user_input}"
    inputs = t5_tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = t5_model.generate(**inputs, max_new_tokens=5)
    t5_predicted_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    t5_predicted_text = t5_predicted_text.split(",")[0].strip()

    t5_emotion = None
    for emotion in label2emotion.values():
        if t5_predicted_text in emotion.lower():
            t5_emotion = emotion
            break

    if t5_emotion:
        print(f"  → T5 predicts: {t5_emotion}")
        emotion_scores[t5_emotion] = emotion_scores.get(t5_emotion, 0) + 1
    else:
        print(f"⚠️ T5 output '{t5_predicted_text}' not matched!")

    # --- Electra + LoRA ---
    inputs = electra_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    outputs = electra_adapter_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]
    top_idx = torch.argmax(probs).item()
    electra_emotion = list(label2emotion.values())[top_idx]
    electra_confidence = probs[top_idx].item()

    print(f"  → Electra-LoRA predicts: {electra_emotion} (Confidence: {electra_confidence:.4f})")
    emotion_scores[electra_emotion] = emotion_scores.get(electra_emotion, 0) + 1

    # Final Decision: Majority voting
    predicted_emotion = max(emotion_scores, key=emotion_scores.get)

    print(f"\n[INFO] Final Ensemble Predicted Emotion: {predicted_emotion}\n")
    return predicted_emotion

# 9. Predicting Proficiency
id2label_manual = {0: "A2", 1: "B1", 2: "B2"}

def predict_proficiency(user_input):
    inputs = proficiency_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    outputs = proficiency_model(**inputs)
    logits = outputs["logits_proficiency"]

    probs = torch.softmax(logits, dim=1)[0]

    print("\n[INFO] Proficiency Model Confidence Scores:")
    for idx, prob in enumerate(probs.tolist()):
        print(f"  → {id2label_manual[idx]}: {prob:.4f}")

    best_idx = torch.argmax(probs).item()
    best_confidence = probs[best_idx].item()

    predicted_label = id2label_manual[best_idx]

    if predicted_label == "B2" and best_confidence > 0.83:
        predicted_label = "Advanced"

    print(f"\n[INFO] Predicted Proficiency Level: {predicted_label}\n")
    return predicted_label

# 10. Generating Answer
def generate_answer(user_input, style, emotion):
    prompt = (
        f"Respond to the following message in a {style.lower()} tone, while being helpful and emotionally supportive "
        f"(the user is feeling {emotion}).\n\nUser: {user_input}"
    )
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()


# 11. Generating Emotional Support Message
def generate_emotional_message(emotion):
    prompt = f"Write a short emotional support message for someone feeling {emotion}."
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# 12. Simplifying Text Based on Proficiency
def simplify_text_for_proficiency(text, proficiency):
    if proficiency == "A2":
        simple_prompt = f"Rewrite the following text in very simple English suitable for a beginner:\n\n{text}"
    elif proficiency == "B1":
        simple_prompt = f"Rewrite the following text in clear and easy English suitable for intermediate level:\n\n{text}"
    elif proficiency == "B2":
        simple_prompt = f"Rewrite the following text to be slightly easier and more fluent for an upper-intermediate English learner:\n\n{text}"
    else:
        return text  # Advanced

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": simple_prompt}],
        temperature=0.5,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# 13. Full Pipeline
def full_pipeline(user_prompt):
    # Step 1: Emotion Detection
    emotion = predict_emotion(user_prompt)

    # Step 2: Proficiency Detection
    proficiency = predict_proficiency(user_prompt)

    # Step 3: Style/Tone Detection
    style = predict_style(user_prompt)[0]
    print(f"[INFO] Predicted Style: {style}")

    # Step 4: Create a single unified response prompt
    combined_prompt = (
        f"Write a response to the following user message in a {style.lower()} tone, while being emotionally supportive "
        f"(the user is feeling {emotion}). The language should be easy to understand for someone at {proficiency} level.\n\n"
        f"User: {user_prompt}"
    )

    # Step 5: Generating the final unified answer
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.5,
        max_tokens=512
    )
    final_message = response.choices[0].message.content.strip()

    return final_message



# 14. Example Usage
if __name__ == "__main__":
    user_input = "I don’t understand why my model is overfitting. I reduced the learning rate and added dropout, but the validation loss still increases. What should I do?"
    final_output = full_pipeline(user_input)
    print("\nFinal Customized Emotional Support Output:\n")
    print(final_output)
