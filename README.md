# EmoLingo-ChatBot
EmoLingo is an NLP-based chatbot that predicts user emotion, tone, and English proficiency using fine-tuned transformer models. It integrates these predictions into a unified response pipeline and features an interactive Streamlit UI for real-time, emotionally aware conversations.

## 🚀 Features

- 🎭 **Emotion Detection** — Identifies emotions like joy, sadness, anger, etc.
- ✍️ **Tone Classification** — Distinguishes between various writing tones (formal, casual, empathetic, etc.)
- 🧠 **English Proficiency Estimation** — Assesses language fluency and command of English
- 🔗 **Multi-Model Response Pipeline** — Combines outputs for emotion-aware and style-consistent replies
- 💬 **Streamlit Interface** — User-friendly, interactive chat experience

## 📁 Project Structure

    Emolingo-ChatBot/
    ├── app.py                       # Main application script for Streamlit UI
    ├── emotion_classification.py    # Module for emotion detection
    ├── writing_tone.py              # Module for tone classification
    ├── Writing_proficiency.py       # Module for assessing English proficiency
    ├── pipeline.py                  # Integrates all models into a unified pipeline
    ├── writing_tone_dataset.csv     # Dataset for training tone classification model
    ├── requirements.txt             # Necessary Packages
    └── README.md

## Demo

https://github.com/user-attachments/assets/ab90e120-3d03-4862-ae7d-fcd1560f3161

