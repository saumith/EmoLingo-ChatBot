import streamlit as st
from pipeline import (
    predict_emotion,
    predict_proficiency,
    predict_style,
    full_pipeline
)
from datetime import datetime

# Page config
st.set_page_config(page_title="EmoLingo ChatBot", page_icon="🤖", layout="wide")

# ✅ Clean CSS (light mode, normal background)
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: white;
        color: black;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;
        background-color: white;
    }
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chat messages */
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        color: black;
    }

    .chat-timestamp {
        font-size: 0.75rem;
        color: gray;
    }

    /* Text input */
    textarea, .stTextInput > div > input {
        background-color: white;
        color: black;
        border: 1px solid #ccc;
    }

    /* Button styling */
    .stButton button {
        background-color: #f0f2f6;
        color: black;
        border: 1px solid black;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 EmoLingo ChatBot")

# New Chat button
if st.button("🆕 New Chat"):
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hello! 👋 I'm your EmoLingo assistant. How can I help you today?", "time": datetime.now().strftime("%H:%M")}
    ]
    st.rerun()  # ✅ Fixed rerun

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hello! 👋 I'm your EmoLingo assistant. How can I help you today?", "time": datetime.now().strftime("%H:%M")}
    ]

# Display chat history
for message_data in st.session_state['messages']:
    role = message_data["role"]
    content = message_data["content"]
    with st.chat_message(role):
        st.markdown(content)

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state['messages'].append({"role": "user", "content": prompt, "time": datetime.now().strftime("%H:%M")})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process context
    context = "\n".join([msg["content"] for msg in st.session_state['messages'] if msg["role"] == "user"])

    with st.chat_message("assistant"):
        with st.spinner("Bot is thinking..."):
            # Predict
            emotion = predict_emotion(prompt)
            proficiency = predict_proficiency(prompt)
            style = predict_style(prompt)[0]
            final_response = full_pipeline(context)

            # Compose reply
            reply = (
                f"🔍 **Analysis Results:**\n"
                f"- Emotion: `{emotion}`\n"
                f"- Proficiency: `{proficiency}`\n"
                f"- Tone: `{style}`\n\n"
                f"💡 **AI Response:**\n{final_response}"
            )

            st.markdown(reply)

            # Save assistant message
            st.session_state['messages'].append({"role": "assistant", "content": reply, "time": datetime.now().strftime("%H:%M")})
