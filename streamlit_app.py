"""
Voice-Enabled Chatbot - Streamlit web app (deployed on Streamlit Community Cloud).

Pipeline:  microphone audio -> Whisper (speech-to-text)
                            -> DistilBERT (intent classification)
                            -> response generator (bot.py)
                            -> text reply + spoken reply (gTTS)

Run locally:  streamlit run streamlit_app.py
"""

import io

import streamlit as st
from scipy.io import wavfile

from bot import load_asr, load_model, predict, respond, transcribe

st.set_page_config(page_title="Voice-Enabled Chatbot", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner="Loading Whisper and DistilBERT ...")
def load_models():
    tokenizer, model = load_model()
    return load_asr(), tokenizer, model


def speak(text):
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text).write_to_fp(buf)
        return buf.getvalue()
    except Exception:                       # TTS is optional; text reply still works
        return None


asr, tokenizer, intent_model = load_models()
if "history" not in st.session_state:
    st.session_state.history = []          # list of (user_text, bot_reply)
    st.session_state.last_audio = None
    st.session_state.result = None

st.title("🎙️ Voice-Enabled Chatbot")
st.write("Speak to the bot: **Whisper** converts your speech to text, a fine-tuned "
         "**DistilBERT** model classifies the intent, and the bot replies in text and voice.")
st.caption("Try: \"what time is it\", \"tell me a joke\", \"what is 25 times 4\", "
           "\"how do you spell rhythm\", \"define gravity\", \"flip a coin\", "
           "\"what can I ask you\".")

left, right = st.columns(2)
with left:
    st.subheader("1. Ask a question")
    audio = st.audio_input("Record your voice (click the mic, speak, click stop)")
    with st.form("text_form", clear_on_submit=True):
        typed = st.text_input("...or type a message")
        sent = st.form_submit_button("Send")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.result = None

# work out which input is new on this run
text = None
if audio is not None and audio.file_id != st.session_state.last_audio:
    st.session_state.last_audio = audio.file_id
    sr, y = wavfile.read(io.BytesIO(audio.getvalue()))
    with st.spinner("Recognising speech ..."):
        text = transcribe(asr, sr, y)
elif sent and typed.strip():
    text = typed.strip()

if text:
    intent, conf, ranked = predict(text, tokenizer, intent_model)
    last_reply = st.session_state.history[-1][1] if st.session_state.history else ""
    reply = respond(intent, text, last_reply)
    st.session_state.history.append((text, reply))
    st.session_state.result = {"text": text, "ranked": ranked, "reply": reply,
                               "audio": speak(reply)}
elif audio is not None and text == "":
    st.warning("I couldn't hear anything. Please try recording again.")

with right:
    res = st.session_state.result
    st.subheader("2. Recognized speech (Whisper)")
    st.info(res["text"] if res else "-")
    st.subheader("3. Detected intent (DistilBERT, top-3)")
    if res:
        for name, p in res["ranked"]:
            st.progress(p, text=f"{name}: {p * 100:.1f} %")
    else:
        st.write("-")
    st.subheader("4. Chatbot response")
    st.success(res["reply"] if res else "-")
    if res and res["audio"]:
        st.audio(res["audio"], format="audio/mp3", autoplay=bool(text))

st.divider()
st.subheader("Conversation")
for user_text, bot_reply in st.session_state.history:
    st.chat_message("user").write(user_text)
    st.chat_message("assistant").write(bot_reply)
