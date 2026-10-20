"""
Chatbot brain: Whisper speech recognition, DistilBERT intent classifier and
response generator.

Used by streamlit_app.py (the web UI) and by the training / evaluation scripts.
"""

import ast
import json
import operator
import os
import random
import re
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_DIR = "model/distilbert_intent"               # written by train_distilbert.py
MODEL_ID = "Xyphitos/distilbert-clinc25-intent"     # same model on the Hugging Face Hub
ASR_MODEL = "openai/whisper-base.en"
CONF_THRESHOLD = 0.5          # below this the bot treats the query as out-of-scope

# the 25 CLINC150 intents this assistant answers; everything else -> "oos"
INTENTS = [
    "greeting", "goodbye", "thank_you", "yes", "no", "repeat",
    "time", "date", "calculator", "flip_coin", "roll_dice", "spelling", "definition",
    "tell_joke", "fun_fact", "meaning_of_life",
    "what_is_your_name", "who_made_you", "how_old_are_you", "are_you_a_bot",
    "what_can_i_ask_you", "what_are_your_hobbies", "where_are_you_from",
    "do_you_have_pets", "who_do_you_work_for",
]


# ----------------------------------------------------------------------------
# speech recognition: Whisper
# ----------------------------------------------------------------------------
def load_asr():
    return pipeline("automatic-speech-recognition", model=ASR_MODEL,
                    device=0 if torch.cuda.is_available() else -1)


def transcribe(asr, sr, y):
    """sr = sample rate, y = int16 or float numpy array (mono or stereo)"""
    y = y.astype(np.float32)
    if y.ndim > 1:                          # stereo -> mono
        y = y.mean(axis=1)
    if np.abs(y).max() > 1.0:               # int16 range -> [-1, 1]
        y = y / 32768.0
    if sr != 16000:                         # Whisper expects 16 kHz
        y = resample_poly(y, 16000, sr).astype(np.float32)
    return asr({"raw": y, "sampling_rate": 16000})["text"].strip()


# ----------------------------------------------------------------------------
# intent classifier: fine-tuned DistilBERT (see train_distilbert.py)
# ----------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def load_model():
    path = MODEL_DIR if os.path.isdir(MODEL_DIR) else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model, top_k=3):
    enc = tokenizer(text, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)[0]
    top = probs.topk(top_k)
    ranked = [(model.config.id2label[int(i)], float(p)) for p, i in zip(top.values, top.indices)]
    intent, conf = ranked[0]
    if conf < CONF_THRESHOLD:
        intent = "oos"
    return intent, conf, ranked


# ----------------------------------------------------------------------------
# responses
# ----------------------------------------------------------------------------
REPLIES = {
    "greeting": ["Hello! How can I help you today?", "Hi there! Ask me anything.",
                 "Hey! Nice to hear from you."],
    "goodbye": ["Goodbye! Have a great day.", "See you later!", "Bye! Talk to you soon."],
    "thank_you": ["You're welcome!", "Happy to help!", "Anytime!"],
    "yes": ["Great!", "Alright, noted."],
    "no": ["Okay, no problem.", "Alright, let me know if you need anything else."],
    "tell_joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break, and it said: no problem, I'll go to sleep.",
        "Why did the neural network break up? It had too many hidden layers.",
        "There are 10 kinds of people: those who understand binary and those who don't.",
    ],
    "fun_fact": [
        "Honey never spoils. Archaeologists have found edible honey in ancient Egyptian tombs.",
        "Octopuses have three hearts and blue blood.",
        "A day on Venus is longer than a year on Venus.",
        "Bananas are berries, but strawberries are not.",
    ],
    "meaning_of_life": ["According to The Hitchhiker's Guide to the Galaxy, it's 42.",
                        "Learning, growing and being kind. And maybe 42."],
    "what_is_your_name": ["I'm VoiceBot, a speech-enabled deep learning chatbot."],
    "who_made_you": ["I was built by Arnav Panwar as a Speech and NLP lab project."],
    "how_old_are_you": ["I was trained only recently, so I'm very young!"],
    "are_you_a_bot": ["Yes, I'm a bot: Whisper hears you and DistilBERT works out what you mean."],
    "what_can_i_ask_you": [
        "You can ask me the time or date, do arithmetic, flip a coin, roll a die, "
        "spell or define a word, tell a joke or a fun fact, or just chat."],
    "what_are_your_hobbies": ["I enjoy listening to people talk. Literally, it's my job."],
    "where_are_you_from": ["I live in the cloud, on a Streamlit server."],
    "do_you_have_pets": ["No pets, although I'd like a Python."],
    "who_do_you_work_for": ["I work for whoever is talking to me, so right now, you!"],
    "oos": ["Sorry, I didn't understand that. Try asking 'what can I ask you?'",
            "That's outside what I know. You can ask me for the time, a joke, or some maths."],
}

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
       ast.USub: operator.neg}


def safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        if isinstance(node.op, ast.Pow) and abs(safe_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError("unsupported expression")


def calculate(text):
    t = text.lower().replace(",", "")
    for words, sym in [("to the power of", "**"), ("multiplied by", "*"), ("divided by", "/"),
                       ("squared", "**2"), ("cubed", "**3"), ("plus", "+"), ("minus", "-"),
                       ("times", "*"), (" x ", "*"), ("over", "/"), ("mod", "%")]:
        t = t.replace(words, f" {sym} ")
    expr = "".join(re.findall(r"[0-9.+\-*/%() ]", t)).strip(" +*/%")
    try:
        value = safe_eval(ast.parse(expr, mode="eval").body)
        value = round(value, 4) if isinstance(value, float) else value
        return f"{' '.join(expr.split())} = {value}"
    except Exception:
        return "Tell me the calculation with numbers, for example 'what is 12 times 7'."


def last_word_after(text, keys):
    t = text.lower().strip(" ?.!")
    for k in keys:
        if k in t:
            words = tokenize(t.split(k, 1)[1])
            words = [w for w in words if w not in ("the", "word", "a", "an", "mean", "means")]
            if words:
                return words[0]
    words = tokenize(t)
    return words[-1] if words else ""


def spell(text):
    word = last_word_after(text, ["spell", "spelling of", "spelled"])
    if not word:
        return "Which word should I spell?"
    return f"{word} is spelled {'-'.join(word.upper())}."


def define(text):
    word = last_word_after(text, ["definition of", "meaning of", "define", "what does",
                                  "what is a", "what is"])
    if not word:
        return "Which word should I define?"
    try:
        url = "https://api.dictionaryapi.dev/api/v2/entries/en/" + word
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.load(r)
        meaning = data[0]["meanings"][0]
        return (f"{word} ({meaning['partOfSpeech']}): "
                f"{meaning['definitions'][0]['definition']}")
    except Exception:
        return f"Sorry, I couldn't find a definition for '{word}'."


def respond(intent, text, last_reply=""):
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if intent == "time":
        return "It's " + now.strftime("%I:%M %p").lstrip("0") + " (IST)."
    if intent == "date":
        return "Today is " + now.strftime("%A, %d %B %Y") + "."
    if intent == "flip_coin":
        return "It's " + random.choice(["heads", "tails"]) + "!"
    if intent == "roll_dice":
        return f"You rolled a {random.randint(1, 6)}."
    if intent == "calculator":
        return calculate(text)
    if intent == "spelling":
        return spell(text)
    if intent == "definition":
        return define(text)
    if intent == "repeat":
        return last_reply or "I haven't said anything yet."
    return random.choice(REPLIES.get(intent, REPLIES["oos"]))
