# 🎙️ Voice-Enabled Chatbot

Speech recognition + deep learning chatbot, deployed on Streamlit Community Cloud.

**Live app:** LIVE_URL

## How it works

1. **Speech recognition**: the browser records your voice and
   [`openai/whisper-base.en`](https://huggingface.co/openai/whisper-base.en) converts it to text.
2. **Intent classification**: DistilBERT fine-tuned on 25 intents of the
   [CLINC150](https://github.com/clinc/oos-eval) dataset plus an out-of-scope class
   ([`Xyphitos/distilbert-clinc25-intent`](https://huggingface.co/Xyphitos/distilbert-clinc25-intent)).
   Predictions below 0.5 confidence fall back to out-of-scope.
3. **Response generation**: template and dynamic replies (time, date, calculator,
   spelling, dictionary lookup, coin and dice, jokes, small talk) in `bot.py`.
4. **Text-to-speech**: the reply is also spoken back using gTTS.

The page shows the recognized speech, the top-3 intents with confidence, the response,
and the conversation history.

## Results (CLINC150 test set, 26 classes)

| Model | In-scope acc | OOS recall | Macro-F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 90.9 % | 92.1 % | 0.893 |
| BiLSTM (trained from scratch) | 86.7 % | 97.2 % | 0.891 |
| **DistilBERT (deployed)** | **95.1 %** | **98.1 %** | **0.957** |

End-to-end voice test (70 TTS-synthesised test sentences): Whisper WER 4.5 %,
intent accuracy 98.6 % from voice (98.6 % from typed text).

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Web UI |
| `bot.py` | Whisper ASR, DistilBERT intent prediction, response generator |
| `train_distilbert.py` | Fine-tunes DistilBERT, writes `model/distilbert_intent/` |
| `train_bilstm.py` | BiLSTM and TF-IDF baselines for comparison |
| `eval_voice.py` | End-to-end speech -> intent evaluation |
| `data/clinc150_full.json` | CLINC150 dataset (Larson et al., 2019, CC BY 3.0) |
| `figures/`, `results*.json`, `run_output_*.txt` | Training curves, confusion matrices, logs |

## Run locally

```bash
pip install -r requirements.txt
python train_distilbert.py     # optional: the app downloads the model from the Hub otherwise
streamlit run streamlit_app.py
```
