"""
End-to-end voice evaluation: speech -> Whisper -> DistilBERT.

Test sentences from the CLINC150 test split are spoken with the two Windows SAPI
voices (Microsoft David / Zira) and saved as WAV files, then run through the same
transcribe() + predict() functions the web app uses. Reports Whisper word error
rate (WER) and intent accuracy on the transcripts vs the original typed text.

Writes : data/voice_test/*.wav, results_voice.json
Run    : python eval_voice.py   (Windows only, for the SAPI voices)
"""

import json
import os
import random
import re
import subprocess
import time

from scipy.io import wavfile

from bot import INTENTS, load_asr, load_model, predict, transcribe

asr = load_asr()
tokenizer, intent_model = load_model()
random.seed(0)
OUT_DIR = "data/voice_test"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("STEP 1: Pick test sentences (2 per intent + 20 out-of-scope)")
print("=" * 70)
data = json.load(open("data/clinc150_full.json"))
rows = []
for intent in INTENTS:
    pool = [t for t, y in data["test"] if y == intent]
    rows += [(t, intent) for t in random.sample(pool, 2)]
rows += [(t, "oos") for t in random.sample([t for t, _ in data["oos_test"]], 20)]
print(f"{len(rows)} sentences ({len(rows) - 20} in-scope, 20 oos)")

print("\n" + "=" * 70)
print("STEP 2: Synthesise speech with Windows SAPI (David / Zira, 16 kHz WAV)")
print("=" * 70)
list_file = os.path.abspath(os.path.join(OUT_DIR, "sentences.txt"))
open(list_file, "w", encoding="utf-8").write("\n".join(t for t, _ in rows))
ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$voices = @('Microsoft David Desktop', 'Microsoft Zira Desktop')
$i = 0
foreach ($line in Get-Content '{list_file}') {{
    $s.SelectVoice($voices[$i % 2])
    $s.SetOutputToWaveFile('{os.path.abspath(OUT_DIR)}\\clip_' + $i.ToString('000') + '.wav', $fmt)
    $s.Speak($line)
    $i++
}}
$s.SetOutputToNull()
"""
subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
print(f"Saved {len(rows)} clips to {OUT_DIR}/")


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(b)]


print("\n" + "=" * 70)
print("STEP 3: Whisper transcription + DistilBERT intent prediction")
print("=" * 70)
errors, ref_words = 0, 0
text_correct, voice_correct, asr_time = 0, 0, 0.0
for i, (sentence, gold) in enumerate(rows):
    sr, audio = wavfile.read(f"{OUT_DIR}/clip_{i:03d}.wav")
    t0 = time.time()
    heard = transcribe(asr, sr, audio)
    asr_time += time.time() - t0
    ref, hyp = words(sentence), words(heard)
    errors += edit_distance(ref, hyp)
    ref_words += len(ref)
    pred_text = predict(sentence, tokenizer, intent_model)[0]
    pred_voice = predict(heard, tokenizer, intent_model)[0]
    text_correct += pred_text == gold
    voice_correct += pred_voice == gold
    if i % 6 == 0 or pred_voice != gold:
        flag = "OK " if pred_voice == gold else "ERR"
        print(f"[{flag}] said : {sentence}\n      heard: {heard}\n"
              f"      gold={gold}  pred(voice)={pred_voice}")

n = len(rows)
res = {
    "n_clips": n,
    "wer": errors / ref_words,
    "intent_acc_text": text_correct / n,
    "intent_acc_voice": voice_correct / n,
    "asr_seconds_per_clip": asr_time / n,
}
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Whisper WER                     : {res['wer'] * 100:.2f} %")
print(f"Intent accuracy (typed text)    : {res['intent_acc_text'] * 100:.2f} %")
print(f"Intent accuracy (voice, end2end): {res['intent_acc_voice'] * 100:.2f} %")
print(f"ASR time per clip               : {res['asr_seconds_per_clip']:.2f} s")
json.dump(res, open("results_voice.json", "w"), indent=2)
print("Saved results_voice.json")
