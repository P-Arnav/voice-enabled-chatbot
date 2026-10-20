"""
Fine-tune DistilBERT as the intent classifier (same 26-class CLINC150 splits as
train_intent_model.py). This is the model the deployed app uses.

Writes  : model/distilbert_intent/ (weights + tokenizer), results_distilbert.json,
          figures/distilbert_curves.png, figures/confusion_matrix_distilbert.png

Run:  python train_distilbert.py
"""

import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bot import INTENTS, CONF_THRESHOLD, MODEL_DIR

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.makedirs("figures", exist_ok=True)

print("=" * 70)
print("STEP 1: Load CLINC150 and build the 26-class dataset")
print("=" * 70)
data = json.load(open("data/clinc150_full.json"))


def build_split(split, n_other):
    keep = [(t, y) for t, y in data[split] if y in INTENTS]
    other = [(t, "oos") for t, y in data[split] if y not in INTENTS]
    oos = [(t, "oos") for t, _ in data["oos_" + split]]
    rows = keep + oos + random.sample(other, n_other)
    random.shuffle(rows)
    return rows


train = build_split("train", 1500)
val = build_split("val", 300)
test_in = [(t, y) for t, y in data["test"] if y in INTENTS]
test_oos = [(t, "oos") for t, _ in data["oos_test"]]
test = test_in + test_oos
labels = INTENTS + ["oos"]
label_id = {l: i for i, l in enumerate(labels)}
print(f"Train {len(train)} | Val {len(val)} | Test {len(test)} "
      f"(in-scope {len(test_in)}, oos {len(test_oos)})")

print("\n" + "=" * 70)
print("STEP 2: Load pretrained DistilBERT (distilbert-base-uncased)")
print("=" * 70)
BASE = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)}, label2id=label_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}   device: {device}")
print("Tokenised example:", tokenizer.tokenize(train[0][0]))


def batches(rows, size, shuffle=False):
    if shuffle:
        rows = random.sample(rows, len(rows))
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        enc = tokenizer([t for t, _ in chunk], padding=True, truncation=True,
                        max_length=32, return_tensors="pt")
        y = torch.tensor([label_id[l] for _, l in chunk])
        yield enc.to(device), y.to(device)


def predict_probs(rows):
    model.eval()
    out = []
    with torch.no_grad():
        for enc, _ in batches(rows, 128):
            out.append(torch.softmax(model(**enc).logits, dim=1).cpu())
    return torch.cat(out)


print("\n" + "=" * 70)
print("STEP 3: Fine-tune (AdamW, lr=5e-5, batch 32, 4 epochs, linear decay)")
print("=" * 70)
EPOCHS, BATCH = 4, 32
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1 - s / steps)
history = {"train_loss": [], "val_acc": []}
start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = 0.0
    for enc, y in batches(train, BATCH, shuffle=True):
        loss = model(**enc, labels=y).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(y)
    val_acc = accuracy_score([label_id[l] for _, l in val], predict_probs(val).argmax(1))
    history["train_loss"].append(total / len(train))
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch}/{EPOCHS} | train_loss {total / len(train):.4f} | val_acc {val_acc:.4f}")
train_time = time.time() - start
print(f"Fine-tuning time: {train_time:.1f} s")

model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"Saved model -> {MODEL_DIR}/")

print("\n" + "=" * 70)
print("STEP 4: Evaluate on the CLINC150 test set")
print("=" * 70)
probs = predict_probs(test)
conf, pred = probs.max(1)
pred_thr = torch.where(conf < CONF_THRESHOLD, torch.tensor(label_id["oos"]), pred).numpy()
y = np.array([label_id[l] for _, l in test])
n_in = len(test_in)
res = {
    "overall_acc": accuracy_score(y, pred_thr),
    "in_scope_acc": accuracy_score(y[:n_in], pred_thr[:n_in]),
    "oos_recall": accuracy_score(y[n_in:], pred_thr[n_in:]),
    "macro_f1": f1_score(y, pred_thr, average="macro"),
}
print(f"DistilBERT (+ conf threshold {CONF_THRESHOLD})  overall {res['overall_acc']:.4f} | "
      f"in-scope {res['in_scope_acc']:.4f} | oos recall {res['oos_recall']:.4f} | "
      f"macro-F1 {res['macro_f1']:.4f}")

# latency on CPU, which is what the free hosting tier runs on
model.to("cpu")
device = "cpu"
t0 = time.time()
for t, _ in test[:200]:
    enc = tokenizer(t, return_tensors="pt")
    with torch.no_grad():
        model(**enc)
latency_ms = (time.time() - t0) / 200 * 1000
print(f"CPU latency per utterance: {latency_ms:.1f} ms")

print("\n" + "=" * 70)
print("STEP 5: Save figures and results")
print("=" * 70)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ep = range(1, EPOCHS + 1)
ax[0].plot(ep, history["train_loss"], marker="o")
ax[0].set_title("DistilBERT training loss")
ax[1].plot(ep, history["val_acc"], marker="o", color="tab:orange")
ax[1].set_title("DistilBERT validation accuracy")
for a in ax:
    a.set_xlabel("Epoch")
    a.set_xticks(list(ep))
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/distilbert_curves.png", dpi=150)

cm = confusion_matrix(y, pred_thr, labels=range(len(labels)))
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 9))
plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(fraction=0.046)
plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
plt.yticks(range(len(labels)), labels, fontsize=8)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("DistilBERT confusion matrix (row-normalised, test set)")
plt.tight_layout()
plt.savefig("figures/confusion_matrix_distilbert.png", dpi=150)
print("Saved figures/distilbert_curves.png, figures/confusion_matrix_distilbert.png")

json.dump({"distilbert": res, "epochs": EPOCHS, "train_time_s": train_time,
           "cpu_latency_ms": latency_ms, "history": history},
          open("results_distilbert.json", "w"), indent=2)
print("Saved results_distilbert.json")
