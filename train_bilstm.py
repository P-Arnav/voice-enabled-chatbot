"""
Train a BiLSTM intent classifier from scratch on a 25-intent subset of CLINC150
(+ out-of-scope), plus a TF-IDF + logistic regression baseline. Both are comparison
models for the report; the deployed app uses DistilBERT (train_distilbert.py).

Dataset : data/clinc150_full.json  (Larson et al., 2019 - github.com/clinc/oos-eval)
Writes  : model/intent_bilstm.pt, results.json, figures/*.png

Run:  python train_intent_model.py
"""

import json
import os
import random
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from bot import INTENTS, CONF_THRESHOLD, tokenize

MODEL_PATH = "model/intent_bilstm.pt"
MAX_LEN = 25


def encode(text, vocab):
    ids = [vocab.get(w, 1) for w in tokenize(text)][:MAX_LEN]     # 1 = <unk>
    if not ids:
        ids = [1]
    return ids + [0] * (MAX_LEN - len(ids))                        # 0 = <pad>


# Embedding -> BiLSTM -> max-pool over time -> Dropout -> Linear
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=100, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                  # ignore padding in the pool
        out, _ = self.lstm(self.embedding(x))          # (B, T, 2H)
        out = out.masked_fill(~mask, -1e9)
        pooled = out.max(dim=1).values                 # (B, 2H)
        return self.fc(self.dropout(pooled))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
os.makedirs("model", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("=" * 70)
print("STEP 1: Load CLINC150 and build the 26-class dataset")
print("=" * 70)
data = json.load(open("data/clinc150_full.json"))


def build_split(split, n_other):
    """keep our 25 intents; CLINC's own oos samples + n_other utterances of the
    other 125 intents (which this bot cannot handle) become the 'oos' class"""
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
print(f"Classes        : {len(labels)} (25 intents + oos)")
print(f"Train examples : {len(train)}  (oos = {sum(y == 'oos' for _, y in train)})")
print(f"Val examples   : {len(val)}")
print(f"Test examples  : {len(test)}  (in-scope {len(test_in)}, CLINC oos {len(test_oos)})")
print("Sample rows    :")
for t, y in train[:5]:
    print(f"   {y:<20s} | {t}")

print("\n" + "=" * 70)
print("STEP 2: Tokenise and build the vocabulary")
print("=" * 70)
counts = Counter(w for t, _ in train for w in tokenize(t))
vocab = {"<pad>": 0, "<unk>": 1}
for w, _ in counts.most_common():
    vocab[w] = len(vocab)
lengths = [len(tokenize(t)) for t, _ in train]
print(f"Vocabulary size      : {len(vocab)}")
print(f"Utterance length     : mean {np.mean(lengths):.1f}, max {max(lengths)} tokens")
print(f"Padded/truncated to  : 25 tokens")
print(f"Example encoding     : '{train[0][0]}'")
print(f"                       {encode(train[0][0], vocab)}")


def to_tensors(rows):
    X = torch.tensor([encode(t, vocab) for t, _ in rows])
    y = torch.tensor([label_id[l] for _, l in rows])
    return X, y


X_train, y_train = to_tensors(train)
X_val, y_val = to_tensors(val)
X_test, y_test = to_tensors(test)

print("\n" + "=" * 70)
print("STEP 3: Build the BiLSTM model")
print("=" * 70)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = IntentClassifier(len(vocab), len(labels)).to(device)
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {n_params:,}")

print("\n" + "=" * 70)
print("STEP 4: Train (Adam, lr=1e-3, batch 64, 20 epochs, weighted CE, keep best val)")
print("=" * 70)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# oos is ~40% of the training data, so weight classes by inverse frequency
class_counts = torch.bincount(y_train, minlength=len(labels)).float()
class_weights = class_counts.sum() / (len(labels) * class_counts)
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
EPOCHS, BATCH = 20, 64
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val, best_state = 0.0, None
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(X_train))
    total_loss, correct = 0.0, 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i + BATCH]
        xb, yb = X_train[idx].to(device), y_train[idx].to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
        correct += (out.argmax(1) == yb).sum().item()

    model.eval()
    with torch.no_grad():
        out = model(X_val.to(device))
        val_loss = loss_fn(out, y_val.to(device)).item()
        val_acc = (out.argmax(1).cpu() == y_val).float().mean().item()
    tr_loss, tr_acc = total_loss / len(X_train), correct / len(X_train)
    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    mark = ""
    if val_acc > best_val:
        best_val, best_epoch = val_acc, epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mark = "  <- best"
    print(f"Epoch {epoch:2d}/{EPOCHS} | loss {tr_loss:.4f} acc {tr_acc:.4f} | "
          f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}{mark}")

train_time = time.time() - start
print(f"Training time: {train_time:.1f} s | best val_acc {best_val:.4f} at epoch {best_epoch}")

model.load_state_dict(best_state)
model = model.cpu().eval()
torch.save({"state_dict": best_state, "vocab": vocab, "labels": labels}, MODEL_PATH)
print(f"Saved model -> {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1e6:.1f} MB)")

print("\n" + "=" * 70)
print("STEP 5: Evaluate on the CLINC150 test set")
print("=" * 70)
with torch.no_grad():
    probs = torch.softmax(model(X_test), dim=1)
conf, pred = probs.max(1)
pred_thr = torch.where(conf < CONF_THRESHOLD, torch.tensor(label_id["oos"]), pred)
y = y_test.numpy()
n_in = len(test_in)


def report(name, p):
    p = np.asarray(p)
    r = {
        "overall_acc": accuracy_score(y, p),
        "in_scope_acc": accuracy_score(y[:n_in], p[:n_in]),
        "oos_recall": accuracy_score(y[n_in:], p[n_in:]),
        "macro_f1": f1_score(y, p, average="macro"),
    }
    print(f"{name:<34s} overall {r['overall_acc']:.4f} | in-scope {r['in_scope_acc']:.4f} | "
          f"oos recall {r['oos_recall']:.4f} | macro-F1 {r['macro_f1']:.4f}")
    return r


res_bilstm = report("BiLSTM (argmax)", pred.numpy())
res_bilstm_thr = report(f"BiLSTM (+ conf threshold {CONF_THRESHOLD})", pred_thr.numpy())

# classical baseline for comparison
tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, C=10)
clf.fit(tfidf.fit_transform([t for t, _ in train]), y_train.numpy())
res_base = report("Baseline: TF-IDF + LogReg", clf.predict(tfidf.transform([t for t, _ in test])))

print("\nSample predictions:")
for i in random.sample(range(len(test)), 8):
    t, gold = test[i]
    print(f"   '{t}'\n      gold={gold:<18s} pred={labels[pred_thr[i]]:<18s} conf={conf[i]:.2f}")

print("\n" + "=" * 70)
print("STEP 6: Save figures and results")
print("=" * 70)
ep = range(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, history["train_loss"], label="train")
ax[0].plot(ep, history["val_loss"], label="validation")
ax[0].set_title("Loss")
ax[1].plot(ep, history["train_acc"], label="train")
ax[1].plot(ep, history["val_acc"], label="validation")
ax[1].set_title("Accuracy")
for a in ax:
    a.set_xlabel("Epoch")
    a.legend()
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/training_curves.png", dpi=150)

cm = confusion_matrix(y, pred_thr.numpy(), labels=range(len(labels)))
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(10, 9))
plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(fraction=0.046)
plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
plt.yticks(range(len(labels)), labels, fontsize=8)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("BiLSTM confusion matrix (row-normalised, test set)")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150)
print("Saved figures/training_curves.png, figures/confusion_matrix.png")

json.dump({
    "n_train": len(train), "n_val": len(val), "n_test_in": n_in, "n_test_oos": len(test_oos),
    "vocab_size": len(vocab), "n_params": n_params, "epochs": EPOCHS,
    "best_epoch": best_epoch, "best_val_acc": best_val, "train_time_s": train_time,
    "bilstm": res_bilstm, "bilstm_threshold": res_bilstm_thr, "tfidf_logreg": res_base,
    "history": history,
}, open("results.json", "w"), indent=2)
print("Saved results.json")
