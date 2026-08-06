"""
Barebones starting point: GPT-2 (frozen) + small MLP head for sentiment
classification. No early stopping, no threshold tuning, no bells and
whistles -- just the core pipeline, meant to be extended.

Fill in load_data() with your actual file paths, then run.
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# 1. Load your data from a CSV. Each row's index 1 is the review text,
#    index 2 is the label (0 = negative, 1 = positive). Returns a list
#    of review strings and a list of int labels.
# ---------------------------------------------------------------------
def load_data(csv_path):
    import csv
    texts, labels = [], []
    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            texts.append(row[1])
            labels.append(int(row[2]))
    return texts, labels


# ---------------------------------------------------------------------
# 2. GPT-2 as a frozen feature extractor.
#    We take the last non-padding token's hidden state, since in a
#    left-to-right model that's the only position that has seen the
#    whole review.
# ---------------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

gpt2 = GPT2Model.from_pretrained("gpt2").to(DEVICE)
gpt2.eval()
for p in gpt2.parameters():
    p.requires_grad = False


@torch.no_grad()
def get_features(texts, batch_size=16, max_len=512):
    features = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len).to(DEVICE)
        out = gpt2(**enc)
        hidden = out.last_hidden_state  # (batch, seq_len, hidden_dim)

        last_idx = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=DEVICE)
        pooled = hidden[batch_idx, last_idx]  # (batch, hidden_dim)

        features.append(pooled.cpu())
    return torch.cat(features, dim=0)


# ---------------------------------------------------------------------
# 3. A small MLP classification head.
# ---------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------
# 4. Main script: load data, extract features, train, evaluate.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # --- Load data (replace with your real paths) ---
    texts, labels = load_data("data/train.csv")
    test_texts, test_labels = load_data("data/test.csv")

    # --- Split off a validation set from the training data ---
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # --- Extract frozen GPT-2 features (slow-ish once, then cached in memory) ---
    train_feats = get_features(train_texts).to(DEVICE)
    val_feats = get_features(val_texts).to(DEVICE)
    test_feats = get_features(test_texts).to(DEVICE)

    train_labels_t = torch.tensor(train_labels, device=DEVICE)
    val_labels_t = torch.tensor(val_labels, device=DEVICE)

    # --- Class weights to handle imbalance (inverse frequency) ---
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    weight_neg = len(train_labels) / (2 * n_neg)
    weight_pos = len(train_labels) / (2 * n_pos)
    class_weights = torch.tensor([weight_neg, weight_pos], device=DEVICE)

    # --- Train the MLP head ---
    head = MLPHead(in_dim=train_feats.size(1)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 20
    for epoch in range(1, EPOCHS + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(train_feats)
        loss = criterion(logits, train_labels_t)
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels_t).float().mean().item()
        print(f"Epoch {epoch:2d} | train loss: {loss.item():.4f} | val acc: {val_acc:.4f}")

    # --- Evaluate on the test set ---
    head.eval()
    with torch.no_grad():
        test_preds = head(test_feats).argmax(dim=1).cpu().numpy()
    print("\nTest set results:")
    print(classification_report(test_labels, test_preds, target_names=["neg", "pos"]))