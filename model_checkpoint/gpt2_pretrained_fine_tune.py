import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

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
        next(reader)  # skip header row
        for row in reader:
            texts.append(row[1])
            labels.append(int(row[2]))
    return texts, labels


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, max_len=512):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), return_tensors="pt", padding=True,
                     truncation=True, max_length=512)
    return enc, torch.tensor(labels, dtype=torch.long)


class GPT2Classifier(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2") 
        gpt2_dim = self.gpt2.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(gpt2_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, enc):
        out = self.gpt2(**enc)
        hidden = out.last_hidden_state  # (batch, seq_len, hidden_dim)

        last_idx = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch_idx, last_idx]  # (batch, hidden_dim)

        return self.head(pooled)


def compute_class_weights(labels):
    n_total = len(labels)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    weight_neg = n_total / (2 * n_neg)
    weight_pos = n_total / (2 * n_pos)
    return torch.tensor([weight_neg, weight_pos], device=DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over `loader`. Pass an optimizer to train; omit it to
    just evaluate (no gradient updates)."""
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for enc, yb in loader:
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            yb = yb.to(DEVICE)

            if train:
                optimizer.zero_grad()
            logits = model(enc)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            n += yb.size(0)

    return total_loss / n, correct / n


def make_optimizer(model):
    return torch.optim.AdamW([
        {"params": model.gpt2.parameters(), "lr": 2e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)


def get_predictions(model, loader):
    """Run the model over a loader and return (true_labels, predicted_labels)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for enc, yb in loader:
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            preds = model(enc).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
    return all_labels, all_preds


if __name__ == "__main__":
    BATCH_SIZE = 6   # reduce further if you hit out-of-memory issues
    EPOCHS = 5        # set this based on your exploratory runs
    VAL_SIZE = 0.0    # set to 0.0 for the final run

    texts, labels = load_data("data/train.csv")
    test_texts, test_labels = load_data("data/public_test.csv")

    if VAL_SIZE > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=VAL_SIZE, stratify=labels, random_state=42
        )
        val_loader = DataLoader(ReviewDataset(val_texts, val_labels),
                                 batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    else:
        train_texts, train_labels = texts, labels
        val_loader = None

    train_loader = DataLoader(ReviewDataset(train_texts, train_labels),
                               batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = GPT2Classifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_labels))
    optimizer = make_optimizer(model)

    mode = "with held-out validation" if val_loader is not None else "on full training set (no validation)"
    print(f"=== Training for {EPOCHS} epochs, {mode} ===")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        line = f"Epoch {epoch} | train loss: {train_loss:.4f} | train acc: {train_acc:.4f}"

        if val_loader is not None:
            val_loss, val_acc = run_epoch(model, val_loader, criterion)
            eval_labels, eval_preds = get_predictions(model, val_loader)
            cm = confusion_matrix(eval_labels, eval_preds)

            print(line + f" | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")
            print("  Val confusion matrix (rows=true, cols=predicted; order=[neg, pos]):")
            print(f"  {cm[0]}")
            print(f"  {cm[1]}")
        else:
            print(line)

    SAVE_PATH = "gpt2_sentiment_model.pt"
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nSaved final model weights to {SAVE_PATH}")

    test_loader = DataLoader(ReviewDataset(test_texts, test_labels),
                              batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    eval_labels, eval_preds = get_predictions(model, test_loader)

    print("\nTest set results:")
    print(classification_report(eval_labels, eval_preds, target_names=["neg", "pos"]))