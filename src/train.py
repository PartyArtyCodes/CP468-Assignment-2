"""
Training script for the LSTM seq2seq summarization model.

Usage:
    python train.py

Expects data/processed/{train,validation,test}.jsonl and
data/processed/{source,target}_vocab.json to already exist (i.e. run
prepare_data.py and build_vocab.py first).

Reports: model size (parameter count), training time, hardware used,
and ROUGE-1/2/L on the validation set at the end of training.
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer

from dataset import build_datasets
from model import Encoder, Decoder, Seq2Seq, count_parameters

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
DATA_DIR = "data/processed"
CHECKPOINT_DIR = Path("checkpoints")

EMB_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.3

BATCH_SIZE = 256
N_EPOCHS = 3
LEARNING_RATE = 1e-3
CLIP = 1.0
TEACHER_FORCING_RATIO = 0.5

MAX_SOURCE_LEN = 150
MAX_TARGET_LEN = 40

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0.0

    for source, target in dataloader:
        source, target = source.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(source, target, teacher_forcing_ratio=TEACHER_FORCING_RATIO)

        output_dim = output.shape[-1]
        loss = criterion(
            output[:, 1:].reshape(-1, output_dim),
            target[:, 1:].reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0.0

    for source, target in dataloader:
        source, target = source.to(DEVICE), target.to(DEVICE)

        output = model(source, target, teacher_forcing_ratio=0.0)
        output_dim = output.shape[-1]
        loss = criterion(
            output[:, 1:].reshape(-1, output_dim),
            target[:, 1:].reshape(-1),
        )

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def ids_to_text(ids, index_to_token, eos_idx, pad_idx, sos_idx):
    words = []

    for token_id in ids:
        token_id = int(token_id)

        if token_id == eos_idx:
            break
        if token_id in (pad_idx, sos_idx):
            continue

        words.append(index_to_token[token_id])

    return " ".join(words)


@torch.no_grad()
def evaluate_rouge(model, dataloader, target_vocab, target_i2t, max_examples=500):
    model.eval()

    sos_idx = target_vocab["<sos>"]
    eos_idx = target_vocab["<eos>"]
    pad_idx = target_vocab["<pad>"]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n_scored = 0

    for source, target in dataloader:
        source = source.to(DEVICE)
        generated = model.greedy_decode(source, sos_idx, eos_idx, max_len=target.size(1))

        for i in range(source.size(0)):
            if n_scored >= max_examples:
                break

            hyp = ids_to_text(generated[i].tolist(), target_i2t, eos_idx, pad_idx, sos_idx)
            ref = ids_to_text(target[i].tolist(), target_i2t, eos_idx, pad_idx, sos_idx)

            if not ref.strip():
                continue

            scores = scorer.score(ref, hyp)
            for key in totals:
                totals[key] += scores[key].fmeasure

            n_scored += 1

        if n_scored >= max_examples:
            break

    return {key: value / max(n_scored, 1) for key, value in totals.items()}


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ds, val_ds, test_ds, source_vocab, target_vocab, target_i2t = build_datasets(
        DATA_DIR, MAX_SOURCE_LEN, MAX_TARGET_LEN
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    encoder = Encoder(len(source_vocab), source_vocab["<pad>"], EMB_DIM, HIDDEN_DIM, dropout=DROPOUT)
    decoder = Decoder(len(target_vocab), target_vocab["<pad>"], EMB_DIM, HIDDEN_DIM, dropout=DROPOUT)
    model = Seq2Seq(encoder, decoder, source_vocab["<pad>"], DEVICE).to(DEVICE)

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab["<pad>"])

    best_val_loss = float("inf")
    start_time = time.time()

    history = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_dl, optimizer, criterion, CLIP)
        val_loss = evaluate_loss(model, val_dl, criterion)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{N_EPOCHS} | "
            f"train_loss {train_loss:.4f} (ppl {math.exp(train_loss):.2f}) | "
            f"val_loss {val_loss:.4f} (ppl {math.exp(val_loss):.2f}) | "
            f"{epoch_time:.1f}s"
        )

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": epoch_time}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            print("  -> saved new best checkpoint")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time / 60:.1f} minutes")

    # Load best checkpoint before final evaluation
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=DEVICE))

    print("\nComputing ROUGE on validation set...")
    rouge_scores = evaluate_rouge(model, val_dl, target_vocab, target_i2t)
    print("ROUGE (validation):", rouge_scores)

    report = {
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "trainable_parameters": n_params,
        "total_training_time_seconds": total_time,
        "best_val_loss": best_val_loss,
        "rouge_validation": rouge_scores,
        "history": history,
        "hyperparameters": {
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "teacher_forcing_ratio": TEACHER_FORCING_RATIO,
            "seed": SEED,
        },
    }

    report_path = CHECKPOINT_DIR / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nSaved training report to {report_path}")
    print("Report includes model size, training time, and hardware")


if __name__ == "__main__":
    main()
#First test:
"""
Device: cpu
Trainable parameters: 18,524,281
Epoch 1/3 | train_loss 6.5315 (ppl 686.45) | val_loss 6.1094 (ppl 450.07) | 1651.0s
  -> saved new best checkpoint
Epoch 2/3 | train_loss 5.8745 (ppl 355.86) | val_loss 5.9789 (ppl 395.00) | 2003.6s
  -> saved new best checkpoint
Epoch 3/3 | train_loss 5.6216 (ppl 276.33) | val_loss 5.8501 (ppl 347.26) | 2151.7s
  -> saved new best checkpoint

Total training time: 96.8 minutes
"""