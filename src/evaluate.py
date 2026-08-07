"""
Final evaluation: run this AFTER train.py and llm_baseline.py have both
completed.

What this does:
  1. Reloads the best LSTM checkpoint and generates predictions on the TEST
     set (train.py only reported ROUGE on validation).
  2. Computes ROUGE-1/2/L for the LSTM on the test set.
  3. Loads the LLM outputs from results/llm_baseline_outputs.jsonl and
     computes ROUGE-1/2/L per prompt variant (zero_shot, few_shot) on the
     SAME test examples.
  4. Builds the required 10-example side-by-side qualitative comparison
     table (source, reference, LSTM output, LLM output) and saves it as
     both JSON and a readable Markdown table you can paste into your report.

Usage:
    python evaluate.py --max_source_len 150 --max_target_len 40

IMPORTANT: --max_source_len / --max_target_len must match whatever values
you actually used in train.py when you trained the checkpoint (defaults
below match train.py's original defaults; override if you changed them).
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer

from dataset import build_datasets
from model import Encoder, Decoder, Seq2Seq

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
DATA_DIR = "data/processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_training_config():
    report_path = CHECKPOINT_DIR / "training_report.json"

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    return report["hyperparameters"]


def rebuild_model(source_vocab, target_vocab, config):
    encoder = Encoder(
        len(source_vocab),
        source_vocab["<pad>"],
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
    decoder = Decoder(
        len(target_vocab),
        target_vocab["<pad>"],
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    )
    model = Seq2Seq(encoder, decoder, source_vocab["<pad>"], DEVICE).to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    return model


@torch.no_grad()
def generate_lstm_predictions(model, test_ds, target_vocab, target_i2t, source_i2t=None, batch_size=32):
    dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    sos_idx = target_vocab["<sos>"]
    eos_idx = target_vocab["<eos>"]
    pad_idx = target_vocab["<pad>"]

    predictions = []
    idx = 0

    for source, target in dataloader:
        source = source.to(DEVICE)
        generated = model.greedy_decode(source, sos_idx, eos_idx, max_len=target.size(1))

        for i in range(source.size(0)):
            example = test_ds.examples[idx]

            hyp = ids_to_text(generated[i].tolist(), target_i2t, eos_idx, pad_idx, sos_idx)
            ref = ids_to_text(target[i].tolist(), target_i2t, eos_idx, pad_idx, sos_idx)

            predictions.append(
                {
                    "gem_id": example.get("gem_id"),
                    "article": example["article"],
                    "reference_summary": ref if ref.strip() else example["summary"],
                    "lstm_summary": hyp,
                }
            )
            idx += 1

    return predictions


def compute_rouge(pairs, hyp_key, ref_key="reference_summary"):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n_scored = 0

    for record in pairs:
        ref = record[ref_key]
        hyp = record[hyp_key]

        if not ref.strip():
            continue

        scores = scorer.score(ref, hyp)
        for key in totals:
            totals[key] += scores[key].fmeasure

        n_scored += 1

    return {key: value / max(n_scored, 1) for key, value in totals.items()}, n_scored


def load_llm_outputs():
    llm_path = RESULTS_DIR / "llm_baseline_outputs.jsonl"

    if not llm_path.exists():
        print(f"WARNING: {llm_path} not found. Run llm_baseline.py first. "
              "Skipping LLM comparison.")
        return None

    records = []
    with open(llm_path, "r", encoding="utf-8") as file:
        for line in file:
            records.append(json.loads(line))

    by_variant = {}
    for record in records:
        by_variant.setdefault(record["prompt_variant"], []).append(record)

    return by_variant


def build_qualitative_table(lstm_predictions, llm_by_variant, n_examples=10):
    """Pick a spread of examples (not just the first N) so they illustrate
    varied behavior, per the assignment's requirement to avoid cherry-picking
    only best cases."""

    llm_zero_shot_by_id = {}
    if llm_by_variant and "zero_shot" in llm_by_variant:
        llm_zero_shot_by_id = {r["gem_id"]: r for r in llm_by_variant["zero_shot"]}

    # Only sample from examples that actually have an LLM output, otherwise
    # a spread across the full test set mostly misses the (smaller) LLM subset.
    eligible = [p for p in lstm_predictions if p["gem_id"] in llm_zero_shot_by_id]

    if not eligible:
        print("WARNING: no overlap between LSTM predictions and LLM outputs — "
              "check that llm_baseline.py and evaluate.py used the same test examples.")
        eligible = lstm_predictions  # fallback so the script doesn't crash

    step = max(len(eligible) // n_examples, 1)
    sample_indices = list(range(0, len(eligible), step))[:n_examples]

    rows = []
    for i in sample_indices:
        pred = eligible[i]
        gem_id = pred["gem_id"]
        llm_record = llm_zero_shot_by_id.get(gem_id)

        rows.append(
            {
                "gem_id": gem_id,
                "source": pred["article"][:400],  # truncate long articles for readability
                "reference": pred["reference_summary"],
                "lstm_output": pred["lstm_summary"],
                "llm_output": llm_record["llm_summary"] if llm_record else "(not found)",
                "error_category": "",  # TODO: fill in manually after reading each row
            }
        )

    return rows


def save_markdown_table(rows, output_path):
    lines = [
        "| # | Source (truncated) | Reference | LSTM Output | LLM Output | Error Category |",
        "|---|---|---|---|---|---|",
    ]

    for i, row in enumerate(rows, start=1):
        source = row["source"].replace("\n", " ").replace("|", "/")
        reference = row["reference"].replace("\n", " ").replace("|", "/")
        lstm_out = row["lstm_output"].replace("\n", " ").replace("|", "/")
        llm_out = row["llm_output"].replace("\n", " ").replace("|", "/")

        lines.append(
            f"| {i} | {source} | {reference} | {lstm_out} | {llm_out} | {row['error_category']} |"
        )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_source_len", type=int, default=300,
                         help="Must match the value used in train.py")
    parser.add_argument("--max_target_len", type=int, default=60,
                         help="Must match the value used in train.py")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and vocab...")
    _, _, test_ds, source_vocab, target_vocab, target_i2t = build_datasets(
        DATA_DIR, args.max_source_len, args.max_target_len
    )

    print("Loading training config and best checkpoint...")
    config = load_training_config()
    model = rebuild_model(source_vocab, target_vocab, config)

    print(f"Generating LSTM predictions on {len(test_ds)} test examples...")
    lstm_predictions = generate_lstm_predictions(model, test_ds, target_vocab, target_i2t)

    lstm_output_path = RESULTS_DIR / "lstm_test_outputs.jsonl"
    with open(lstm_output_path, "w", encoding="utf-8") as file:
        for record in lstm_predictions:
            file.write(json.dumps(record) + "\n")
    print(f"Saved LSTM test predictions to {lstm_output_path}")

    print("Computing LSTM test ROUGE...")
    lstm_rouge, lstm_n = compute_rouge(lstm_predictions, hyp_key="lstm_summary")
    print(f"LSTM test ROUGE (n={lstm_n}):", lstm_rouge)

    print("\nLoading LLM baseline outputs...")
    llm_by_variant = load_llm_outputs()

    llm_rouge_by_variant = {}
    if llm_by_variant:
        for variant_name, records in llm_by_variant.items():
            rouge, n = compute_rouge(records, hyp_key="llm_summary")
            llm_rouge_by_variant[variant_name] = {"rouge": rouge, "n_examples": n}
            print(f"LLM [{variant_name}] test ROUGE (n={n}):", rouge)

    print("\nBuilding qualitative comparison table...")
    table_rows = build_qualitative_table(lstm_predictions, llm_by_variant, n_examples=10)

    table_json_path = RESULTS_DIR / "qualitative_comparison.json"
    with open(table_json_path, "w", encoding="utf-8") as file:
        json.dump(table_rows, file, indent=2)

    table_md_path = RESULTS_DIR / "qualitative_comparison.md"
    save_markdown_table(table_rows, table_md_path)

    print(f"Saved qualitative comparison table to {table_json_path} and {table_md_path}")
    print("\nNOTE: open qualitative_comparison.md and manually fill in the "
          "'Error Category' column for each row (e.g. repetition, hallucination, "
          "under-translation, fluent-but-wrong, OOV failure) as required by the assignment.")

    final_report = {
        "lstm_test_rouge": lstm_rouge,
        "lstm_n_examples": lstm_n,
        "llm_test_rouge_by_variant": llm_rouge_by_variant,
    }

    final_report_path = RESULTS_DIR / "final_evaluation_report.json"
    with open(final_report_path, "w", encoding="utf-8") as file:
        json.dump(final_report, file, indent=2)

    print(f"\nSaved final evaluation summary to {final_report_path}")
    print("This has your headline numbers for the report's results section.")


if __name__ == "__main__":
    main()