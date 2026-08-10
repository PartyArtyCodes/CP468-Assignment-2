# LSTM vs. LLM: Text Summarization (WikiLingua English)

CP468 Course Project — sequence-to-sequence summarization with an LSTM
(encoder-decoder + attention, trained from scratch) compared against an LLM
(Claude API, zero-shot and few-shot) on the same held-out test set.

## Requirements

- Python 3.11+ (tested on 3.13)
- ~2 GB free disk space (dataset + checkpoints)
- An Anthropic API key (for the LLM baseline step only)

## 1. Setup

```powershell
# from the project's src/ folder
python -m pip install -r requirements.txt
```

`requirements.txt` should contain (pin exact versions with `pip freeze` once
your environment is finalized):
```
torch
datasets<4.0
huggingface_hub<0.35
rouge-score
anthropic
```

> Note: `datasets` must be pinned below version 4.0 — newer versions dropped
> support for the loading-script format that `GEM/wiki_lingua` still uses.

## 2. Prepare the data

Downloads WikiLingua (English) from Hugging Face, filters, splits, and saves
train/validation/test as `.jsonl` under `data/processed/`.

```powershell
python prepare_data.py
```

Expected output ends with `FINAL SPLITS:` showing 20,000 / 1,000 / 500 examples.

## 3. Build vocabularies

Reads only `train.jsonl` and builds source/target vocab JSON files.

```powershell
python build_vocab.py
```

## 4. Validate the data pipeline

Checks split sizes, confirms no train/val/test leakage (via `gem_parent_id`),
computes OOV rates, and hashes each split file for reproducibility.

```powershell
python validate_data.py
```

Expected output ends with `DATA VALIDATION PASSED`.

## 5. Train the LSTM model

Trains the bidirectional LSTM encoder + Bahdanau attention + LSTM decoder
model. Saves the best checkpoint (by validation loss) to
`checkpoints/best_model.pt` and a full training report to
`checkpoints/training_report.json`.

```powershell
python train.py
```

Key hyperparameters (edit at the top of `train.py` before running):
- `EMB_DIM = 128`, `HIDDEN_DIM = 256`
- `BATCH_SIZE = 256`
- `N_EPOCHS = 3`
- `MAX_SOURCE_LEN = 150`, `MAX_TARGET_LEN = 40`

On CPU, this takes roughly 25-35 minutes per epoch (~1.5-2 hours total for 3
epochs) — timing will vary by machine.

## 6. Run the LLM baseline

Requires an Anthropic API key. Runs Claude on the same test set in both
zero-shot and few-shot (k=3) prompt settings, and reports token usage + cost.

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
python llm_baseline.py
```

Defaults to 40 test examples (edit `--n_examples` to change; the default is
set intentionally low to control API cost — see `results/llm_baseline_cost_report.json`
for actual spend after running).

## 7. Final evaluation

Generates LSTM predictions on the full test set, computes ROUGE-1/2/L for
both systems, and builds the 10-example qualitative comparison table.

```powershell
python evaluate.py --max_source_len 150 --max_target_len 40
```

`--max_source_len` / `--max_target_len` must match whatever values were
used in `train.py` (defaults above match this project's actual run).

**Outputs:**
- `results/lstm_test_outputs.jsonl` — LSTM predictions on all 500 test examples
- `results/final_evaluation_report.json` — headline ROUGE numbers for both systems
- `results/qualitative_comparison.md` / `.json` — 10-example side-by-side table
  (fill in the "Error Category" column manually after review)

## Full run, start to finish

```powershell
python -m pip install -r requirements.txt
python prepare_data.py
python build_vocab.py
python validate_data.py
python train.py
$env:ANTHROPIC_API_KEY = "sk-ant-your-key-here"
python llm_baseline.py
python evaluate.py --max_source_len 150 --max_target_len 40
```

## Reproducibility notes

- Random seed fixed at `42` throughout (data shuffling, vocab tie-breaking,
  PyTorch training).
- Dataset license: WikiLingua (GEM/wiki_lingua, English config) — CC BY-NC-SA 3.0.
- Hardware used for the reported results: CPU only (no GPU), see
  `checkpoints/training_report.json` for exact specs and timing.
