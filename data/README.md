# Dataset

## Source

This project uses the English configuration of the WikiLingua dataset:

- Hugging Face identifier: `GEM/wiki_lingua`
- Configuration: `en`
- Task: English abstractive article summarization
- Input field: `source`
- Target field: `target`
- License: CC BY-NC-SA 3.0

WikiLingua contains WikiHow article-summary pairs and was introduced by
Ladhak et al. (2020).

## Fixed Dataset Selection

The official WikiLingua training, validation and test splits were preserved.
A fixed random seed of 42 was used when selecting the project subsets.

The final split sizes are:

- Training: 20,000 examples
- Validation: 1,000 examples
- Test: 500 examples

Only examples satisfying the following conditions were retained:

- Article length: 50–300 whitespace-separated words
- Summary length: 5–60 whitespace-separated words

The generated JSONL files are excluded from Git because they can be
reproduced using the provided scripts.

## Vocabulary

Separate source and target vocabularies were constructed using only the
training split.

- Source minimum frequency: 3
- Target minimum frequency: 2
- Maximum source vocabulary size: 30,000
- Maximum target vocabulary size: 15,000
- Final source vocabulary size: 25,325
- Final target vocabulary size: 12,793

The special tokens are:

- `<pad>`
- `<unk>`
- `<sos>`
- `<eos>`

## Validation Results

No examples overlap between the training, validation and test splits.

Out-of-vocabulary rates:

| Split | Article OOV | Summary OOV |
|---|---:|---:|
| Validation | 1.09% | 2.54% |
| Test | 1.06% | 2.41% |

## Reproducing the Dataset

From the repository root, run:

```powershell
python src\prepare_data.py
python src\build_vocab.py
python src\validate_data.py