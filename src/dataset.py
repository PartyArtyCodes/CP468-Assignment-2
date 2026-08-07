import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from text_utils import tokenize


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    return vocab["token_to_index"], vocab["index_to_token"]


def load_jsonl(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


class SummarizationDataset(Dataset):
    """Reads a train/validation/test .jsonl produced by prepare_data.py and
    encodes each (article, summary) pair using the vocab produced by
    build_vocab.py. Sequences are padded/truncated to a fixed length so they
    can be batched directly."""

    def __init__(
        self,
        jsonl_path,
        source_vocab,
        target_vocab,
        max_source_len=300,
        max_target_len=60,
    ):
        self.examples = load_jsonl(jsonl_path)
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def encode(text, vocab, max_len):
        tokens = tokenize(text)
        unk_idx = vocab["<unk>"]

        ids = [vocab.get(token, unk_idx) for token in tokens]
        ids = [vocab["<sos>"]] + ids[: max_len - 2] + [vocab["<eos>"]]
        ids = ids + [vocab["<pad>"]] * (max_len - len(ids))

        return ids

    def __getitem__(self, idx):
        example = self.examples[idx]

        source_ids = self.encode(example["article"], self.source_vocab, self.max_source_len)
        target_ids = self.encode(example["summary"], self.target_vocab, self.max_target_len)

        return torch.tensor(source_ids), torch.tensor(target_ids)


def build_datasets(data_dir="data/processed", max_source_len=300, max_target_len=60):
    """Convenience loader: reads vocabs + all three splits and returns
    (train_ds, val_ds, test_ds, source_vocab, target_vocab, target_i2t)."""

    data_dir = Path(data_dir)

    source_vocab, _ = load_vocab(data_dir / "source_vocab.json")
    target_vocab, target_i2t = load_vocab(data_dir / "target_vocab.json")

    train_ds = SummarizationDataset(
        data_dir / "train.jsonl", source_vocab, target_vocab, max_source_len, max_target_len
    )
    val_ds = SummarizationDataset(
        data_dir / "validation.jsonl", source_vocab, target_vocab, max_source_len, max_target_len
    )
    test_ds = SummarizationDataset(
        data_dir / "test.jsonl", source_vocab, target_vocab, max_source_len, max_target_len
    )

    return train_ds, val_ds, test_ds, source_vocab, target_vocab, target_i2t