import json
from collections import Counter
from pathlib import Path

from text_utils import tokenize


TRAIN_FILE = Path("data/processed/train.jsonl")
OUTPUT_DIR = Path("data/processed")

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<sos>",
    "<eos>"
]

SOURCE_VOCAB_SIZE = 30_000
TARGET_VOCAB_SIZE = 15_000

SOURCE_MIN_FREQUENCY = 3
TARGET_MIN_FREQUENCY = 2


def read_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)


def create_vocabulary(counter, maximum_size, minimum_frequency):
    eligible_tokens = [
        (token, frequency)
        for token, frequency in counter.items()
        if frequency >= minimum_frequency
    ]

    eligible_tokens.sort(
        key=lambda item: (-item[1], item[0])
    )

    available_places = maximum_size - len(SPECIAL_TOKENS)

    selected_tokens = [
        token
        for token, _ in eligible_tokens[:available_places]
    ]

    index_to_token = SPECIAL_TOKENS + selected_tokens

    token_to_index = {
        token: index
        for index, token in enumerate(index_to_token)
    }

    return {
        "token_to_index": token_to_index,
        "index_to_token": index_to_token,
        "size": len(index_to_token),
        "minimum_frequency": minimum_frequency,
        "maximum_size": maximum_size
    }


def save_vocabulary(vocabulary, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            vocabulary,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():
    source_counter = Counter()
    target_counter = Counter()
    example_count = 0

    print("Reading training data...")

    for example in read_training_data(TRAIN_FILE):
        source_counter.update(tokenize(example["article"]))
        target_counter.update(tokenize(example["summary"]))
        example_count += 1

    print("Training examples read:", example_count)
    print("Unique article tokens:", len(source_counter))
    print("Unique summary tokens:", len(target_counter))

    source_vocabulary = create_vocabulary(
        source_counter,
        SOURCE_VOCAB_SIZE,
        SOURCE_MIN_FREQUENCY
    )

    target_vocabulary = create_vocabulary(
        target_counter,
        TARGET_VOCAB_SIZE,
        TARGET_MIN_FREQUENCY
    )

    source_path = OUTPUT_DIR / "source_vocab.json"
    target_path = OUTPUT_DIR / "target_vocab.json"

    save_vocabulary(source_vocabulary, source_path)
    save_vocabulary(target_vocabulary, target_path)

    print("\nSource vocabulary size:", source_vocabulary["size"])
    print("Target vocabulary size:", target_vocabulary["size"])

    print("\nSaved:")
    print(source_path)
    print(target_path)


if __name__ == "__main__":
    main()