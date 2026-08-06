import json
from pathlib import Path

from datasets import load_dataset


SEED = 42

MIN_ARTICLE_WORDS = 50
MAX_ARTICLE_WORDS = 300

MIN_SUMMARY_WORDS = 5
MAX_SUMMARY_WORDS = 60

SPLIT_SIZES = {
    "train": 20_000,
    "validation": 1_000,
    "test": 500
}

OUTPUT_DIR = Path("data/processed")


def count_words(text):
    return len(text.split())


def valid_example(example):
    article_length = count_words(example["source"])
    summary_length = count_words(example["target"])

    return (
        MIN_ARTICLE_WORDS <= article_length <= MAX_ARTICLE_WORDS
        and MIN_SUMMARY_WORDS <= summary_length <= MAX_SUMMARY_WORDS
    )


def prepare_split(dataset, split_name, requested_size):
    print(f"\nFiltering {split_name} split...")

    filtered = dataset[split_name].filter(valid_example)

    print("Examples remaining after filtering:", len(filtered))

    if len(filtered) < requested_size:
        raise ValueError(
            f"Only {len(filtered)} valid examples are available, "
            f"but {requested_size} were requested."
        )

    selected = (
        filtered
        .shuffle(seed=SEED)
        .select(range(requested_size))
        .select_columns(["gem_id", "gem_parent_id", "source", "target"])
        .rename_column("source", "article")
        .rename_column("target", "summary")
    )

    output_path = OUTPUT_DIR / f"{split_name}.jsonl"

    selected.to_json(
        output_path,
        orient="records",
        lines=True
    )

    print(f"Saved {len(selected)} examples to {output_path}")

    return selected


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading English WikiLingua...")

    dataset = load_dataset(
        "GEM/wiki_lingua",
        "en",
        trust_remote_code=True
    )

    prepared_splits = {}

    for split_name, requested_size in SPLIT_SIZES.items():
        prepared_splits[split_name] = prepare_split(
            dataset,
            split_name,
            requested_size
        )

    metadata = {
        "dataset": "GEM/wiki_lingua",
        "configuration": "en",
        "license": "CC BY-NC-SA 3.0",
        "random_seed": SEED,
        "minimum_article_words": MIN_ARTICLE_WORDS,
        "maximum_article_words": MAX_ARTICLE_WORDS,
        "minimum_summary_words": MIN_SUMMARY_WORDS,
        "maximum_summary_words": MAX_SUMMARY_WORDS,
        "split_sizes": SPLIT_SIZES
    }

    metadata_path = OUTPUT_DIR / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nSaved metadata to {metadata_path}")

    print("\nFINAL SPLITS:")
    for split_name, split in prepared_splits.items():
        print(f"{split_name}: {len(split)} examples")


if __name__ == "__main__":
    main()