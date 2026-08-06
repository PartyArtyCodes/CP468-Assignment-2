import hashlib
import json
from pathlib import Path

from text_utils import tokenize


DATA_DIR = Path("data/processed")

SPLIT_FILES = {
    "train": DATA_DIR / "train.jsonl",
    "validation": DATA_DIR / "validation.jsonl",
    "test": DATA_DIR / "test.jsonl"
}

EXPECTED_COUNTS = {
    "train": 20_000,
    "validation": 1_000,
    "test": 500
}


def load_jsonl(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def calculate_oov_rate(examples, field_name, vocabulary):
    total_tokens = 0
    unknown_tokens = 0

    for example in examples:
        tokens = tokenize(example[field_name])

        total_tokens += len(tokens)
        unknown_tokens += sum(
            token not in vocabulary
            for token in tokens
        )

    return {
        "total_tokens": total_tokens,
        "unknown_tokens": unknown_tokens,
        "oov_rate": unknown_tokens / total_tokens
    }


def main():
    splits = {
        name: load_jsonl(path)
        for name, path in SPLIT_FILES.items()
    }

    print("SPLIT COUNTS")

    for name, examples in splits.items():
        actual_count = len(examples)
        expected_count = EXPECTED_COUNTS[name]

        print(f"{name}: {actual_count}")

        if actual_count != expected_count:
            raise ValueError(
                f"{name} contains {actual_count} examples; "
                f"expected {expected_count}."
            )

    split_ids = {
        name: {
            example["gem_parent_id"]
            for example in examples
        }
        for name, examples in splits.items()
    }

    overlaps = {
        "train_validation": len(
            split_ids["train"] & split_ids["validation"]
        ),
        "train_test": len(
            split_ids["train"] & split_ids["test"]
        ),
        "validation_test": len(
            split_ids["validation"] & split_ids["test"]
        )
    }

    print("\nSPLIT OVERLAPS")

    for name, count in overlaps.items():
        print(f"{name}: {count}")

        if count != 0:
            raise ValueError(f"Data leakage detected in {name}.")

    with open(
        DATA_DIR / "source_vocab.json",
        "r",
        encoding="utf-8"
    ) as file:
        source_vocabulary = json.load(file)["token_to_index"]

    with open(
        DATA_DIR / "target_vocab.json",
        "r",
        encoding="utf-8"
    ) as file:
        target_vocabulary = json.load(file)["token_to_index"]

    oov_results = {}

    print("\nOUT-OF-VOCABULARY RATES")

    for split_name in ["validation", "test"]:
        source_result = calculate_oov_rate(
            splits[split_name],
            "article",
            source_vocabulary
        )

        target_result = calculate_oov_rate(
            splits[split_name],
            "summary",
            target_vocabulary
        )

        oov_results[split_name] = {
            "article": source_result,
            "summary": target_result
        }

        print(
            f"{split_name} article OOV: "
            f"{source_result['oov_rate']:.2%}"
        )

        print(
            f"{split_name} summary OOV: "
            f"{target_result['oov_rate']:.2%}"
        )

    hashes = {
        name: calculate_sha256(path)
        for name, path in SPLIT_FILES.items()
    }

    report = {
        "split_counts": {
            name: len(examples)
            for name, examples in splits.items()
        },
        "split_overlaps": overlaps,
        "oov_results": oov_results,
        "sha256": hashes
    }

    report_path = DATA_DIR / "validation_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nDATA VALIDATION PASSED")
    print("Saved:", report_path)


if __name__ == "__main__":
    main()