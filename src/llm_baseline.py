"""
LLM baseline for the summarization task.

Runs Claude on the *same* test.jsonl used for the LSTM model, under two
prompt variants (zero-shot and few-shot), and saves outputs + a cost
estimate so results can be compared side-by-side with the LSTM in
evaluate.py.

Requires: pip install anthropic
Requires: ANTHROPIC_API_KEY set in your environment.

Usage:
    python llm_baseline.py --n_examples 500
"""

import argparse
import json
import time
from pathlib import Path

import anthropic


MODEL_NAME = "claude-sonnet-4-6"  # pick whichever low-cost model your group wants to use
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results")

# Approximate USD per 1M tokens -- update these to the current published
# pricing for MODEL_NAME before reporting your final cost estimate.
PRICE_PER_1M_INPUT_TOKENS = 3.00
PRICE_PER_1M_OUTPUT_TOKENS = 15.00

ZERO_SHOT_TEMPLATE = (
    "Summarize the following article in 1-2 sentences. "
    "Only output the summary, with no preamble or explanation.\n\n"
    "Article: {article}\n\nSummary:"
)

FEW_SHOT_EXAMPLES = [
    # TODO: replace these with 3-5 real (article, summary) pairs taken from
    # your TRAIN split (never from validation/test, to avoid leakage).
    {
        "article": "PLACEHOLDER_TRAIN_ARTICLE_1",
        "summary": "PLACEHOLDER_TRAIN_SUMMARY_1",
    },
    {
        "article": "PLACEHOLDER_TRAIN_ARTICLE_2",
        "summary": "PLACEHOLDER_TRAIN_SUMMARY_2",
    },
    {
        "article": "PLACEHOLDER_TRAIN_ARTICLE_3",
        "summary": "PLACEHOLDER_TRAIN_SUMMARY_3",
    },
]


def build_few_shot_prompt(article):
    parts = [
        "Summarize the article in 1-2 sentences. "
        "Only output the summary, with no preamble or explanation.\n"
    ]

    for example in FEW_SHOT_EXAMPLES:
        parts.append(f"Article: {example['article']}\nSummary: {example['summary']}\n")

    parts.append(f"Article: {article}\nSummary:")

    return "\n".join(parts)


def load_jsonl(file_path):
    examples = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            examples.append(json.loads(line))

    return examples


def run_prompt_variant(client, examples, prompt_fn, variant_name, max_tokens=100):
    results = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, example in enumerate(examples):
        prompt = prompt_fn(example["article"])

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        results.append(
            {
                "gem_id": example.get("gem_id"),
                "article": example["article"],
                "reference_summary": example["summary"],
                "llm_summary": summary,
                "prompt_variant": variant_name,
            }
        )

        if (i + 1) % 25 == 0:
            print(f"  [{variant_name}] {i + 1}/{len(examples)} done")

        time.sleep(0.1)  # gentle rate limiting

    cost = (
        total_input_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
        + total_output_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
    )

    usage = {
        "variant": variant_name,
        "n_examples": len(examples),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_cost_usd": round(cost, 4),
    }

    return results, usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=None, help="Limit test set size (default: use all)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_examples = load_jsonl(DATA_DIR / "test.jsonl")
    if args.n_examples:
        test_examples = test_examples[: args.n_examples]

    print(f"Running LLM baseline on {len(test_examples)} test examples using {MODEL_NAME}")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    all_results = []
    all_usage = []

    print("\n--- Zero-shot ---")
    zero_shot_results, zero_shot_usage = run_prompt_variant(
        client,
        test_examples,
        lambda article: ZERO_SHOT_TEMPLATE.format(article=article),
        variant_name="zero_shot",
    )
    all_results.extend(zero_shot_results)
    all_usage.append(zero_shot_usage)

    print("\n--- Few-shot (k={}) ---".format(len(FEW_SHOT_EXAMPLES)))
    few_shot_results, few_shot_usage = run_prompt_variant(
        client,
        test_examples,
        build_few_shot_prompt,
        variant_name="few_shot",
    )
    all_results.extend(few_shot_results)
    all_usage.append(few_shot_usage)

    results_path = OUTPUT_DIR / "llm_baseline_outputs.jsonl"
    with open(results_path, "w", encoding="utf-8") as file:
        for record in all_results:
            file.write(json.dumps(record) + "\n")

    usage_path = OUTPUT_DIR / "llm_baseline_cost_report.json"
    with open(usage_path, "w", encoding="utf-8") as file:
        json.dump(all_usage, file, indent=2)

    total_cost = sum(u["estimated_cost_usd"] for u in all_usage)

    print(f"\nSaved LLM outputs to {results_path}")
    print(f"Saved cost report to {usage_path}")
    print(f"Estimated total cost: ${total_cost:.4f}")
    print(
        "\nReminder: update PRICE_PER_1M_INPUT_TOKENS / PRICE_PER_1M_OUTPUT_TOKENS "
        "to the current published pricing for your chosen model before reporting "
        "the final number in your write-up."
    )


if __name__ == "__main__":
    main()