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
        "article": """Go online to Pixar news sources, such as http://www.pixarpost.com/ and https://pixarplanet.com. These websites typically post about different charitable auction opportunities that happen from time to time. The purpose of these is to raise money for different organizations, such as the LA Food Bank or the Singing Stones School. Follow the directions on the website in order to place your bid, as you may get a tour if you have the highest one. While this is a good way to get a tour, it can be extremely expensive. Another way that Pixar uses charity fundraising to offer tours is by temporarily giving the public an opportunity to buy touring tickets, usually just before a film premier. Check out https://www.charitybuzz.com/ to see if any charity ticket opportunities have been posted. This is another pricey option. For example, tickets for a charity event in 2011 that included a black tie dinner, a screening, and an exclusive tour were $1,000 USD each. While it may be harder to get a tour by winning a contest, it's a great option to pursue if you don't have any personal connections and you also don't want to spend a lot of money. Check out https://pixarplanet.com/blog/category/competition/ regularly to see what contests are posted. Contest entry requirements range in difficulty from answering a prompt with a short essay response to creating a trailer for a film. The prizes also vary, from special edition Blu-Ray disc sets to tours of the studio.""",
        "summary": "Participate in charitable auctions that are posted online. Purchase tour tickets for special charity events. Participate in contests that are posted online.",
    },
        {
        "article": """These shorts should have an elastic band. The length can range from your mid-thigh down to your knees. These are often branded as athletic shorts. Basketball shorts, running shorts, and soccer shorts are good options. For women playing sports, it is essential that you wear a sports bra underneath your shirt instead of a normal bra. These tight, polyester bras will allow you to run around without interference or injury. Short-sleeved shirts or sleeveless tank tops are good for volleyball because they allow a full range of motion while keeping you cool. Compression shirts, sweat-proof athletic tops, and mesh jerseys are all good options as well. Choose a cotton or polyester blend. Make sure that you can fully move your arms. Find shoes with good traction. Check the soles for deep rubber grooves. You should be able to wiggle your toes comfortably in these shoes. While you can buy specialized volleyball shoes, you don't have to unless you are a competitive player. Other types of court shoes, such as tennis or basketball shoes, are good substitutes. If you have long hair, you will want to keep it out of your face. You can put it in a ponytail or a French braid. A tight bun also works. Make sure that it is tight enough so it won't untangle or get in the way.  If you have shorter hair, you may opt for a headband to keep hair out of your eyes. Choose one that has elastic bands to keep it on your head even during rough play. Or, choose messy bun and wear a headband. A lot of people wear little skinny headbands or pre-wrap. Another cute things is tie big ribbon bows in your ponytail.""",
        "summary": "Choose loose fitting shorts. Put on a sports bra. Wear a t-shirt or tank top. Find comfortable athletic shoes. Put up your hair up.",
    },
    {
        "article": """When you're taking bust measurements, a bra will help lift your breasts, but make sure there's no extra padding to alter the size. If you don't have a non-padded bra, it's fine to take your measurement without a bra. Use a measuring tape to measure just beneath your bust. Keep the measuring tape level and pull it taut, but not tight enough to dig into your skin. If you get a fraction, round the measurement up to the next whole number. Add 4 to the measurement if it's an even number or 5 if it's an odd number. The resulting number is your band size. For instance,  if your under bust measurement is 31 inches, your band size will be 36. The fullest part of your breasts is usually around the nipple line. Make sure the tape measure is even and pull it so it is just tight enough to rest against you all the way around. Round the bust measurement up to the nearest whole number. For instance, if your bust measures 33.5 inches, round up to 34. The difference in your bust measurement and your under cup measurement will give you your cup size. Use the original under bust measurement, not your calculated band size. The difference between cup sizes is about an inch. If the difference between measurements is a 1, your cup size is A, if 2, it would be B and so on. In the case of the previous examples, you would subtract 31 from 34 to get 3, which would make your measurement a 36C.""",
        "summary": "Put on a non-padded bra if you have one. Measure under your bust and calculate your band size. Measure around your bust at the fullest point. Subtract your under bust measurement from your bust size.",
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
    parser.add_argument("--n_examples", type=int, default=40, help="Limit test set size (default: use all)")
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