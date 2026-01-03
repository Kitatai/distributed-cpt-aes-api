#!/usr/bin/env python3
"""
Generate random essay sample patterns for experiments.

Creates 50 independent patterns with test/dev/fewshot splits per prompt:
- test_ids: 10% of essays for evaluation (shuffled)
- dev_ids: 10 essays for epoch selection (shuffled)
- fewshot_ids: 5 essays for few-shot examples (shuffled)

Few-shot usage: 0-shot=[], 1-shot=fewshot_ids[:1], 3-shot=fewshot_ids[:3], 5-shot=fewshot_ids[:5]

Usage:
    python generate_sample_patterns.py [--seed SEED] [--n-patterns N]
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd


def load_asap_data(data_path: Path) -> pd.DataFrame:
    """Load ASAP dataset."""
    return pd.read_csv(data_path, sep='\t', encoding='latin-1')


def generate_patterns(
    df: pd.DataFrame,
    n_patterns: int = 50,
    n_dev: int = 10,
    n_fewshot: int = 5,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Generate random sample patterns for each prompt.

    For each pattern:
    1. Randomly select 10% of essays as test data
    2. From remaining 90%, select n_dev essays for epoch selection
    3. From remaining, select n_fewshot essays for few-shot examples
    All selections are shuffled.

    Args:
        df: ASAP dataframe
        n_patterns: Number of independent patterns to generate
        n_dev: Number of essays for epoch selection (default: 10)
        n_fewshot: Number of essays for few-shot examples (default: 5)
        test_ratio: Ratio of essays to use for test (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with patterns for each prompt
    """
    patterns = {}

    for prompt_id in range(1, 9):
        prompt_df = df[df['essay_set'] == prompt_id]
        essay_ids = prompt_df['essay_id'].tolist()
        n_total = len(essay_ids)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_test

        if n_train < n_dev + n_fewshot:
            raise ValueError(
                f"Prompt {prompt_id}: Not enough training essays ({n_train}) "
                f"for dev ({n_dev}) + fewshot ({n_fewshot})"
            )

        # Generate patterns with deterministic seeds
        prompt_patterns = []
        for pattern_idx in range(n_patterns):
            # Each pattern uses a unique seed derived from base seed
            pattern_seed = seed + prompt_id * 1000 + pattern_idx
            rng = random.Random(pattern_seed)

            # Step 1: Randomly select test essays (10%)
            shuffled_ids = essay_ids.copy()
            rng.shuffle(shuffled_ids)
            test_ids = shuffled_ids[:n_test]
            remaining_ids = shuffled_ids[n_test:]

            # Step 2: From remaining, select dev essays
            rng.shuffle(remaining_ids)
            dev_ids = remaining_ids[:n_dev]
            remaining_ids = remaining_ids[n_dev:]

            # Step 3: From remaining, select fewshot essays
            rng.shuffle(remaining_ids)
            fewshot_ids = remaining_ids[:n_fewshot]

            # Shuffle all selected lists
            rng.shuffle(test_ids)
            rng.shuffle(dev_ids)
            rng.shuffle(fewshot_ids)

            prompt_patterns.append({
                "test_ids": test_ids,
                "dev_ids": dev_ids,
                "fewshot_ids": fewshot_ids,
            })

        patterns[f"prompt{prompt_id}"] = prompt_patterns

    return patterns


def main():
    parser = argparse.ArgumentParser(description="Generate essay sample patterns")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-patterns",
        type=int,
        default=50,
        help="Number of patterns to generate (default: 50)",
    )
    parser.add_argument(
        "--n-dev",
        type=int,
        default=10,
        help="Number of essays for epoch selection (default: 10)",
    )
    parser.add_argument(
        "--n-fewshot",
        type=int,
        default=5,
        help="Number of essays for few-shot examples (default: 5)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Ratio of essays to use for test (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/sample_patterns.json)",
    )

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "asap" / "training_set_rel3.tsv"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = script_dir / "data" / "sample_patterns.json"

    # Load data
    print(f"Loading ASAP data from {data_path}")
    df = load_asap_data(data_path)
    print(f"Loaded {len(df)} essays")

    # Show per-prompt stats
    print("\nPer-prompt statistics:")
    for prompt_id in range(1, 9):
        n_total = len(df[df['essay_set'] == prompt_id])
        n_test = int(n_total * args.test_ratio)
        n_train = n_total - n_test
        print(f"  Prompt {prompt_id}: {n_total} total -> {n_test} test + {n_train} train")

    # Generate patterns
    print(f"\nGenerating {args.n_patterns} patterns per prompt")
    print(f"  test_ratio: {args.test_ratio} (10%)")
    print(f"  n_dev: {args.n_dev} (epoch selection)")
    print(f"  n_fewshot: {args.n_fewshot} (few-shot examples)")
    print(f"  seed: {args.seed}")

    patterns = generate_patterns(
        df,
        n_patterns=args.n_patterns,
        n_dev=args.n_dev,
        n_fewshot=args.n_fewshot,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Create output structure
    output = {
        "metadata": {
            "seed": args.seed,
            "n_patterns": args.n_patterns,
            "test_ratio": args.test_ratio,
            "n_dev": args.n_dev,
            "n_fewshot": args.n_fewshot,
            "generated_at": datetime.now().isoformat(),
            "description": (
                f"Random essay sample patterns for experiments. "
                f"Each pattern contains: test_ids ({args.test_ratio*100:.0f}% for evaluation), "
                f"dev_ids ({args.n_dev} for epoch selection), "
                f"fewshot_ids ({args.n_fewshot} for few-shot examples). "
                f"All lists are shuffled. "
                f"Few-shot usage: 0-shot=[], 1-shot=fewshot_ids[:1], 3-shot=fewshot_ids[:3], 5-shot=fewshot_ids[:5]. "
                f"Patterns are independent and reproducible with the given seed."
            ),
        },
        "patterns": patterns,
    }

    # Verify counts
    print("\nVerification:")
    for prompt_name, prompt_patterns in patterns.items():
        p0 = prompt_patterns[0]
        print(f"  {prompt_name}: {len(prompt_patterns)} patterns, "
              f"test={len(p0['test_ids'])}, dev={len(p0['dev_ids'])}, fewshot={len(p0['fewshot_ids'])}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show example
    print("\nExample (prompt1, pattern 0):")
    p0 = patterns['prompt1'][0]
    print(f"  test_ids ({len(p0['test_ids'])}): {p0['test_ids'][:5]}... (first 5)")
    print(f"  dev_ids ({len(p0['dev_ids'])}): {p0['dev_ids']}")
    print(f"  fewshot_ids ({len(p0['fewshot_ids'])}): {p0['fewshot_ids']}")


if __name__ == "__main__":
    main()
