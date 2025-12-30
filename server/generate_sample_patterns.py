#!/usr/bin/env python3
"""
Generate random essay sample patterns for experiments.

Creates 50 independent patterns of 10 randomly sampled essays per prompt.
Uses fixed seed for reproducibility.

Usage:
    python generate_sample_patterns.py [--seed SEED] [--n-patterns N] [--n-samples N]
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
    n_samples: int = 10,
    seed: int = 42,
) -> dict:
    """
    Generate random sample patterns for each prompt.

    Args:
        df: ASAP dataframe
        n_patterns: Number of independent patterns to generate
        n_samples: Number of essays per pattern
        seed: Random seed for reproducibility

    Returns:
        Dictionary with patterns for each prompt
    """
    patterns = {}

    for prompt_id in range(1, 9):
        prompt_df = df[df['essay_set'] == prompt_id]
        essay_ids = prompt_df['essay_id'].tolist()

        if len(essay_ids) < n_samples:
            raise ValueError(
                f"Prompt {prompt_id} has only {len(essay_ids)} essays, "
                f"but {n_samples} samples requested"
            )

        # Generate patterns with deterministic seeds
        prompt_patterns = []
        for pattern_idx in range(n_patterns):
            # Each pattern uses a unique seed derived from base seed
            pattern_seed = seed + prompt_id * 1000 + pattern_idx
            rng = random.Random(pattern_seed)

            # Sample without replacement
            sampled_ids = rng.sample(essay_ids, n_samples)
            prompt_patterns.append(sampled_ids)

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
        "--n-samples",
        type=int,
        default=10,
        help="Number of essays per pattern (default: 10)",
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

    # Generate patterns
    print(f"Generating {args.n_patterns} patterns x {args.n_samples} samples per prompt")
    print(f"Seed: {args.seed}")

    patterns = generate_patterns(
        df,
        n_patterns=args.n_patterns,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    # Create output structure
    output = {
        "metadata": {
            "seed": args.seed,
            "n_patterns": args.n_patterns,
            "n_samples_per_pattern": args.n_samples,
            "generated_at": datetime.now().isoformat(),
            "description": (
                f"Random essay sample patterns for experiments. "
                f"Each pattern contains {args.n_samples} randomly sampled essay IDs. "
                f"Patterns are independent and reproducible with the given seed."
            ),
        },
        "patterns": patterns,
    }

    # Verify counts
    print("\nVerification:")
    for prompt_name, prompt_patterns in patterns.items():
        print(f"  {prompt_name}: {len(prompt_patterns)} patterns, "
              f"{len(prompt_patterns[0])} samples each")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show example
    print("\nExample (prompt1, pattern 0):")
    print(f"  Essay IDs: {patterns['prompt1'][0]}")


if __name__ == "__main__":
    main()
