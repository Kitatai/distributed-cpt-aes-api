#!/usr/bin/env python3
"""
Show task status.
Usage: uv run python status.py [--matrix]
"""

import sys
import argparse
import requests


def print_matrix(data):
    """Print task matrix."""
    models = ["llama", "qwen", "mistral"]

    # Header
    print(f"{'Prompt':<10}", end="")
    for m in models:
        print(f"{m:<15}", end="")
    print()
    print("-" * 55)

    # Rows
    for prompt in [f"prompt{i}" for i in range(1, 9)]:
        row = data.get(prompt, {})
        print(f"{prompt:<10}", end="")
        for m in models:
            cell = row.get(m, {})
            status = cell.get("status", "-")
            epoch = cell.get("epoch", 0)
            worker = cell.get("worker", "")
            if worker:
                worker = worker.split("_")[0][:6]
            print(f"{status}({epoch:02d}){worker:<6} ", end="")
        print()

    print()
    print("Status: P=Pending, R=Running, C=Completed, F=Failed")


def print_status(data):
    """Print overall status."""
    print(f"Total: {data['total_tasks']}")
    print(f"  Pending:   {data['pending']}")
    print(f"  Running:   {data['running']}")
    print(f"  Completed: {data['completed']}")
    print(f"  Failed:    {data['failed']}")


def main():
    parser = argparse.ArgumentParser(description="Show task status")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--matrix", action="store_true", help="Show matrix view")
    args = parser.parse_args()

    try:
        if args.matrix:
            resp = requests.get(f"{args.server}/tasks/matrix")
            resp.raise_for_status()
            print_matrix(resp.json())
        else:
            resp = requests.get(f"{args.server}/tasks")
            resp.raise_for_status()
            print_status(resp.json())

    except requests.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
