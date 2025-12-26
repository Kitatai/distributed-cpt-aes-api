#!/usr/bin/env python3
"""
Initialize task queue.
Usage: uv run python init_tasks.py [--force]
"""

import sys
import argparse
import requests


def main():
    parser = argparse.ArgumentParser(description="Initialize tasks")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="Server URL")
    parser.add_argument("--force", action="store_true", help="Force reinitialize")
    args = parser.parse_args()

    url = f"{args.server}/tasks/init"
    params = {"force": "true"} if args.force else {}

    try:
        resp = requests.post(url, params=params)
        resp.raise_for_status()
        result = resp.json()
        print(result.get("message", "Done"))
    except requests.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
