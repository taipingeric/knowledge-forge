#!/usr/bin/env python3
"""Run the search_knowledge LangChain tool against a knowledge Bundle from the CLI.

Example:
    uv run scripts/search_knowledge.py knowledge/mysql deadlock mvcc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_forge.tools import build_search_knowledge_tool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Path to a knowledge Bundle directory")
    parser.add_argument("keywords", nargs="+", help="Keywords to search for")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_knowledge = build_search_knowledge_tool(args.bundle)
    raw = search_knowledge.invoke({"keywords": args.keywords})
    print(json.dumps(json.loads(raw), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
