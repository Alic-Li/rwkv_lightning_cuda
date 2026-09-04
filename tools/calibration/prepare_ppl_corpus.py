#!/usr/bin/env python3
"""Prepare a deterministic Chinese corpus for RWKV perplexity calibration."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path


DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.zh"
DATASET_SPLIT = "train"
SHUFFLE_SEED = 42
DEFAULT_NUM_ARTICLES = 800
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
NUM_SHARDS = 6


def iter_huggingface_articles():
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "the 'datasets' package is required for HuggingFace extraction; "
            "install it with 'pip install datasets pyarrow'"
        ) from error
    data_files = [
        f"{os.environ['HF_ENDPOINT']}/datasets/{DATASET_NAME}/resolve/main/"
        f"{DATASET_CONFIG}/train-{index:05d}-of-{NUM_SHARDS:05d}.parquet"
        for index in range(NUM_SHARDS)
    ]
    dataset = load_dataset("parquet", data_files=data_files, split=DATASET_SPLIT, streaming=True).shuffle(
        seed=SHUFFLE_SEED
    )
    for row in dataset:
        text = row.get("text")
        if not isinstance(text, str):
            raise RuntimeError("HuggingFace row has no string 'text' field")
        yield text


def iter_fallback_articles(path):
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid fallback JSON on line {line_number}") from error
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise RuntimeError(f"fallback row {line_number} has no string 'text' field")
            yield text


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="UTF-8 corpus output path")
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_NUM_ARTICLES,
        help=f"number of shuffled articles (default: {DEFAULT_NUM_ARTICLES})",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help=f"HuggingFace endpoint (default: {DEFAULT_HF_ENDPOINT})",
    )
    parser.add_argument(
        "--fallback-jsonl",
        help="local nlp_chinese_corpus wiki2019zh JSONL file if HuggingFace is unavailable",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive")
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    try:
        articles = iter_huggingface_articles()
        source = f"{DATASET_NAME}/{DATASET_CONFIG}"
        selected = list(itertools.islice(articles, args.n))
        if len(selected) != args.n:
            raise RuntimeError(f"dataset ended after {len(selected)} articles; need {args.n}")
    except Exception as error:
        if not args.fallback_jsonl:
            raise RuntimeError(
                f"HuggingFace extraction failed: {error}; provide --fallback-jsonl "
                "from the nlp_chinese_corpus wiki2019zh 2019-02-07 snapshot"
            ) from error
        source = "nlp_chinese_corpus/wiki2019zh (2019-02-07 snapshot)"
        selected = list(itertools.islice(iter_fallback_articles(args.fallback_jsonl), args.n))
        if len(selected) != args.n:
            raise RuntimeError(f"fallback ended after {len(selected)} articles; need {args.n}")

    output_path = Path(args.output)
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        output.write("\n\n".join(selected))
    print(f"corpus_source={source}")
    print(f"dataset_split={DATASET_SPLIT}")
    print(f"shuffle_seed={SHUFFLE_SEED}")
    print(f"articles={len(selected)}")
    print(f"utf8_bytes={output_path.stat().st_size}")
    print(f"sha256={sha256(output_path)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"prepare_ppl_corpus failed: {error}") from error
