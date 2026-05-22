# -*- coding: utf-8 -*-
"""
NSMC 데이터를 data/ 폴더에 다운로드하고 과제용 파일을 생성합니다.
실행: 프로젝트 루트에서 `python download_data.py`
"""

import os
import csv
import json
import random
import re
import urllib.request
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NSMC_TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
NSMC_TEST_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

RAW_TRAIN_PATH = os.path.join(DATA_DIR, "ratings_train.txt")
RAW_TEST_PATH = os.path.join(DATA_DIR, "ratings_test.txt")
LM_TRAIN_PATH = os.path.join(DATA_DIR, "nsmc_lm_train.txt")
LM_VAL_PATH = os.path.join(DATA_DIR, "nsmc_lm_val.txt")
SENTIMENT_TRAIN_PATH = os.path.join(DATA_DIR, "nsmc_sentiment_train.jsonl")
SENTIMENT_VAL_PATH = os.path.join(DATA_DIR, "nsmc_sentiment_val.jsonl")
SENTIMENT_TEST_PATH = os.path.join(DATA_DIR, "nsmc_sentiment_test.jsonl")

DEFAULT_LM_CHAR_LIMIT = 1_500_000
DEFAULT_VAL_RATIO = 0.08
DEFAULT_SEED = 42


def _download(url: str, path: str) -> None:
    if os.path.isfile(path):
        print(f"이미 존재합니다: {path}")
        return
    print(f"다운로드 중: {url}")
    urllib.request.urlretrieve(url, path)
    print(f"저장됨: {path}")


def _clean_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _read_nsmc_tsv(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = _clean_text(row.get("document"))
            label = row.get("label")
            if not text or label not in {"0", "1"}:
                continue
            rows.append({"text": text, "label": int(label)})
    return rows


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_lm_files(rows: list[dict], lm_char_limit: int, val_ratio: float) -> None:
    texts: list[str] = []
    total_chars = 0
    for row in rows:
        text = row["text"]
        if total_chars >= lm_char_limit:
            break
        texts.append(text)
        total_chars += len(text) + 1

    split_idx = max(1, int(len(texts) * (1.0 - val_ratio)))
    train_text = "\n".join(texts[:split_idx])
    val_text = "\n".join(texts[split_idx:])

    Path(LM_TRAIN_PATH).write_text(train_text, encoding="utf-8")
    Path(LM_VAL_PATH).write_text(val_text, encoding="utf-8")
    print(f"사전 학습 train 텍스트: {LM_TRAIN_PATH} ({len(train_text):,}자)")
    print(f"사전 학습 val 텍스트: {LM_VAL_PATH} ({len(val_text):,}자)")


def prepare_data(
    lm_char_limit: int = DEFAULT_LM_CHAR_LIMIT,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """
    NSMC 원본 TSV를 과제용 LM 텍스트와 감성 분류 JSONL로 변환합니다.

    Returns:
        생성된 주요 파일 경로 딕셔너리
    """
    train_rows = _read_nsmc_tsv(RAW_TRAIN_PATH)
    test_rows = _read_nsmc_tsv(RAW_TEST_PATH)

    rng = random.Random(seed)
    rng.shuffle(train_rows)

    val_size = max(1, int(len(train_rows) * val_ratio))
    val_rows = train_rows[:val_size]
    train_rows_for_cls = train_rows[val_size:]

    _write_lm_files(train_rows, lm_char_limit=lm_char_limit, val_ratio=val_ratio)
    _write_jsonl(SENTIMENT_TRAIN_PATH, train_rows_for_cls)
    _write_jsonl(SENTIMENT_VAL_PATH, val_rows)
    _write_jsonl(SENTIMENT_TEST_PATH, test_rows)

    print(f"감성 분류 train: {SENTIMENT_TRAIN_PATH} ({len(train_rows_for_cls):,}개)")
    print(f"감성 분류 val: {SENTIMENT_VAL_PATH} ({len(val_rows):,}개)")
    print(f"감성 분류 test: {SENTIMENT_TEST_PATH} ({len(test_rows):,}개)")

    return {
        "lm_train": LM_TRAIN_PATH,
        "lm_val": LM_VAL_PATH,
        "sentiment_train": SENTIMENT_TRAIN_PATH,
        "sentiment_val": SENTIMENT_VAL_PATH,
        "sentiment_test": SENTIMENT_TEST_PATH,
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    _download(NSMC_TRAIN_URL, RAW_TRAIN_PATH)
    _download(NSMC_TEST_URL, RAW_TEST_PATH)
    return prepare_data()


if __name__ == "__main__":
    main()
