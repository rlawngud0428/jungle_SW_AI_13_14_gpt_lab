# -*- coding: utf-8 -*-
"""
감성 분류 미세 조정 단위 테스트.
실행: `pytest tests/test_finetune.py -v`
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


class DummyTokenizer:
    """테스트에서 실제 BPE 학습 없이 Dataset 동작만 확인하기 위한 간단한 tokenizer."""

    def get_pad_id(self):
        """ReviewSentimentDataset이 padding ID를 가져갈 수 있게 0을 반환한다."""
        return 0

    def encode(self, text, add_bos_eos=False):
        """문자열을 작고 안전한 토큰 ID 범위로 바꿔 테스트 입력을 만든다."""
        ids = [(ord(ch) % 50) + 4 for ch in text]
        if add_bos_eos:
            return [2] + ids + [3]
        return ids


GPT_CONFIG_TINY = {
    "vocab_size": 128,
    "context_length": 16,
    "emb_dim": 16,
    "n_heads": 4,
    "n_layers": 1,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


class TestMakeSentimentDataset:
    """make_sentiment_dataset 구현 후 실행."""

    def test_make_sentiment_dataset_splits_rows(self):
        """NSMC 형식 TSV에서 빈 리뷰를 제거하고 train/val/test를 나누는지 확인한다."""
        from finetune import make_sentiment_dataset

        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "ratings_train.txt"
            test_path = Path(tmp) / "ratings_test.txt"
            train_path.write_text(
                "id\tdocument\tlabel\n"
                "1\t정말 좋았다\t1\n"
                "2\t별로였다\t0\n"
                "3\t다시 보고 싶다\t1\n"
                "4\t\t0\n",
                encoding="utf-8",
            )
            test_path.write_text(
                "id\tdocument\tlabel\n"
                "5\t괜찮았다\t1\n",
                encoding="utf-8",
            )
            try:
                train_data, val_data, test_data = make_sentiment_dataset(
                    train_path, test_path, val_ratio=0.34, seed=42
                )
            except NotImplementedError:
                pytest.fail("make_sentiment_dataset 미구현")
        assert len(train_data) + len(val_data) == 3
        assert len(test_data) == 1
        assert set(train_data[0].keys()) == {"text", "label"}
        assert train_data[0]["label"] in {0, 1}


class TestReviewSentimentDataset:
    """ReviewSentimentDataset 구현 후 실행."""

    def test_review_sentiment_dataset_getitem(self):
        """ReviewSentimentDataset이 리뷰를 token ID로 바꾸고 max_length까지 padding하는지 확인한다."""
        from finetune import ReviewSentimentDataset

        data = [{"text": "재미있다", "label": 1}]
        ds = ReviewSentimentDataset(data, DummyTokenizer(), max_length=8)
        try:
            input_ids, label = ds[0]
        except NotImplementedError:
            pytest.fail("ReviewSentimentDataset.__getitem__ 미구현")
        assert input_ids.shape == (8,)
        assert input_ids.dtype == torch.long
        assert label == 1


class TestGPTForSequenceClassification:
    """GPTForSequenceClassification 구현 후 실행."""

    def test_sequence_classification_shape(self):
        """GPT backbone 위 분류 head가 샘플마다 num_labels 크기의 logits를 내는지 확인한다."""
        from model import GPTModel
        from finetune import GPTForSequenceClassification

        try:
            backbone = GPTModel(GPT_CONFIG_TINY)
            model = GPTForSequenceClassification(backbone, num_labels=2)
            input_ids = torch.randint(0, GPT_CONFIG_TINY["vocab_size"], (2, 8))
            logits = model(input_ids)
        except (NotImplementedError, TypeError):
            pytest.fail("GPTForSequenceClassification 미구현")
        assert logits.shape == (2, 2)


class TestSentimentTrainEval:
    """훈련/평가 함수가 호출 가능한지 확인."""

    def test_train_eval_functions_exist(self):
        """train_epoch_sentiment/evaluate_sentiment 함수가 import 가능하고 callable인지 확인한다."""
        from finetune import train_epoch_sentiment, evaluate_sentiment

        assert callable(train_epoch_sentiment)
        assert callable(evaluate_sentiment)
