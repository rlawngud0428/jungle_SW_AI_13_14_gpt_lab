# -*- coding: utf-8 -*-
"""
데이터셋·데이터 로더·입력 임베딩 단위 테스트.
실행: `pytest tests/test_dataset.py -v`
"""

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# =============================================================================
# GPTDataset
# =============================================================================


class TestGPTDataset:
    """GPTDataset __len__, __getitem__ 구현 후 실행."""

    def test_dataset_length(self):
        """토큰 개수, context_length, stride로 만들 수 있는 학습 샘플 수가 맞는지 확인한다."""
        from dataset import GPTDataset

        token_ids = list(range(100))
        context_length = 10
        stride = 5
        try:
            ds = GPTDataset(token_ids, context_length, stride=stride)
        except NotImplementedError:
            pytest.fail("GPTDataset._length 미구현")
        if len(ds) == 0:
            pytest.fail("GPTDataset._length 미구현")
        # input 길이 10과 target 길이 10을 만들려면 마지막 다음 토큰 1개가 더 필요하다.
        expected = (100 - 10 - 1) // 5 + 1
        assert len(ds) == expected

    def test_dataset_getitem_shape(self):
        """GPTDataset[0]이 input/target LongTensor를 만들고 target이 한 칸 shift되는지 확인한다."""
        from dataset import GPTDataset

        token_ids = list(range(50))
        context_length = 8
        try:
            ds = GPTDataset(token_ids, context_length, stride=8)
            inp, tgt = ds[0]
        except NotImplementedError:
            pytest.fail("GPTDataset.__getitem__ 미구현")
        assert inp.shape == (context_length,)
        assert tgt.shape == (context_length,)
        assert inp.dtype == torch.long
        # target은 input을 한 칸 시프트한 다음 토큰
        torch.testing.assert_close(tgt[:-1], inp[1:])


# =============================================================================
# create_dataloader
# =============================================================================


class TestCreateDataloader:
    """create_dataloader 구현 후 실행."""

    def test_dataloader_batch_shape(self):
        """create_dataloader()가 학습 루프에 넣을 수 있는 배치 shape를 반환하는지 확인한다."""
        from dataset import create_dataloader

        token_ids = list(range(200))
        context_length = 16
        batch_size = 4
        try:
            loader = create_dataloader(
                token_ids,
                context_length,
                batch_size=batch_size,
                shuffle=False,
            )
            inp, tgt = next(iter(loader))
        except NotImplementedError:
            pytest.fail("create_dataloader 미구현")
        assert inp.shape == (batch_size, context_length)
        assert tgt.shape == (batch_size, context_length)


# =============================================================================
# InputEmbedding
# =============================================================================


class TestInputEmbedding:
    """InputEmbedding forward 구현 후 실행."""

    def test_input_embedding_shape(self):
        """InputEmbedding이 토큰 ID 배치를 (batch, seq_len, emb_dim) 텐서로 바꾸는지 확인한다."""
        from embeddings import InputEmbedding

        batch_size, seq_len = 2, 8
        vocab_size, emb_dim, context_length = 1000, 64, 128
        try:
            emb = InputEmbedding(vocab_size, emb_dim, context_length, drop_rate=0.0)
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            out = emb(x)
        except (NotImplementedError, TypeError):
            pytest.fail("InputEmbedding 미구현")
        assert out.shape == (batch_size, seq_len, emb_dim)
