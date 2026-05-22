# -*- coding: utf-8 -*-
"""
BPE 토크나이저 단위 테스트.
실행: 프로젝트 루트에서 `pytest tests/test_bpe.py -v`
특정 클래스만: `pytest tests/test_bpe.py -v -k "TestBPETokenizer"`
"""

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpe import (
    BPETokenizer,
    SPECIAL_IDS,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    BYTE_OFFSET,
    NUM_BYTES,
)

# =============================================================================
# 특수 토큰 ID
# =============================================================================


class TestSpecialTokens:
    """특수 토큰 ID가 0,1,2,3으로 고정인지 확인."""

    def test_special_ids_fixed(self):
        """pad/unk/bos/eos ID가 문서와 코드에서 약속한 고정값인지 확인한다."""
        assert SPECIAL_IDS[PAD_TOKEN] == 0
        assert SPECIAL_IDS[UNK_TOKEN] == 1
        assert SPECIAL_IDS[BOS_TOKEN] == 2
        assert SPECIAL_IDS[EOS_TOKEN] == 3


# =============================================================================
# BPETokenizer (load/save/encode/decode)
# =============================================================================


class TestBPETokenizer:
    """BPETokenizer: _init_special_tokens, load, save, encode, decode 구현 후 실행."""

    def test_init_special_tokens(self):
        """_init_special_tokens()가 특수 토큰 4개와 UTF-8 byte 토큰 256개를 준비하는지 확인한다."""
        tok = BPETokenizer(vocab_size=300)
        try:
            tok._init_special_tokens()
        except NotImplementedError:
            pytest.fail("_init_special_tokens 미구현")
        assert len(tok.id_to_token) >= BYTE_OFFSET + NUM_BYTES
        assert tok.token_to_id[PAD_TOKEN] == 0
        assert tok.token_to_id[EOS_TOKEN] == 3
        assert BYTE_OFFSET in tok.id_to_token
        assert BYTE_OFFSET + 255 in tok.id_to_token

    def test_save_load_restores_vocab(self):
        """save 후 load하면 byte/merge 토큰과 merge 규칙이 그대로 복원되는지 확인한다."""
        tok = BPETokenizer(vocab_size=300)
        try:
            tok._init_special_tokens()
            tok.merges = [(BYTE_OFFSET + 65, BYTE_OFFSET + 66)]
            tok.id_to_token[BYTE_OFFSET + NUM_BYTES] = (BYTE_OFFSET + 65, BYTE_OFFSET + 66)
            tok.token_to_id[(BYTE_OFFSET + 65, BYTE_OFFSET + 66)] = BYTE_OFFSET + NUM_BYTES
        except NotImplementedError:
            pytest.fail("_init_special_tokens 미구현")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tok.save(path)
            tok2 = BPETokenizer(vocab_size=300)
            tok2.load(path)
            assert tok2.merges == tok.merges
            assert tok2.id_to_token.get(0) == PAD_TOKEN
        except NotImplementedError:
            pytest.fail("save/load 미구현")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_encode_decode_restores_original_text(self):
        """영어/한국어 혼합 텍스트가 encode -> decode 뒤에도 깨지지 않는지 확인한다."""
        tok = BPETokenizer(vocab_size=300)
        try:
            tok._init_special_tokens()
            sample = "이 영화는 정말 좋았다! English 123"
            ids = tok.encode(sample, add_bos_eos=True)
            assert ids[0] == SPECIAL_IDS[BOS_TOKEN]
            assert ids[-1] == SPECIAL_IDS[EOS_TOKEN]
            text = tok.decode(ids, skip_special=True)
            assert text == sample
        except NotImplementedError:
            pytest.fail("encode/decode 미구현")

    def test_get_special_ids(self):
        """get_pad_id/get_unk_id/get_bos_id/get_eos_id가 고정 ID를 반환하는지 확인한다."""
        tok = BPETokenizer(vocab_size=10)
        assert tok.get_pad_id() == 0
        assert tok.get_unk_id() == 1
        assert tok.get_bos_id() == 2
        assert tok.get_eos_id() == 3


# =============================================================================
# train (통합)
# =============================================================================


class TestBPETrain:
    """train() 구현 후 소량 코퍼스로 vocab 크기 증가 확인."""

    def test_train_increases_vocab(self):
        """train()이 소량 한국어 코퍼스에서 vocab을 초기 byte 토큰 이상으로 구성하는지 확인한다."""
        tok = BPETokenizer(vocab_size=300)
        try:
            tok.train("이 영화는 정말 좋았다. 이 영화는 다시 보고 싶다.")
            assert len(tok.id_to_token) >= BYTE_OFFSET + NUM_BYTES
            assert len(tok.id_to_token) <= 300
        except NotImplementedError:
            pytest.fail("train 미구현")
