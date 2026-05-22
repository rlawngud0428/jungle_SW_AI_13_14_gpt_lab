# -*- coding: utf-8 -*-
"""
GPT 모델 단위 테스트 (LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel, generate_text_simple).
실행: `pytest tests/test_model.py -v`
"""

import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 테스트용 작은 설정
GPT_CONFIG_SMALL = {
    "vocab_size": 1000,
    "context_length": 64,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


# =============================================================================
# LayerNorm
# =============================================================================


class TestLayerNorm:
    """LayerNorm forward 구현 후 실행."""

    def test_layernorm_shape(self):
        """LayerNorm이 마지막 차원을 정규화하되 전체 텐서 shape는 유지하는지 확인한다."""
        from model import LayerNorm

        try:
            ln = LayerNorm(32)
            x = torch.randn(4, 10, 32)
            out = ln(x)
        except NotImplementedError:
            pytest.fail("LayerNorm 미구현")
        assert out.shape == x.shape


# =============================================================================
# GELU
# =============================================================================


class TestGELU:
    """GELU forward 구현 후 실행."""

    def test_gelu_shape(self):
        """GELU 활성화 함수가 입력과 같은 shape의 출력을 반환하는지 확인한다."""
        from model import GELU

        try:
            gelu = GELU()
            x = torch.randn(2, 8)
            out = gelu(x)
        except NotImplementedError:
            pytest.fail("GELU 미구현")
        assert out.shape == x.shape


# =============================================================================
# FeedForward
# =============================================================================


class TestFeedForward:
    """FeedForward forward 구현 후 실행."""

    def test_feedforward_shape(self):
        """FeedForward 블록이 hidden 차원을 확장했다가 다시 d_model shape로 되돌리는지 확인한다."""
        from model import FeedForward

        d_model = 64
        try:
            ffn = FeedForward(d_model, dropout=0.0)
            x = torch.randn(2, 10, d_model)
            out = ffn(x)
        except NotImplementedError:
            pytest.fail("FeedForward 미구현")
        assert out.shape == (2, 10, d_model)


# =============================================================================
# TransformerBlock
# =============================================================================


class TestTransformerBlock:
    """TransformerBlock forward 구현 후 실행."""

    def test_transformer_block_shape(self):
        """TransformerBlock이 attention과 FFN을 거친 뒤 입력과 같은 shape를 유지하는지 확인한다."""
        from model import TransformerBlock

        d_model, n_heads = 64, 4
        try:
            block = TransformerBlock(d_model, n_heads, drop_rate=0.0)
            x = torch.randn(2, 8, d_model)
            out = block(x, causal_mask=True)
        except NotImplementedError:
            pytest.fail("TransformerBlock 미구현")
        assert out.shape == (2, 8, d_model)


# =============================================================================
# GPTModel
# =============================================================================


class TestGPTModel:
    """GPTModel forward 구현 후 실행."""

    def test_gpt_forward_shape(self):
        """targets 없이 호출한 GPTModel이 다음 토큰 예측용 logits를 반환하는지 확인한다."""
        from model import GPTModel

        config = GPT_CONFIG_SMALL.copy()
        try:
            model = GPTModel(config)
            batch_size, seq_len = 2, 16
            idx = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
            logits = model(idx, targets=None)
        except (NotImplementedError, TypeError):
            pytest.fail("GPTModel 미구현")
        if isinstance(logits, tuple):
            logits = logits[1]
        assert logits.shape == (batch_size, seq_len, config["vocab_size"])

    def test_gpt_forward_with_targets_returns_loss(self):
        """targets를 함께 넘기면 GPTModel이 scalar loss와 logits를 함께 반환하는지 확인한다."""
        from model import GPTModel

        config = GPT_CONFIG_SMALL.copy()
        try:
            model = GPTModel(config)
            batch_size, seq_len = 2, 16
            idx = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
            targets = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
            loss, logits = model(idx, targets=targets)
        except (NotImplementedError, TypeError):
            pytest.fail("GPTModel(targets=...) 미구현")
        assert loss.dim() == 0
        assert loss.item() >= 0


# =============================================================================
# generate_text_simple
# =============================================================================


class TestGenerateTextSimple:
    """generate_text_simple 구현 후 실행."""

    def test_generate_text_simple_shape(self):
        """generate_text_simple()이 지정한 개수만큼 새 토큰을 뒤에 붙이는지 확인한다."""
        from model import GPTModel, generate_text_simple

        config = GPT_CONFIG_SMALL.copy()
        try:
            model = GPTModel(config)
            batch_size, start_len = 1, 4
            idx = torch.randint(0, config["vocab_size"], (batch_size, start_len))
            out = generate_text_simple(model, idx, max_new_tokens=6, context_size=config["context_length"])
        except NotImplementedError:
            pytest.fail("generate_text_simple 미구현")
        assert out.shape == (batch_size, start_len + 6)
