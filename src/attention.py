# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 각 head가 담당할 feature 차원입니다. 예: d_model=64, n_heads=4이면 head_dim=16입니다.
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model) # 여러 head의 결과를 다시 d_model 차원으로 결합하는 출력 projection입니다.
        self.dropout = nn.Dropout(drop_rate)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        # x는 (batch_size, seq_len, d_model)입니다.
        batch_size, seq_len, input_dim = x.shape
        if input_dim != self.d_model:
            raise ValueError(f"expected input dim {self.d_model}, got {input_dim}")

        # 각 토큰 embedding을 Query, Key, Value 공간으로 선형 변환합니다.
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (B, T, d_model)을 (B, T, n_heads, head_dim)으로 행렬 분할하여
        # 여러 attention head가 서로 다른 부분 공간을 보게 합니다.
        keys = keys.view(batch_size, seq_len, self.n_heads, self.head_dim)
        queries = queries.view(batch_size, seq_len, self.n_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # attention 계산을 head별로 병렬 수행하기 위해 head 차원을 앞으로 옮깁니다.
        # 전치: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 각 query 토큰이 모든 key 토큰과 얼마나 관련 있는지 점수로 계산합니다.
        attn_scores = queries @ keys.transpose(2, 3)

        if causal_mask:
            # mask는 register_buffer로 저장하지 않고, 현재 입력 길이에 맞춰 일반 텐서로 만듭니다.
            # shape (T, T)의 2D mask는 attn_scores의 (B, n_heads, T, T)에 자동 broadcast됩니다.
            causal_mask_tensor = torch.triu(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=attn_scores.device,
                ),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask_tensor, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # attention weight로 value를 가중합한 뒤, head 차원을 다시 토큰 차원 뒤로 보냅니다.
        context_vec = (attn_weights @ values).transpose(1, 2)

        # transpose 뒤에는 메모리가 불연속일 수 있으므로 contiguous 후 view로 head를 합칩니다.
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.d_model)
        context_vec = self.out_proj(context_vec)

        if return_attention_weights:
            return context_vec, attn_weights
        return context_vec
