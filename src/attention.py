# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn
import math


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
        self.head_dim = d_model // n_heads
        # TODO: qkv projection, output projection, dropout을 정의하세요.
        # raise NotImplementedError("MultiHeadAttention.__init__을 구현하세요.")
        self.q = nn.Linear(self.d_model, self.d_model)
        self.k = nn.Linear(self.d_model, self.d_model)
        self.v = nn.Linear(self.d_model, self.d_model)

        self.output = nn.Linear(self.d_model, self.d_model)

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
        # raise NotImplementedError("MultiHeadAttention.forward를 구현하세요.")

        # Q,K,V 만들기
        batch_size, seq_len, d_model = x.shape
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        # C를 이제 multi head로 바꾸기
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 계산하기 쉽게? 자리 옮겨주기
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 행렬곱을 하기 위해서 마지막 두 차원이 맞아야 한다.
        # 그래서 Q (seq_len, head_dim) 이므로 K를 뒤집는다. (head_dim, seq_len)
        # 그럼 결과가 (seq_len, seq_len)이 된다. -> 각 토큰이 각 토큰을 얼마나 볼지를 의미한다.
        reversed_K = K.transpose(2, 3)
        score = Q @ reversed_K / math.sqrt(self.head_dim)

        if (causal_mask):
            # 코잘 어텐션을 위한 (미래를 가리기위한) 마스크 생성
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal= 1)
            masked_score = score.masked_fill(mask.bool(), -torch.inf)
            score = masked_score
        
        # attention_weight = torch.softmax(score, dim=seq_len)
        # 위에 작성한 것처럼 했다가 테스트 실패함. 마지막 차원을 기준으로 softmax를 적용한다길래 seq_len을 넣었다가 실패.
        # 마지막 차원이니 그냥 dim=-1을 넣어주면 됨.
        attention_weight = torch.softmax(score, dim= -1)
        attention_weight = self.dropout(attention_weight)
        
        # output = attention_weight * V
        # 위에 작성한 것처럼 했다가 테스트 실패함. 행렬곱은 당연히 그냥 *로 할 수 없음..
        output = attention_weight @ V
        output = output.transpose(1, 2)

        # output = output.view(batch_size, seq_len, self.n_heads * self.head_dim)
        # 위에 작성한 것처럼 했다가 테스트 실패함. transpose를 진행한 뒤에는 메모리가 불연속적일 수 있어서 reshape를 사용하거나, 메모리 정리후 view 사용
        output = output.reshape(batch_size, seq_len, self.n_heads * self.head_dim)

        # 위에서 만들어준 output도 한번 통과시켜 줘야함
        output = self.output(output)

        if (return_attention_weights):
            return (output, attention_weight)
        return output
