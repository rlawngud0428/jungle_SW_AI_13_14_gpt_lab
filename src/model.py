# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        # raise NotImplementedError("LayerNorm.forward를 구현하세요.")
        mean = x.mean(dim= -1, keepdim= True)
        var = x.var(dim= -1, keepdim= True, unbiased= False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        # raise NotImplementedError("GELU.forward를 구현하세요.")

        # return torch.nn.GELU(x)
        # 위에건 틀림. 위에건 nn.module의 클래스임
        return torch.nn.functional.gelu(x)


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        # raise NotImplementedError("FeedForward.__init__을 구현하세요.")

        self.network = nn.Sequential(
            nn.Linear(d_model, mult * d_model),
            GELU(),
            nn.Linear(mult * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        # raise NotImplementedError("FeedForward.forward를 구현하세요.")

        return self.network(x)


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        # raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias)
        self.ffn = FeedForward(d_model=d_model, dropout=drop_rate)

        # layernorm이 2개 필요하다고 해서 2개 만들었는데 왜 2개가 필요할까
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)

        # torch.dropout(drop_rate)
        # 위의 건 아니라고 함. 왜지?
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        # raise NotImplementedError("TransformerBlock.forward를 구현하세요.")
        
        # temp = x
        # x = x + self.layernorm1
        # x = x + self.attention
        # x = x + self.dropout
        # x = x + temp
        # residual connection이 그냥 더하기 라고 해서 무작정 더했더니 이건 아니라고 한다.
        # 왜냐? 점마들은 모듈이니까

        x = x + self.dropout(self.attention(self.layernorm1(x), causal_mask=causal_mask))
        x = x + self.dropout(self.ffn(self.layernorm2(x)))

        return x


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        # raise NotImplementedError("GPTModel.__init__을 구현하세요.")

        # 인자로 쓸 것들 선언
        vocab_size = config["vocab_size"]
        context_length = config["context_length"]
        emb_dim = config["emb_dim"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        drop_rate = config["drop_rate"]
        qkv_bias = config["qkv_bias"]

        # 함수도 선언? 함수라고 봐야하나?
        self.embedding = InputEmbedding(vocab_size=vocab_size, emb_dim=emb_dim, context_length=context_length, drop_rate=drop_rate)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=emb_dim, n_heads=n_heads, drop_rate=drop_rate, qkv_bias=qkv_bias) for _ in range(n_layers)])
        self.norm = LayerNorm(normalized_shape=emb_dim)
        
        # nn.Linear는 입력 벡터에 선형 변환을 적용하는 layer
        # y = xW^T + b (수식)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        # raise NotImplementedError("GPTModel.forward를 구현하세요.")

        # 받은 토큰 ID 리스트를 벡터로
        x = self.embedding(idx)

        # 트랜스포머 블록 개수 만큼 순환
        for block in self.blocks:
            x = block(x)

        # 정규화
        x = self.norm(x)

        # 벡터를 vocab의 토큰 스코어로
        logits = self.lm_head(x)

        if (targets is not None):
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            # logits.size(-1) -> vocab_size
            # 원하는 입력 값 -> input : (16, 1000) | target : (16, )

            return (loss, logits)
        else:
            return logits
            


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    # raise NotImplementedError("generate_text_simple을 구현하세요.")

    # 새 토큰을 몇 개 만들지 정하는 반복문
    for _ in range(max_new_tokens):

        # 현재까지의 토큰부터 context_size 만큼 가져옴
        idx_cond = idx[:, -context_size:]

        # 이 안에서는 gradiant를 계산하지 않음 (why?) -> 텍스트 생성은 학습이 아닌 추론. weight를 업데이트 X
        with torch.no_grad():

            # 모델에 context를 넣어서 다음 토큰 예측 점수를 얻음 (이건 확률이 아닌 score -> softmax 적용 전)
            logits = model(idx_cond)

        # 모델은 모든 위치에 대해서 다음 토큰 예측을 하는데, 우리는 마지막에서 다음 토큰 예측만 필요하므로
        # 마지막 위치의 logits만 가져옴
        logits = logits[:, -1, :]

        # vocab_size 차원에서 가장 큰 값을 찾겠다.
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # 새로 고른 토큰을 기존 토큰 뒤에 붙인다.
        idx = torch.cat((idx, next_token), dim=1)

    # 최종 토큰 시퀀스를 반환
    return idx
