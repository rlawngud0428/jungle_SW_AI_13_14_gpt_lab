# -*- coding: utf-8 -*-
"""Inference CLI helpers for inspecting a trained language-model run."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _make_biased_model():
    from model import GPTModel

    config = {
        "vocab_size": 300,
        "context_length": 16,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 1,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = GPTModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lm_head.bias[4 + ord("A")] = 3.0
        model.lm_head.bias[4 + ord("B")] = 2.0
        model.lm_head.bias[4 + ord("C")] = 1.0
    return model


def test_predict_next_tokens_returns_decoded_top_k_candidates():
    from bpe import BPETokenizer
    from scripts.infer_lm import predict_next_tokens

    tokenizer = BPETokenizer(vocab_size=300)
    model = _make_biased_model()

    predictions = predict_next_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt="x",
        device=torch.device("cpu"),
        top_k=2,
        context_size=16,
    )

    assert [prediction["text"] for prediction in predictions] == ["A", "B"]
    assert [prediction["token_id"] for prediction in predictions] == [
        4 + ord("A"),
        4 + ord("B"),
    ]
    assert predictions[0]["probability"] > predictions[1]["probability"]


def test_generate_completion_returns_full_and_generated_text():
    from bpe import BPETokenizer
    from scripts.infer_lm import generate_completion

    tokenizer = BPETokenizer(vocab_size=300)
    model = _make_biased_model()

    result = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt="x",
        device=torch.device("cpu"),
        max_new_tokens=3,
        context_size=16,
        temperature=0.0,
        top_k=None,
    )

    assert result["full_text"] == "xAAA"
    assert result["generated_text"] == "AAA"
    assert result["generated_token_ids"] == [4 + ord("A")] * 3
