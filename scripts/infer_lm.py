# -*- coding: utf-8 -*-
"""Inspect next-token predictions and generated text from a trained mini GPT run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from bpe import BPETokenizer
from model import GPTModel
from train import generate, load_checkpoint

from scripts.console import configure_utf8_stdio
from scripts.train_lm import resolve_device, set_seed, write_json


def _encode_prompt(tokenizer: BPETokenizer, prompt: str) -> list[int]:
    token_ids = tokenizer.encode(prompt)
    if token_ids:
        return token_ids
    return [tokenizer.get_bos_id()]


def _decode_token(tokenizer: BPETokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id])
    if text:
        return text
    raw_token = tokenizer.id_to_token.get(token_id)
    if isinstance(raw_token, str):
        return raw_token
    return ""


def load_run(
    run_dir: Path,
    checkpoint: Path | None,
    device: torch.device,
) -> tuple[GPTModel, BPETokenizer, dict, Path]:
    config_path = run_dir / "config.json"
    tokenizer_path = run_dir / "tokenizer.json"
    checkpoint_path = checkpoint if checkpoint is not None else run_dir / "final_checkpoint.pt"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer file: {tokenizer_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")

    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config_payload["model"]

    tokenizer = BPETokenizer(vocab_size=model_config["vocab_size"])
    tokenizer.load(tokenizer_path)

    model = GPTModel(model_config)
    load_checkpoint(model, optimizer=None, path=str(checkpoint_path), device=device)
    model.to(device)
    model.eval()

    return model, tokenizer, config_payload, checkpoint_path


def predict_next_tokens(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    top_k: int = 10,
    context_size: int | None = None,
) -> list[dict]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    context_size = context_size or model.config["context_length"]
    token_ids = _encode_prompt(tokenizer, prompt)
    idx = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(idx[:, -context_size:])
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        values, indices = torch.topk(probabilities, k=min(top_k, probabilities.size(-1)))

    if was_training:
        model.train()

    predictions = []
    for rank, (token_id, probability) in enumerate(
        zip(indices[0].tolist(), values[0].tolist()),
        start=1,
    ):
        predictions.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "text": _decode_token(tokenizer, int(token_id)),
                "probability": float(probability),
            }
        )
    return predictions


def generate_completion(
    model: GPTModel,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
    context_size: int | None = None,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> dict:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0.")

    context_size = context_size or model.config["context_length"]
    token_ids = _encode_prompt(tokenizer, prompt)
    idx = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    output = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id(),
    )

    output_ids = output[0].tolist()
    generated_token_ids = output_ids[len(token_ids) :]
    return {
        "prompt": prompt,
        "prompt_token_ids": token_ids,
        "generated_token_ids": generated_token_ids,
        "full_token_ids": output_ids,
        "generated_text": tokenizer.decode(generated_token_ids),
        "full_text": tokenizer.decode(output_ids),
    }


def _format_token_text(text: str) -> str:
    if text == "":
        return "<empty/special>"
    return repr(text)


def format_inference_report(
    prompt: str,
    predictions: list[dict],
    completion: dict,
    run_dir: Path,
    checkpoint: Path,
    device: torch.device,
) -> str:
    lines = [
        "[Run]",
        f"run_dir: {run_dir}",
        f"checkpoint: {checkpoint}",
        f"device: {device}",
        "",
        "[Prompt]",
        prompt,
        "",
        "[Next token candidates]",
    ]
    for prediction in predictions:
        lines.append(
            f"{prediction['rank']:>2}. "
            f"id={prediction['token_id']:<5} "
            f"prob={prediction['probability']:.4f} "
            f"text={_format_token_text(prediction['text'])}"
        )
    lines.extend(
        [
            "",
            "[Generated]",
            completion["full_text"],
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "full-001")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt", default="이 영화는")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10, help="Number of next-token candidates.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=40,
        help="Top-k sampling limit for generation. Use 0 to disable.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    sample_top_k = None if args.sample_top_k == 0 else args.sample_top_k
    model, tokenizer, config_payload, checkpoint_path = load_run(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        device=device,
    )
    context_size = config_payload["model"]["context_length"]
    predictions = predict_next_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        top_k=args.top_k,
        context_size=context_size,
    )
    completion = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        context_size=context_size,
        temperature=args.temperature,
        top_k=sample_top_k,
    )

    payload = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "prompt": args.prompt,
        "next_token_candidates": predictions,
        "completion": completion,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output_json, payload)

    print(
        format_inference_report(
            prompt=args.prompt,
            predictions=predictions,
            completion=completion,
            run_dir=args.run_dir,
            checkpoint=checkpoint_path,
            device=device,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
