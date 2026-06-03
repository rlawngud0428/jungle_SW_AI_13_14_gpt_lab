# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def _read_sentiment_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text = str(item.get("text", "")).strip()
                if text:
                    rows.append({"text": text, "label": int(item["label"])})
        return rows

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("document") or row.get("text") or "").strip()
            label = row.get("label")
            if not text or label is None:
                continue
            rows.append({"text": text, "label": int(label)})

    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    # raise NotImplementedError("make_sentiment_dataset을 구현하세요.")
    train_rows = _read_sentiment_rows(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_rows)

    val_size = int(len(train_rows) * val_ratio)
    if val_ratio > 0 and len(train_rows) > 1:
        val_size = max(1, val_size)
    val_size = min(val_size, len(train_rows))

    val_data = train_rows[:val_size]
    train_data = train_rows[val_size:]
    test_data = _read_sentiment_rows(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_path / "nsmc_sentiment_train.jsonl", train_data)
        _write_jsonl(output_path / "nsmc_sentiment_val.jsonl", val_data)
        _write_jsonl(output_path / "nsmc_sentiment_test.jsonl", test_data)

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        # raise NotImplementedError("ReviewSentimentDataset.__getitem__을 구현하세요.")
        item = self.data[idx]
        ids = self.tokenizer.encode(item["text"], add_bos_eos=True)
        ids = ids[: self.max_length]
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))

        input_ids = torch.tensor(ids, dtype=torch.long)
        label = int(item["label"])
        return input_ids, label


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        # raise NotImplementedError("GPTForSequenceClassification.__init__을 구현하세요.")
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        # raise NotImplementedError("GPTForSequenceClassification.forward를 구현하세요.")
        x = self.gpt.embedding(input_ids)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.norm(x)

        pooled = x[:, -1, :]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    # raise NotImplementedError("train_epoch_sentiment를 구현하세요.")
    model.to(device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_count += batch_size

    if total_count == 0:
        return float("nan"), float("nan")

    return total_loss / total_count, total_correct / total_count


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    # raise NotImplementedError("evaluate_sentiment를 구현하세요.")
    was_training = model.training
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            loss, logits = model(input_ids, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_count += batch_size

    if was_training:
        model.train()

    if total_count == 0:
        return float("nan"), float("nan")

    return total_loss / total_count, total_correct / total_count
