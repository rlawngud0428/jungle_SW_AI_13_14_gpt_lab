# -*- coding: utf-8 -*-
"""GPT 사전 학습 유틸리티 과제 템플릿."""

import matplotlib.pyplot as plt
import torch

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: GPTModel,
    device: torch.device,
) -> torch.Tensor:
    """TODO: 한 배치를 device로 옮긴 뒤 다음 토큰 예측 cross entropy loss를 계산합니다."""
    # raise NotImplementedError("calc_loss_batch를 구현하세요.")
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, target_batch)
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    # raise NotImplementedError("calc_loss_loader를 구현하세요.")
    if len(data_loader) == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    batches_seen = 0
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            if batches_seen >= max_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            batches_seen += 1

    if was_training:
        model.train()

    return total_loss / batches_seen if batches_seen > 0 else float("nan")


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    # raise NotImplementedError("save_checkpoint를 구현하세요.")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    # raise NotImplementedError("load_checkpoint를 구현하세요.")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return int(checkpoint.get("epoch", 0)), int(checkpoint.get("global_step", 0))


def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """TODO: temperature와 top-k 샘플링을 지원하는 생성 함수를 구현합니다."""
    # raise NotImplementedError("generate를 구현하세요.")
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                top_values, _ = torch.topk(logits, k)
                min_top_value = top_values[:, [-1]]
                logits = torch.where(
                    logits < min_top_value,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

            if eos_id is not None and torch.all(next_token == eos_id):
                break

    if was_training:
        model.train()

    return idx


def generate_and_print_sample(
    model: GPTModel,
    tokenizer,
    device: torch.device,
    start_context: str,
    max_new_tokens: int = 50,
    context_size: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> None:
    """TODO: start_context를 encode하고 generate 후 decode하여 출력합니다."""
    # raise NotImplementedError("generate_and_print_sample을 구현하세요.")
    model.eval()
    encoded = tokenizer.encode(start_context)
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None,
    )
    decoded = tokenizer.decode(out[0].tolist())
    print(decoded)


def train_model(
    model: GPTModel,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    ckpt_freq: int | None = None,
    start_epoch: int = 0,
    global_step: int = 0,
) -> list[float]:
    """TODO: 사전 학습 루프를 구현하고 epoch별 train loss 리스트를 반환합니다."""
    # raise NotImplementedError("train_model을 구현하세요.")
    train_losses = []
    model.to(device)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0.0
        batches_seen = 0

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches_seen += 1
            global_step += 1

            if eval_freq > 0 and global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
                print(
                    f"Ep {epoch + 1} (step {global_step}): "
                    f"train loss {train_loss:.3f}, val loss {val_loss:.3f}"
                )

            if ckpt_freq is not None and ckpt_freq > 0 and global_step % ckpt_freq == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    path=f"checkpoint_step_{global_step}.pt",
                )

        avg_epoch_loss = epoch_loss / batches_seen if batches_seen > 0 else float("nan")
        train_losses.append(avg_epoch_loss)
        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context=start_context,
            context_size=model.config.get("context_length", 256),
        )

    return train_losses


def plot_losses(train_losses: list[float], val_losses: list[float] | None = None) -> None:
    """훈련/검증 손실 그래프를 그리는 제공 함수."""
    plt.plot(train_losses, label="Train")
    if val_losses is not None:
        plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training / Validation Loss")
    plt.show()
