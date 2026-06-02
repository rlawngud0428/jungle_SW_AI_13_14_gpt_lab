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
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    loss, _ = model(input_batch, target_batch)  # GPTModel(idx, targets) -> return loss, logits
    return loss


def calc_loss_loader(
    data_loader,
    model: GPTModel,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """TODO: data_loader의 평균 loss를 계산합니다. 검증에서는 torch.no_grad()를 사용하세요."""
    if num_batches == 0:
        return float("nan")

    was_training = model.training
    model.eval()

    total_loss = 0.0
    num_processed_batches = 0

    with torch.no_grad():
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            num_processed_batches += 1

    model.train(was_training)

    if num_processed_batches == 0:
        return float("nan")

    return total_loss / num_processed_batches


def save_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    path: str,
) -> None:
    """TODO: model/optimizer 상태, epoch, global_step을 torch.save로 저장합니다."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: GPTModel,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> tuple[int, int]:
    """TODO: torch.load로 checkpoint를 읽어 model/optimizer 상태를 복원합니다."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["global_step"]


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
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                topk_values, _ = torch.topk(logits, k)
                min_topk_value = topk_values[:, -1, None]
                logits = logits.masked_fill(logits < min_topk_value, float("-inf"))

            if temperature <= 0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and (idx_next == eos_id).all():
                break

    model.train(was_training)

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
    encoded = tokenizer.encode(start_context)
    if not encoded:
        if hasattr(tokenizer, "get_bos_id"):
            encoded = [tokenizer.get_bos_id()]
        else:
            raise ValueError("start_context must encode to at least one token")

    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.get_eos_id() if hasattr(tokenizer, "get_eos_id") else None

    generated = generate(
        model,
        idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    decoded_text = tokenizer.decode(generated[0].tolist())
    print(decoded_text)


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
    """
        1. 모델을 device로 이동
        2. epoch 반복
        3. batch 반복
        4. gradient 초기화
        5. loss 계산
        6. backward로 gradient 계산
        7. optimizer.step으로 모델 업데이트
        8. 주기적으로 train/val loss 평가
        9. 주기적으로 checkpoint 저장
        10. epoch 평균 loss 기록
        11. epoch마다 샘플 문장 생성
        12. train_losses 반환
    """
    # 모델 파라미터와 입력 배치가 같은 장치에 있어야 forward/backward가 동작합니다.
    model.to(device)

    # 샘플 생성 때는 모델이 가진 최대 문맥 길이(context_length)를 넘지 않도록 자릅니다.
    # config가 없는 예외적인 모델을 위해 기본값 256을 둡니다.
    sample_context_size = model.config.get("context_length", 256)

    # 각 epoch가 끝날 때 계산한 평균 train loss를 이 리스트에 모아 반환합니다.
    train_losses = []

    # start_epoch는 checkpoint에서 이어 학습할 때 이미 끝낸 epoch를 건너뛰기 위한 값입니다.
    # 예: start_epoch=2, num_epochs=5이면 epoch index 2,3,4를 학습합니다.
    for epoch in range(start_epoch, num_epochs):
        # 학습 단계에서는 dropout 같은 layer가 학습 모드로 동작해야 합니다.
        model.train()

        # epoch_loss는 이번 epoch 안의 batch loss 합입니다.
        # 마지막에 batch 수로 나누어 epoch 평균 loss를 만듭니다.
        epoch_loss = 0.0

        # DataLoader가 비어 있을 수도 있으므로 실제 처리한 batch 수를 직접 셉니다.
        num_processed_batches = 0

        # train_loader는 (input_batch, target_batch)를 batch 단위로 돌려줍니다.
        # input_batch는 현재 토큰들이고 target_batch는 한 칸 오른쪽으로 밀린 다음 토큰 정답입니다.
        for input_batch, target_batch in train_loader:
            # 이전 batch에서 계산된 gradient가 parameter.grad에 남아 있으므로 매 step 초기화합니다.
            optimizer.zero_grad()

            # calc_loss_batch는 batch를 device로 옮기고 next-token cross entropy loss를 계산합니다.
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # loss에서 시작해 모든 trainable parameter에 대한 gradient를 계산합니다.
            loss.backward()

            # optimizer가 gradient를 사용해 모델 파라미터를 한 번 갱신합니다.
            optimizer.step()

            # loss.item()은 scalar Tensor를 Python float로 꺼냅니다.
            # 평균 계산용 누적에는 graph가 필요 없으므로 item()을 사용합니다.
            epoch_loss += loss.item()

            # 이번 epoch에서 평균을 낼 때 사용할 batch 수입니다.
            num_processed_batches += 1

            # global_step은 epoch와 무관하게 optimizer update가 몇 번 일어났는지 세는 값입니다.
            # checkpoint 재개 후에도 이어서 증가해야 eval/checkpoint 주기가 유지됩니다.
            global_step += 1

            # eval_freq가 양수이면 global_step 기준으로 주기적 평가를 수행합니다.
            # eval_freq=0 이하이면 평가 출력을 끕니다.
            if eval_freq > 0 and global_step % eval_freq == 0:
                # train loss도 전체 train_loader가 아니라 eval_iter개 batch만 샘플링해 빠르게 추정합니다.
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)

                # validation loader가 없을 때도 학습은 계속할 수 있도록 NaN으로 표시합니다.
                val_loss = (
                    calc_loss_loader(val_loader, model, device, eval_iter)
                    if val_loader is not None
                    else float("nan")
                )

                # 학습 중간 상태를 사람이 볼 수 있도록 현재 step의 train/val loss를 출력합니다.
                print(
                    f"Step {global_step}: "
                    f"train loss {train_loss:.4f}, val loss {val_loss:.4f}"
                )

                # calc_loss_loader는 내부에서 model.eval()을 호출하므로 학습을 이어가려면 다시 train 모드로 돌립니다.
                model.train()

            # ckpt_freq가 지정되면 global_step 기준으로 checkpoint를 저장합니다.
            # None 또는 0 이하이면 checkpoint 자동 저장을 사용하지 않습니다.
            if (
                ckpt_freq is not None
                and ckpt_freq > 0
                and global_step % ckpt_freq == 0
            ):
                # epoch + 1을 저장하면 "다음에 시작할 epoch 번호"로 해석하기 쉽습니다.
                # global_step은 optimizer update 횟수라 재개 후 eval/checkpoint 주기를 맞추는 데 필요합니다.
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    global_step,
                    f"checkpoint_step_{global_step}.pt",
                )

        # batch가 하나 이상 있으면 평균 loss를 계산하고, 비어 있으면 NaN으로 남깁니다.
        avg_epoch_loss = (
            epoch_loss / num_processed_batches
            if num_processed_batches > 0
            else float("nan")
        )

        # 반환값은 epoch별 평균 train loss 리스트입니다.
        train_losses.append(avg_epoch_loss)

        # epoch 번호는 사람이 읽기 쉽게 1부터 출력합니다.
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"train loss {avg_epoch_loss:.4f}"
        )

        # 같은 start_context로 매 epoch 샘플을 출력하면 모델 출력 변화를 눈으로 비교하기 쉽습니다.
        generate_and_print_sample(
            model,
            tokenizer,
            device,
            start_context,
            context_size=sample_context_size,
        )

        # generate_and_print_sample -> generate 경로에서 eval 모드가 될 수 있으므로 다음 epoch 전 train 모드로 복원합니다.
        model.train()

    # 학습 곡선을 그리거나 보고서에 기록할 수 있도록 epoch 평균 loss들을 반환합니다.
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
