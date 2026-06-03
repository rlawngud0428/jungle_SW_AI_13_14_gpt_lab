# PTE/BPE Tokenizer 기반 학습 파라미터 실험 가이드

이 문서는 이미 학습해 둔 tokenizer 산출물과 현재 언어 모델 학습 결과를 기준으로, 학습 파라미터를 바꿔볼 때 무엇을 기대할 수 있고 어떤 단점이 생기는지 정리한 참고 문서입니다.

현재 코드 기준 tokenizer 이름은 `BPETokenizer`이며, 각 학습 run의 tokenizer 데이터는 `runs/<run-name>/tokenizer.json`에 저장됩니다. 질문에서 말한 PTE Tokenizer는 이 프로젝트의 byte-level BPE tokenizer 산출물 기준으로 해석했습니다.

## 핵심 요약

파라미터를 바꿔 실험할 때 가장 중요한 원칙은 한 번에 하나의 축만 바꾸는 것입니다.

| 실험 목적 | 우선 고정할 것 | 바꿔볼 것 | 주로 볼 지표 |
| --- | --- | --- | --- |
| 모델 학습 파라미터 비교 | tokenizer, `train_chars`, `val_chars`, `seed`, `device` | `learning_rate`, `batch_size`, `epochs`, `drop_rate` | `val_loss`, train/val gap, `elapsed_sec` |
| 모델 크기 비교 | tokenizer, 데이터 크기, 학습률 | `emb_dim`, `n_layers`, `n_heads` | `val_loss`, 메모리, 학습 속도 |
| 문맥 길이 비교 | tokenizer, 모델 크기, 학습률 | `context_length` | `val_loss`, 생성 샘플, 메모리 |
| tokenizer 자체 비교 | 데이터 split, 모델 크기, seed | `vocab_size`, tokenizer 학습 데이터 크기 | token 수, 생성 품질, downstream 성능 |

`vocab_size`나 tokenizer 학습 데이터가 바뀌면 token ID 체계와 sequence 길이가 달라집니다. 이 경우 `val_loss` 숫자를 이전 run과 단순 비교하기 어렵고, token 수, 생성 샘플, 감성 분류 fine-tuning 성능까지 함께 봐야 합니다.

## 현재 학습 흐름

`scripts/train_lm.py`를 실행하면 아래 순서로 동작합니다.

1. `data/nsmc_lm_train.txt`, `data/nsmc_lm_val.txt`를 읽습니다.
2. train text로 byte-level BPE tokenizer를 학습합니다.
3. 학습된 tokenizer를 `runs/<run-name>/tokenizer.json`에 저장합니다.
4. train/validation text를 token ID로 변환합니다.
5. `context_length` 단위의 next-token prediction dataset을 만듭니다.
6. GPT 모델을 처음부터 학습합니다.
7. `config.json`, `final_checkpoint.pt`, `metrics.json`, `training_curves.png`를 저장합니다.

주의할 점은 현재 학습 CLI가 매 run마다 tokenizer를 새로 학습한다는 점입니다. 같은 tokenizer 기준으로 모델 파라미터만 엄밀히 비교하려면 다음 중 하나를 지켜야 합니다.

| 방법 | 장점 | 단점 |
| --- | --- | --- |
| 같은 `vocab_size`, 같은 `train_chars`를 사용 | 현재 코드 그대로 비교 가능. BPE 학습은 deterministic이라 같은 입력이면 같은 merge를 기대할 수 있음 | 코드가 명시적으로 기존 `tokenizer.json`을 load하는 것은 아니므로 실험 조건을 꼼꼼히 기록해야 함 |
| 학습 스크립트에 `--tokenizer-path` 옵션을 추가 | 기존 tokenizer를 확실히 고정 가능 | 코드 변경이 필요함 |
| 매번 tokenizer까지 새로 학습 | 실제 전체 파이프라인 변경 효과를 볼 수 있음 | tokenizer 변화와 모델 파라미터 변화가 섞여 원인 분석이 어려움 |

## 현재 저장된 run 기준선

아래 값들은 현재 `runs/` 아래에 저장된 `config.json`, `metrics.json` 기준입니다.

| run | preset | vocab | ctx | emb | layers | heads | batch | epochs | step | train_loss | val_loss | elapsed_sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `codex-accuracy-plot` | smoke | 300 | 16 | 32 | 1 | 4 | 2 | 1 | 109 | 5.3923 | 5.3596 | 2.28 |
| `codex-smoke` | smoke | 300 | 32 | 32 | 1 | 4 | 2 | 1 | 137 | 5.0477 | 5.0259 | 2.75 |
| `dev-001` | dev | 1000 | 64 | 64 | 2 | 4 | 8 | 2 | 646 | 6.2215 | 6.2378 | 13.16 |
| `dev-progress` | dev | 1000 | 64 | 64 | 2 | 4 | 8 | 2 | 646 | 6.2215 | 6.2378 | 9.62 |
| `full-001` | full | 3000 | 128 | 128 | 4 | 4 | 16 | 5 | 1965 | 5.8040 | 5.9412 | 47.39 |
| `smoke-progress` | smoke | 300 | 32 | 32 | 1 | 4 | 2 | 1 | 137 | 5.0477 | 5.0259 | 2.06 |
| `smoke-test` | smoke | 300 | 32 | 32 | 1 | 4 | 2 | 1 | 137 | 5.0477 | 5.0259 | 2.04 |

이 표는 현재 상태를 보는 기준선입니다. 다만 `smoke`, `dev`, `full`은 tokenizer vocabulary, context 길이, 모델 크기, 데이터 크기가 모두 다르므로 단순히 `val_loss`만 보고 어느 쪽이 무조건 좋다고 판단하면 안 됩니다.

## 파라미터별 장단점

### Tokenizer와 데이터

| 파라미터 | 늘리거나 바꿨을 때 장점 | 단점과 주의점 | 추천 실험 |
| --- | --- | --- | --- |
| `--vocab-size` | 자주 나오는 byte pair가 하나의 token으로 묶여 sequence가 짧아질 수 있음. 같은 `context_length`에서 더 긴 실제 문맥을 볼 수 있음 | output layer가 커져 메모리와 연산량이 증가함. token 체계가 바뀌므로 기존 checkpoint와 호환되지 않을 수 있음. vocab이 다른 run끼리 loss 직접 비교가 어려움 | `1000 -> 3000 -> 5000`처럼 단계적으로 비교 |
| `--train-chars` | tokenizer와 모델이 더 많은 문장 패턴을 볼 수 있음. merge rule이 작은 샘플에 과하게 맞는 문제를 줄일 수 있음 | 학습 시간이 늘어남. tokenizer 학습 corpus가 바뀌면 tokenization도 바뀌어 원인 분석이 어려워짐 | tokenizer 실험이 아니라면 고정 |
| `--val-chars` | validation loss 추정이 더 안정적임 | 평가 시간이 늘어남 | 빠른 실험은 작게, 최종 비교는 크게 |

Tokenizer 실험을 할 때는 `train_tokens`, `val_tokens`도 같이 봐야 합니다. vocab이 커지면 같은 문장이라도 token 수가 줄 수 있고, 이 변화가 `context_length`, batch 수, loss 해석에 같이 영향을 줍니다.

### 문맥과 batch

| 파라미터 | 늘렸을 때 장점 | 단점과 주의점 | 추천 실험 |
| --- | --- | --- | --- |
| `--context-length` | 모델이 더 긴 앞 문맥을 보고 다음 token을 예측함. 생성 문장이 더 일관될 가능성이 있음 | attention 메모리와 시간이 대략 sequence 길이에 크게 민감하게 증가함. 현재 dataset은 기본 `stride=context_length`라 context가 길어지면 epoch당 sample 수가 줄 수 있음 | `64`, `128`, `256` 순서로 메모리 확인 |
| `--batch-size` | gradient가 덜 흔들리고 GPU 처리량이 좋아질 수 있음 | CUDA 메모리를 많이 씀. batch가 커지면 epoch당 update step 수가 줄어 같은 epoch라도 weight update 횟수가 달라짐 | 메모리가 허용하는 범위에서 `8`, `16`, `32` 비교 |

메모리가 부족하면 보통 `batch_size`, `context_length`, `emb_dim`, `n_layers` 순서로 줄여봅니다. 가장 즉각적으로 효과가 나는 것은 `batch_size`입니다.

### 모델 크기

| 파라미터 | 늘렸을 때 장점 | 단점과 주의점 | 추천 실험 |
| --- | --- | --- | --- |
| `--emb-dim` | token 표현 차원이 커져 더 복잡한 패턴을 담을 수 있음 | embedding, attention, FFN, output layer가 모두 커져 느려지고 메모리를 더 씀. `emb_dim % n_heads == 0`이어야 함 | `64 -> 128 -> 256` |
| `--n-layers` | Transformer block이 깊어져 더 복잡한 문맥 변환을 학습할 수 있음 | 느려지고 overfitting 위험이 커짐. 데이터가 작으면 효과보다 비용이 클 수 있음 | `2 -> 4 -> 6` |
| `--n-heads` | 여러 attention 관점으로 문맥을 볼 수 있음 | `emb_dim`이 고정되어 있으면 head당 차원이 작아짐. 너무 많으면 overhead만 늘 수 있음 | `emb_dim=128`이면 `4`, `8` 비교 |
| `--drop-rate` | overfitting을 줄이는 regularization 효과 | 너무 높으면 underfitting이 생기고 train loss도 잘 내려가지 않음 | `0.0`, `0.1`, `0.2` |

모델 크기를 키웠는데 `train_loss`와 `val_loss`가 모두 내려가지 않으면 학습률, 데이터 크기, epoch 수가 부족할 수 있습니다. `train_loss`만 내려가고 `val_loss`가 나빠지면 모델이 큰 것보다 regularization이나 데이터가 더 필요한 상황일 수 있습니다.

### Optimizer와 학습 길이

| 파라미터 | 늘렸을 때 장점 | 단점과 주의점 | 추천 실험 |
| --- | --- | --- | --- |
| `--learning-rate` | 적절히 크면 loss가 빠르게 내려감 | 너무 크면 loss가 불안정하거나 발산할 수 있음. 너무 작으면 학습이 거의 진행되지 않음 | `1e-4`, `3e-4`, `5e-4` |
| `--epochs` | 더 오래 학습해 loss가 더 내려갈 수 있음 | 시간이 늘고 overfitting 가능성이 커짐 | train/val gap을 보며 `2`, `5`, `10` 비교 |
| `--eval-freq` | 값을 작게 하면 학습 곡선을 더 촘촘히 볼 수 있음 | 평가가 잦아져 느려짐 | 긴 학습은 `200` 이상, 짧은 학습은 `10~100` |
| `--eval-iter` | 평가 batch 수가 커져 loss 추정이 안정적임 | 평가 시간이 늘어남 | 빠른 실험은 `5~10`, 최종 비교는 `20` 이상 |
| `--ckpt-freq` | 중간 checkpoint를 저장해 실패 시 복구 가능 | 디스크 사용량 증가 | 긴 학습에서만 `500` 등으로 설정 |
| `--seed` | 같은 설정을 재현하기 쉬움 | seed 하나만 보면 우연한 초기화 효과를 놓칠 수 있음 | 최종 후보는 여러 seed로 확인 |

## 결과 해석 체크리스트

| 관찰 | 해석 | 다음 액션 |
| --- | --- | --- |
| train_loss와 val_loss가 함께 감소 | 학습이 정상적으로 진행 중 | 같은 설정으로 epoch를 조금 늘리거나 모델 크기 실험 |
| train_loss는 감소하지만 val_loss가 정체 또는 상승 | overfitting 가능성 | `drop_rate` 증가, 모델 축소, 데이터 증가, epoch 감소 |
| train_loss와 val_loss가 모두 높고 잘 내려가지 않음 | underfitting 또는 학습률 문제 | 모델 크기 증가, `learning_rate` 조정, epoch 증가 |
| loss가 출렁이거나 갑자기 커짐 | 학습률이 크거나 batch가 너무 작을 수 있음 | `learning_rate` 감소, batch 증가 |
| elapsed_sec가 크게 증가했는데 val_loss 개선이 작음 | 비용 대비 효과가 낮음 | 더 작은 모델이나 짧은 context로 회귀 |
| token accuracy는 오르는데 생성 문장이 어색함 | next-token accuracy만으로 생성 품질 판단이 어려움 | 생성 샘플과 downstream fine-tuning 성능도 확인 |

`val_loss`는 next-token prediction cross entropy입니다. 감성 분류 정확도가 아니라 "다음 token을 얼마나 잘 맞히는지"를 보는 값입니다.

## 추천 실험 순서

### 1단계: baseline 고정

먼저 기준이 되는 run을 하나 정합니다. 빠르게 보려면 `dev`, 더 진지하게 비교하려면 `full`을 기준으로 잡습니다.

```powershell
python scripts/train_lm.py --preset dev --run-name exp-base-dev --seed 42
```

### 2단계: 학습률만 비교

모델과 tokenizer를 고정하고 `learning_rate`만 바꿉니다.

```powershell
python scripts/train_lm.py --preset dev --run-name exp-lr-1e-4 --seed 42 --learning-rate 1e-4
python scripts/train_lm.py --preset dev --run-name exp-lr-3e-4 --seed 42 --learning-rate 3e-4
python scripts/train_lm.py --preset dev --run-name exp-lr-5e-4 --seed 42 --learning-rate 5e-4
```

### 3단계: batch와 context 비교

메모리가 허용하는 범위에서 `batch_size`와 `context_length`를 따로 비교합니다.

```powershell
python scripts/train_lm.py --preset dev --run-name exp-ctx-128 --seed 42 --context-length 128
python scripts/train_lm.py --preset dev --run-name exp-batch-16 --seed 42 --batch-size 16
```

### 4단계: 모델 크기 비교

`emb_dim`, `n_layers`는 비용이 크게 늘어나는 축입니다. 한 번에 같이 키우기보다 하나씩 바꿉니다.

```powershell
python scripts/train_lm.py --preset dev --run-name exp-emb-128 --seed 42 --emb-dim 128 --n-heads 4
python scripts/train_lm.py --preset dev --run-name exp-layer-4 --seed 42 --n-layers 4
```

### 5단계: tokenizer vocabulary 비교

이 단계는 모델 파라미터 실험이 아니라 tokenizer까지 포함한 파이프라인 실험입니다. loss 숫자만 보지 말고 token 수와 생성 샘플도 같이 봅니다.

```powershell
python scripts/train_lm.py --preset dev --run-name exp-vocab-1000 --seed 42 --vocab-size 1000
python scripts/train_lm.py --preset dev --run-name exp-vocab-3000 --seed 42 --vocab-size 3000
python scripts/train_lm.py --preset dev --run-name exp-vocab-5000 --seed 42 --vocab-size 5000
```

## 비교 기록 템플릿

실험할 때는 아래 표를 복사해서 채우면 원인 분석이 쉬워집니다.

| run | 고정한 tokenizer 조건 | 바꾼 파라미터 | train_loss | val_loss | val-train gap | val_accuracy | elapsed_sec | 판단 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `exp-base-dev` | vocab 1000, train_chars 200000 | baseline |  |  |  |  |  |  |
| `exp-lr-1e-4` | vocab 1000, train_chars 200000 | lr 1e-4 |  |  |  |  |  |  |
| `exp-lr-5e-4` | vocab 1000, train_chars 200000 | lr 5e-4 |  |  |  |  |  |  |

좋은 후보를 고를 때는 단순히 `val_loss`가 가장 낮은 run만 보지 말고 아래를 함께 확인합니다.

- `train_loss`와 `val_loss` 차이가 너무 크지 않은가
- 같은 시간 대비 개선폭이 충분한가
- `training_curves.png`에서 loss가 안정적으로 내려가는가
- 생성 샘플이 이전보다 덜 깨지고 반복이 줄었는가
- 감성 분류 fine-tuning까지 할 계획이면 downstream accuracy가 좋아지는가

## 상황별 빠른 판단

| 상황 | 먼저 시도할 변경 |
| --- | --- |
| CUDA 메모리 부족 | `--batch-size` 감소, `--context-length` 감소 |
| 학습이 너무 느림 | `--context-length`, `--emb-dim`, `--n-layers` 감소 |
| loss가 거의 안 내려감 | `--learning-rate` 조정, `--epochs` 증가, 모델 크기 증가 |
| train은 좋아지고 val은 나빠짐 | `--drop-rate` 증가, `--epochs` 감소, 모델 크기 감소 |
| 생성 문장이 너무 짧은 패턴만 반복 | `--context-length` 증가, 데이터 증가, epoch 증가 |
| tokenizer가 너무 잘게 쪼개지는 느낌 | `--vocab-size` 증가 |

## 최종 결론

현재 저장된 `full-001`은 더 긴 학습을 거친 기준선으로 볼 수 있고, `dev` preset은 파라미터 탐색용으로 적당합니다. 처음에는 tokenizer 조건을 고정한 채 `learning_rate`, `batch_size`, `context_length`, `drop_rate`를 하나씩 비교하고, 그 다음에 `emb_dim`, `n_layers`, `vocab_size`처럼 비용과 해석 난도가 큰 축을 실험하는 순서가 좋습니다.

특히 `vocab_size`를 바꾸는 순간 "같은 tokenizer 기반 모델 파라미터 비교"가 아니라 "tokenizer까지 포함한 전체 파이프라인 비교"가 됩니다. 이 차이만 놓치지 않으면 실험 결과를 훨씬 덜 헷갈리게 읽을 수 있습니다.
