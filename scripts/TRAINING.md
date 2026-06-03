# 학습 및 벤치마크 가이드

이 프로젝트는 pretrained tokenizer나 pretrained model을 사용하지 않습니다. 기본 학습 명령은 먼저 로컬 byte-level BPE tokenizer를 학습한 뒤, `data/nsmc_lm_train.txt`로 mini GPT 언어 모델을 학습합니다. 단, `--tokenizer-path`를 주면 기존에 저장한 BPE tokenizer를 재사용할 수 있습니다.

## 1. 환경 활성화

이미 만들어진 Conda 환경을 사용합니다.

```powershell
cd D:\jungleCamp\w13-14\gpt-lab
conda activate gpt-lab
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`scripts/train_lm.py`와 `scripts/benchmark_lm.py`는 실행 시작 시 stdout/stderr를 UTF-8로 설정합니다. 그래서 학습 로그, JSON 출력, 한국어 샘플이 Windows에서도 최대한 깨지지 않게 출력됩니다.

그래도 PowerShell에서 한글이 깨져 보이면 터미널 인코딩을 UTF-8로 바꾼 뒤 다시 실행합니다.

```powershell
chcp 65001
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
```

주의할 점이 하나 있습니다. 학습 초반에 생성되는 샘플 문장이 어색하거나 `?`가 섞여 보일 수 있는데, 이것은 콘솔 인코딩 문제가 아니라 아직 모델이 충분히 학습되지 않았기 때문일 수 있습니다. 랜덤하게 초기화된 작은 모델은 초반에 의미 없는 token을 많이 생성합니다.

데이터 파일이 없다면 먼저 생성합니다.

```powershell
python download_data.py
```

긴 학습을 돌리기 전에는 테스트를 먼저 확인합니다.

```powershell
python -m pytest tests/ -q
```

## 2. 테스트는 무엇을 확인하나

이 프로젝트의 테스트는 "모델 품질이 좋다"를 검증하는 것이 아니라, 학습에 필요한 각 부품이 올바른 shape와 저장/로드 규약으로 동작하는지 확인합니다. 긴 학습 전에 깨진 구현이나 CUDA/CLI 문제를 빨리 잡기 위한 안전장치라고 보면 됩니다.

전체 테스트는 아래 명령으로 실행합니다.

```powershell
python -m pytest tests/ -q
```

각 테스트 파일이 확인하는 내용은 다음과 같습니다.

| 테스트 파일 | 확인하는 내용 |
| --- | --- |
| `tests/test_bpe.py` | 특수 토큰 ID, byte token 초기화, BPE train, encode/decode 복원, tokenizer save/load |
| `tests/test_dataset.py` | GPTDataset 길이 계산, input/target token shift, DataLoader batch shape, InputEmbedding 출력 shape |
| `tests/test_attention.py` | Multi-head attention 출력 shape, causal mask가 미래 token을 가리는지, CUDA에서도 mask device가 맞는지 |
| `tests/test_model.py` | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel forward shape, target이 있을 때 loss 반환, greedy generation shape |
| `tests/test_train.py` | batch loss, loader 평균 loss, checkpoint save/load, temperature/top-k generation shape, loss plot 호출, Windows 콘솔 안전 출력 |
| `tests/test_finetune.py` | 감성 분류용 데이터 split, sentiment dataset padding, GPT classification head 출력 shape, train/eval 함수 존재 |
| `tests/test_training_cli.py` | 실제 작은 데이터로 학습 CLI가 tokenizer/config/checkpoint/metrics를 쓰는지, benchmark CLI가 `tokens_per_sec`와 loss를 쓰는지, 스크립트를 파일 경로로 실행할 수 있는지 |

테스트가 통과한다는 뜻은 다음과 같습니다.

- 학습 루프를 돌릴 최소 기능이 연결되어 있습니다.
- tokenizer, dataset, model, loss, checkpoint, CLI가 서로 맞는 인터페이스로 동작합니다.
- 작은 smoke 학습과 benchmark 산출물이 만들어질 수 있습니다.

테스트가 보장하지 않는 것도 있습니다.

- 긴 학습 후 loss가 충분히 낮아진다는 보장은 아닙니다.
- 생성 문장이 자연스럽다는 보장은 아닙니다.
- hyperparameter가 최적이라는 뜻도 아닙니다.

즉, 테스트는 "출발해도 되는 상태인지"를 확인하고, 실제 품질은 `metrics.json`, benchmark, 생성 샘플을 보면서 따로 판단해야 합니다.

여기서 용어가 조금 헷갈릴 수 있습니다. `pytest tests/ -q`의 "테스트"는 코드가 올바르게 구현됐는지 확인하는 소프트웨어 테스트입니다. 반면 머신러닝에서 말하는 "test 성능"은 학습에 쓰지 않은 데이터에서 모델이 얼마나 잘 맞추는지를 보는 평가입니다. 이름은 비슷하지만 목적이 다릅니다.

## 3. 실제 학습에서 일어나는 일

`scripts/train_lm.py`를 실행하면 아래 순서로 실제 학습이 진행됩니다.

1. `data/nsmc_lm_train.txt`와 `data/nsmc_lm_val.txt`를 읽습니다.
2. 기본값은 train text로 byte-level BPE tokenizer를 새로 학습합니다. `--tokenizer-path`를 주면 기존 tokenizer를 불러옵니다.
3. tokenizer로 train/validation text를 token ID로 변환합니다.
4. `context_length` 길이의 input과 다음 token target을 만드는 DataLoader를 구성합니다.
5. GPTModel을 random initialization 상태로 만듭니다.
6. next-token prediction cross entropy loss로 모델을 학습합니다.
7. `runs/<run-name>/`에 tokenizer, config, checkpoint, metrics를 저장합니다.
8. 학습이 끝나면 콘솔에 최종 평가 요약을 따로 출력합니다.

여기서 "학습을 시킨다"는 말은 pretrained model을 가져와 이어 학습하는 것이 아니라, 이 프로젝트의 작은 GPT 모델을 처음부터 NSMC 텍스트로 next-token prediction 학습시키는 뜻입니다.

학습이 끝나면 대략 아래와 같은 블록이 출력됩니다.

```text
[Final evaluation]
run_dir: D:\jungleCamp\w13-14\gpt-lab\runs\smoke-test
device: cuda
epochs: 1
global_step: 137
train_loss (2 batches): 5.0477
val_loss (2 batches): 5.0259
elapsed_sec: 2.75
checkpoint: D:\jungleCamp\w13-14\gpt-lab\runs\smoke-test\final_checkpoint.pt
metrics: D:\jungleCamp\w13-14\gpt-lab\runs\smoke-test\metrics.json
plot: D:\jungleCamp\w13-14\gpt-lab\runs\smoke-test\training_curves.png
```

여기서 `train_loss`와 `val_loss`는 학습이 끝난 모델을 다시 평가 모드로 놓고 계산한 값입니다. `train_loss`는 train loader 일부 batch에서 계산한 평균 loss이고, `val_loss`는 validation loader 일부 batch에서 계산한 평균 loss입니다. 몇 batch를 평균낼지는 `--eval-iter`로 정합니다.

`training_curves.png`에는 두 개의 그래프가 들어갑니다.

| 그래프 | 의미 |
| --- | --- |
| Loss | step이 진행되면서 train/validation cross entropy loss가 어떻게 변하는지 보여줍니다. 낮아질수록 좋습니다. |
| Next-token Accuracy | 각 위치에서 모델이 가장 높은 점수를 준 token이 실제 다음 token과 일치한 비율입니다. 높아질수록 좋습니다. |

이 정확도는 신경망 분류 문제에서 보던 accuracy와 비슷하게 "맞힌 비율"이라는 점에서는 같습니다. 다만 현재 모델은 감정 label을 맞히는 것이 아니라 다음 token을 맞히고 있으므로, 여기서의 accuracy는 감성 분류 정확도가 아니라 token prediction accuracy입니다.

주의할 점은 이 프로젝트의 언어 모델 학습 데이터에는 별도 `test` split이 없다는 것입니다. 그래서 학습 후 바로 보여주는 값은 "test loss"가 아니라 "validation loss"입니다. 진짜 held-out test set을 따로 평가하고 싶다면 `nsmc_lm_test.txt` 같은 별도 파일을 만들고, 평가 스크립트가 그 파일을 읽도록 확장해야 합니다.

## 4. 감성 평가 성능은 어떤 개념인가

이 과제의 큰 흐름은 두 단계로 나눠서 보면 이해하기 쉽습니다.

| 단계 | 하는 일 | 주로 보는 값 |
| --- | --- | --- |
| 언어 모델 사전학습 | 문장을 보고 다음 token을 맞히도록 GPT를 학습 | `train_loss`, `val_loss` |
| 감성 분류 fine-tuning | 영화 리뷰가 긍정인지 부정인지 맞히도록 classifier를 학습 | `accuracy`, classification `loss` |

현재 `scripts/train_lm.py`가 돌리는 것은 첫 번째 단계인 언어 모델 사전학습입니다. 이 단계에서는 모델이 감정을 직접 맞히는 것이 아니라, 한국어 리뷰 텍스트의 token 패턴을 배우는 중입니다. 그래서 성능을 `accuracy`가 아니라 next-token prediction `loss`로 봅니다.

그래도 신경망 학습 그래프처럼 정확도 변화를 보고 싶어서, `scripts/train_lm.py`는 `training_curves.png`에 next-token accuracy도 함께 그립니다. 이 그래프는 "언어모델이 다음 token을 얼마나 맞히는지"를 보는 보조 지표입니다.

감성 분류 성능을 본다는 것은 두 번째 단계입니다. `src/finetune.py`의 `GPTForSequenceClassification`처럼 GPT backbone 위에 classification head를 붙이고, `data/nsmc_sentiment_train.jsonl`, `data/nsmc_sentiment_val.jsonl`, `data/nsmc_sentiment_test.jsonl`로 긍정/부정을 학습하고 평가합니다.

감성 분류에서 중요한 데이터 split은 다음과 같습니다.

| Split | 용도 |
| --- | --- |
| train | 모델 weight를 실제로 업데이트하는 데이터 |
| validation | 학습 중 설정을 비교하고 과적합을 감시하는 데이터 |
| test | 최종 성능을 한 번 확인하는 데이터. 학습이나 설정 선택에 쓰면 안 됩니다. |

감성 평가에서 주로 보는 지표는 다음과 같습니다.

| 지표 | 의미 | 해석 |
| --- | --- | --- |
| accuracy | 전체 리뷰 중 label을 맞힌 비율 | 높을수록 좋습니다. 예: 0.82는 82%를 맞혔다는 뜻입니다. |
| classification loss | 정답 label에 대한 cross entropy loss | 낮을수록 좋습니다. 모델의 확신까지 반영합니다. |
| train accuracy/loss | train split에서의 성능 | 너무 좋고 val/test가 나쁘면 과적합일 수 있습니다. |
| val accuracy/loss | validation split에서의 성능 | 학습 중 모델 설정을 고를 때 주로 봅니다. |
| test accuracy/loss | test split에서의 최종 성능 | 마지막 보고용 숫자로 보는 것이 좋습니다. |

예를 들어 감성 분류 결과가 아래처럼 나왔다고 하면:

```text
train loss 0.42, train acc 0.81
val loss 0.48, val acc 0.78
test loss 0.50, test acc 0.77
```

모델은 train에서 81%, validation에서 78%, test에서 77% 정도 맞히는 상태라고 해석할 수 있습니다. train 성능과 test 성능 차이가 너무 크면 모델이 train 데이터에 과하게 맞춰진 것일 수 있습니다.

정리하면, 지금 학습 CLI의 `val_loss`는 "한국어 리뷰 문장을 얼마나 잘 이어 맞히는가"를 보는 값이고, 감성 평가의 `accuracy`는 "리뷰의 긍정/부정을 얼마나 잘 맞히는가"를 보는 값입니다.

## 5. 학습 파라미터 바꾸는 법

`--preset smoke/dev/full`은 기본값 묶음이고, 필요한 값은 CLI 옵션으로 덮어쓸 수 있습니다. 실제 적용된 설정은 매 run마다 `runs/<run-name>/config.json`에 저장됩니다.

예를 들어 `smoke`를 기본으로 쓰되 모델과 데이터 크기를 조금 키우려면 다음처럼 실행합니다.

```powershell
python scripts/train_lm.py `
  --preset smoke `
  --run-name smoke-custom `
  --context-length 64 `
  --batch-size 4 `
  --emb-dim 64 `
  --n-heads 4 `
  --n-layers 2 `
  --learning-rate 3e-4 `
  --epochs 2 `
  --train-chars 20000 `
  --val-chars 4000
```

자주 바꾸는 옵션은 다음과 같습니다.

| 옵션 | 의미 | 조정 팁 |
| --- | --- | --- |
| `--vocab-size` | BPE vocabulary 크기이자 모델 출력 차원 | byte-level 기본 토큰 때문에 최소 260 이상이어야 합니다. 크면 표현력은 늘지만 모델도 커집니다. |
| `--context-length` | 한 번에 보는 token 길이 | 길수록 긴 문맥을 보지만 attention 메모리가 크게 늘어납니다. 메모리가 부족하면 먼저 줄여보세요. |
| `--emb-dim` | token hidden vector 차원 | 모델 표현력이 커지지만 느려지고 메모리를 더 씁니다. `--n-heads`로 나누어떨어져야 합니다. |
| `--n-heads` | attention head 수 | `emb_dim % n_heads == 0`이어야 합니다. 예: `emb_dim=64`, `n_heads=4`. |
| `--n-layers` | Transformer block 수 | 늘리면 모델이 깊어지고 느려집니다. smoke는 1, dev는 2부터 시작하기 좋습니다. |
| `--drop-rate` | dropout 비율 | 작은 데이터에서 과적합이 심하면 0.1 또는 0.2를 시도합니다. |
| `--batch-size` | 한 step에 처리하는 sample 수 | 크면 빠르고 안정적일 수 있지만 CUDA 메모리를 더 씁니다. |
| `--learning-rate` | AdamW 학습률 | 기본은 `3e-4`입니다. loss가 불안정하면 `1e-4`, 너무 느리면 `5e-4`를 시도합니다. |
| `--epochs` | 전체 train loader 반복 횟수 | 길게 학습할수록 시간이 늘고 checkpoint 품질이 좋아질 수 있습니다. |
| `--train-chars` | 학습에 사용할 train text 문자 수 | smoke/dev에서 빠르게 실험하려고 일부만 쓰는 옵션입니다. `full`은 기본값이 전체 사용입니다. |
| `--val-chars` | validation text 문자 수 | validation loss를 빠르게 보고 싶으면 줄입니다. |
| `--eval-freq` | 몇 step마다 train/val loss를 출력할지 | 너무 작으면 자주 평가해서 느려집니다. |
| `--eval-iter` | 평가 때 평균낼 batch 수 | 클수록 loss 추정이 안정적이지만 느려집니다. |
| `--ckpt-freq` | 몇 step마다 중간 checkpoint를 저장할지 | 0이면 중간 checkpoint를 저장하지 않고 마지막만 저장합니다. |
| `--progress-freq` | 몇 step마다 진행률, 경과 시간, 예상 남은 시간을 출력할지 | 지정하지 않으면 `--eval-freq`와 같은 주기로 출력합니다. 0을 주면 진행률 출력을 끌 수 있습니다. |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:0` | 기본 `auto`는 CUDA가 가능하면 CUDA를 사용합니다. |
| `--seed` | random seed | 실험 비교를 공정하게 하려면 같은 seed를 사용합니다. |
| `--start-context` | epoch 끝 샘플 생성 시작 문장 | 학습 중 출력되는 샘플의 시작 문장을 바꿉니다. |
| `--run-name` | run 저장 폴더 이름 | 실험마다 다르게 주면 결과가 덮어써지지 않습니다. |

상황별 예시는 아래처럼 잡으면 됩니다.

빠른 동작 확인:

```powershell
python scripts/train_lm.py --preset smoke --run-name smoke-001
```

CUDA 메모리가 부족할 때:

```powershell
python scripts/train_lm.py `
  --preset dev `
  --run-name dev-small-memory `
  --batch-size 4 `
  --context-length 32 `
  --emb-dim 64 `
  --n-layers 1
```

조금 더 강한 모델을 실험할 때:

```powershell
python scripts/train_lm.py `
  --preset dev `
  --run-name dev-larger `
  --vocab-size 1500 `
  --context-length 128 `
  --emb-dim 128 `
  --n-heads 4 `
  --n-layers 4 `
  --batch-size 8 `
  --epochs 3
```

같은 설정으로 재현 가능한 비교를 하고 싶을 때:

```powershell
python scripts/train_lm.py --preset dev --run-name dev-seed-42 --seed 42
python scripts/train_lm.py --preset dev --run-name dev-seed-123 --seed 123
```

주의할 점:

- `--emb-dim`은 `--n-heads`로 나누어떨어져야 합니다.
- `--vocab-size`나 모델 구조를 바꾸면 이전 checkpoint와 호환되지 않을 수 있습니다.
- `--context-length`, `--batch-size`, `--emb-dim`, `--n-layers`는 CUDA 메모리에 큰 영향을 줍니다.
- 실험 결과는 `metrics.json`과 `benchmark.json`을 함께 보면서 비교하는 게 좋습니다.

### 같은 BPE를 고정하고 모델 구조만 바꿔 학습하기

기본적으로 `scripts/train_lm.py`는 실행할 때마다 train text로 BPE tokenizer를 새로 학습합니다. 이미 학습해 둔 tokenizer를 고정하고 모델 구조만 바꿔 보고 싶으면 `--tokenizer-path`를 사용합니다.

이 흐름은 benchmark가 아니라 아키텍처 실험에 가깝습니다.

```text
benchmark_lm.py
= 이미 학습된 run을 불러와 학습 없이 loss와 처리 속도만 측정

train_lm.py + 고정 tokenizer
= 같은 BPE vocabulary/merge rule을 쓰면서 모델 구조만 바꿔 새로 학습
```

예를 들어 `full-001`의 BPE tokenizer는 그대로 쓰고, 모델만 더 작게 학습하려면 아래처럼 실행합니다.

```powershell
python scripts/train_lm.py `
  --preset full `
  --run-name full-emb64-layer2 `
  --tokenizer-path runs\full-001\tokenizer.json `
  --emb-dim 64 `
  --n-layers 2 `
  --batch-size 16
```

`--tokenizer-path`를 지정하면 다음처럼 동작합니다.

1. 기존 `tokenizer.json`을 `BPETokenizer.load()`로 불러옵니다.
2. 모델의 `vocab_size`는 불러온 tokenizer의 `vocab_size`에 맞춥니다.
3. 새 run 디렉터리에도 같은 tokenizer를 `tokenizer.json`으로 저장합니다.
4. `config.json`에는 사용한 tokenizer source가 기록됩니다.

주의할 점:

- `--tokenizer-path`를 쓰면 BPE vocab과 merge rule은 고정되고, `--emb-dim`, `--n-layers`, `--n-heads`, `--context-length`, `--batch-size` 같은 모델/학습 설정만 바꿔 비교하기 좋습니다.
- `--vocab-size`를 따로 주지 않으면 불러온 tokenizer의 vocab size를 자동으로 사용합니다.
- `--vocab-size`를 명시했는데 tokenizer의 vocab size와 다르면 에러가 납니다. tokenizer와 모델 출력 차원이 달라지면 학습과 추론이 맞지 않기 때문입니다.
- 기존 checkpoint weight는 모델 구조가 바뀌면 재사용할 수 없습니다. tokenizer만 재사용하고 모델은 새로 학습하는 흐름입니다.

이렇게 만든 run은 학습이 끝난 뒤 기존 도구로 비교할 수 있습니다.

```powershell
python scripts/benchmark_lm.py --run-dir runs\full-emb64-layer2
python scripts/infer_lm.py --run-dir runs\full-emb64-layer2 --prompt "이 영화는"
```

## 6. 학습 프리셋

학습 CLI에는 세 가지 프리셋이 있습니다.

| Preset | 목적 | 주로 쓰는 상황 |
| --- | --- | --- |
| `smoke` | 빠른 동작 확인 | 코드, CUDA, 출력 파일, checkpoint가 정상인지 확인 |
| `dev` | 작은 실험 | 시간을 더 쓰기 전에 loss가 내려가는지 확인 |
| `full` | 긴 학습 | 이후 실험에 쓸 더 나은 checkpoint 생성 |

모든 출력은 `runs/<run-name>/`에 저장됩니다.

| 파일 | 의미 |
| --- | --- |
| `tokenizer.json` | BPE vocabulary와 merge rule |
| `config.json` | 모델 및 학습 설정 |
| `final_checkpoint.pt` | 최종 model/optimizer state |
| `metrics.json` | loss, 실행 시간, token 수, run metadata |
| `checkpoint_step_*.pt` | `--ckpt-freq`가 0보다 클 때 저장되는 중간 checkpoint |

## 7. Smoke 학습

가장 먼저 이 명령을 실행해보면 됩니다.

```powershell
python scripts/train_lm.py --preset smoke --run-name smoke-test
```

학습 중에는 진행률이 아래처럼 출력됩니다.

```text
[Progress] step 10/137 (7.3%) | epoch 1/1 | elapsed 00:00:01 | eta 00:00:12 | 8.40 steps/s
Ep 1 (step 10): train loss 5.744, val loss 5.879
```

`elapsed`는 지금까지 걸린 시간이고, `eta`는 현재 step 속도를 기준으로 추정한 남은 시간입니다. 학습 초반에는 속도 측정이 안정되지 않아서 ETA가 조금 흔들릴 수 있습니다.

진행률을 더 자주 보고 싶으면 `--progress-freq`를 줄입니다.

```powershell
python scripts/train_lm.py --preset smoke --run-name smoke-progress --progress-freq 5
```

진행률을 너무 자주 찍고 싶지 않다면 값을 키웁니다.

```powershell
python scripts/train_lm.py --preset dev --run-name dev-progress --progress-freq 100
```

진행률 출력을 끄고 싶으면 0을 줍니다.

```powershell
python scripts/train_lm.py --preset dev --run-name dev-no-progress --progress-freq 0
```

CPU에서만 디버깅하고 싶다면 다음처럼 실행합니다.

```powershell
python scripts/train_lm.py --preset smoke --device cpu --run-name smoke-cpu
```

자주 바꿔볼 만한 옵션 예시는 아래와 같습니다.

```powershell
python scripts/train_lm.py `
  --preset smoke `
  --run-name smoke-custom `
  --context-length 64 `
  --batch-size 4 `
  --emb-dim 64 `
  --n-layers 2 `
  --train-chars 20000 `
  --val-chars 4000
```

## 8. 더 긴 학습

개발용 작은 학습은 다음처럼 돌립니다.

```powershell
python scripts/train_lm.py --preset dev --run-name dev-001
```

더 오래 돌리는 학습은 다음처럼 실행합니다.

```powershell
python scripts/train_lm.py --preset full --run-name full-001
```

CUDA 메모리가 부족하면 아래 값부터 줄여보세요.

```powershell
--batch-size 8 --context-length 64 --emb-dim 64 --n-layers 2
```

## 9. 벤치마크

이미 학습한 run을 벤치마크합니다.

```powershell
python scripts/benchmark_lm.py --run-dir runs\smoke-test
```

벤치마크 결과를 파일로 저장하려면 다음처럼 실행합니다.

```powershell
python scripts/benchmark_lm.py --run-dir runs\smoke-test --output-json runs\smoke-test\benchmark.json
```

학습하지 않은 프리셋 모델로 현재 머신의 대략적인 처리량만 보고 싶다면 다음 명령을 사용합니다.

```powershell
python scripts/benchmark_lm.py --preset smoke --num-batches 20 --warmup-batches 2
```

## 10. 숫자 읽는 법

`loss`는 next-token cross entropy입니다. 낮을수록 좋습니다. 아주 짧은 smoke run에서는 절대적인 수치보다 전체 파이프라인이 정상 동작하는지가 더 중요합니다.

`tokens_per_sec`는 선택한 `batch_size`와 `context_length`에서 측정한 forward-pass 처리량입니다. 로컬 설정끼리 비교할 때 사용하면 됩니다. 모델이 커지거나 context가 길어지거나 batch가 커지면 이 값도 달라집니다.

학습 직후 콘솔에 보이는 `[Final evaluation]`은 품질 확인용이고, `metrics.json`에는 같은 값이 JSON으로 저장됩니다. 여러 실험을 비교할 때는 각 run의 `metrics.json`에서 `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, `global_step`, `elapsed_sec`를 비교하면 됩니다. 그래프는 각 run의 `training_curves.png`를 열어보면 됩니다.

실험할 때는 아래처럼 비교하면 좋습니다.

```text
smoke: 코드 경로가 정상 동작하는지 확인
dev: loss가 내려가기 시작하는지 확인
full: checkpoint 품질을 신경 쓰는 학습
```

## 11. 자주 쓰는 명령

프로젝트 환경의 Python을 직접 지정해서 전체 테스트를 실행합니다.

```powershell
& 'C:\ProgramData\anaconda3\envs\gpt-lab\python.exe' -m pytest tests/ -q
```

셸에서 `conda activate`에 의존하지 않고 학습을 실행합니다.

```powershell
& 'C:\ProgramData\anaconda3\envs\gpt-lab\python.exe' scripts/train_lm.py --preset smoke --run-name smoke-direct
```

셸에서 `conda activate`에 의존하지 않고 벤치마크를 실행합니다.

```powershell
& 'C:\ProgramData\anaconda3\envs\gpt-lab\python.exe' scripts/benchmark_lm.py --run-dir runs\smoke-direct
```

## 12. full-001로 글자 추론 보기

학습된 `full-001` run을 사용해 입력 문장 다음에 올 후보 token과 이어쓰기 결과를 함께 볼 수 있습니다.

```powershell
& 'C:\ProgramData\anaconda3\envs\gpt-lab\python.exe' scripts/infer_lm.py --run-dir runs\full-001 --prompt "이 영화는"
```

출력의 `[Next token candidates]`는 현재 prompt 바로 다음에 올 가능성이 높은 token 후보와 확률입니다. byte-level BPE를 쓰기 때문에 후보가 항상 완성된 한 글자처럼 보이지는 않을 수 있습니다.

이어쓰기 길이와 샘플링을 바꾸고 싶으면 다음 옵션을 조절합니다.

```powershell
& 'C:\ProgramData\anaconda3\envs\gpt-lab\python.exe' scripts/infer_lm.py `
  --run-dir runs\full-001 `
  --prompt "배우 연기가" `
  --top-k 10 `
  --max-new-tokens 80 `
  --temperature 0.8 `
  --sample-top-k 40
```

재현 가능한 greedy 결과만 보고 싶으면 `--temperature 0 --sample-top-k 0`을 사용합니다.
