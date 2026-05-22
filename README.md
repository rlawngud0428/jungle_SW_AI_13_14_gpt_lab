# 과제 - mini GPT 구현

## 1. 개요

이 프로젝트는 PyTorch만 사용해 작은 GPT 계열 언어 모델을 직접 구현하는 학생용 템플릿입니다. 완성할 모델은 거대한 ChatGPT가 아니라, LLM의 핵심 component를 이해하기 위한 교육용 mini GPT입니다.

학생용 소스는 `TODO`와 `NotImplementedError`가 남아 있는 상태입니다. 처음 테스트를 실행하면 실패하는 것이 정상이며, 각 단계의 TODO를 구현하면서 해당 테스트 파일을 하나씩 통과시키면 됩니다.

참고 도서:

- 『밑바닥부터 만들면서 배우는 LLM』
- 교재 소스 코드: `https://github.com/rickiepark/llm-from-scratch`

---

## 2. 코드 관리

- 팀별 GitHub 저장소를 준비합니다.
- 개인 작업은 각자 branch를 만들어 진행합니다.
- `main` 또는 `master`에는 직접 push하지 않습니다.
- Pull Request로 팀원이 리뷰한 뒤 병합합니다.
- 병합 전에는 관련 테스트와 전체 테스트를 통과시킵니다.
- 데이터 파일, checkpoint, token, 비밀번호는 commit하지 않습니다.

---

## 3. 개발 환경

### 3.1 요구 사항

- Python 3.11
- 허용 라이브러리:
  - `torch`
  - `torch.nn`
  - `torch.utils.data`
  - `numpy`
  - `matplotlib`
  - `pytest`
- 금지:
  - Hugging Face `transformers`, `datasets`, `tokenizers`
  - `sentencepiece`
  - `spacy`
  - `nltk`
  - `lightning`
  - `accelerate`
  - 외부 pretrained model
  - 외부 tokenizer vocabulary

교재에서는 `tiktoken`을 사용하는 부분이 있지만, 이 과제에서는 tokenizer를 직접 구현해야 하므로 `tiktoken`도 사용하지 않습니다.

### 3.2 Colab에서 사용

1. 메뉴 **런타임 → 런타임 유형 변경**에서 **Python 3**, **GPU**를 선택합니다.
2. 브라우저에서 다음 주소를 엽니다. `USERNAME`, 저장소명, 브랜치명은 본인 환경에 맞게 바꿉니다. 브랜치가 `main`이면 URL에서 `master`를 `main`으로 바꿉니다.

```text
https://colab.research.google.com/github/USERNAME/gpt-lab/blob/master/gpt-lab.ipynb
```

3. 노트북의 **1. 환경설정** 코드 셀을 가장 먼저 실행합니다.
4. Colab 입력창에 GitHub 저장소 URL을 입력합니다.

```text
github.com/USERNAME/gpt-lab.git
```

5. Private 저장소라면 GitHub Personal Access Token을 입력합니다. 공개 저장소라면 Enter를 누르면 됩니다.
6. 그 다음 셀부터 데이터 로드 → BPE/모델 → 학습·평가 → 미세 조정 순서로 진행합니다.

### 3.3 로컬에서 실행

```bash
cd gpt-lab
conda create -n gpt-lab python=3.11 -y
conda activate gpt-lab
pip install -r requirements.txt
pytest tests/ -v
```

---

## 4. 프로젝트 소스

```text
gpt-lab/
├── README.md
├── REPORT.md
├── requirements.txt
├── download_data.py
├── gpt-lab.ipynb
├── data/
├── src/
│   ├── __init__.py
│   ├── bpe.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── model.py
│   ├── train.py
│   └── finetune.py
└── tests/
    ├── test_bpe.py
    ├── test_dataset.py
    ├── test_attention.py
    ├── test_model.py
    ├── test_train.py
    └── test_finetune.py
```

| 파일 | 역할 |
| --- | --- |
| `download_data.py` | NSMC 원본 데이터를 내려받고 과제용 파일 생성 |
| `gpt-lab.ipynb` | Colab/로컬 실행 순서 안내 노트북 |
| `src/bpe.py` | UTF-8 byte-level BPE tokenizer |
| `src/dataset.py` | GPT 사전 학습용 Dataset과 DataLoader |
| `src/embeddings.py` | token embedding + position embedding |
| `src/attention.py` | causal multi-head self-attention |
| `src/model.py` | LayerNorm, GELU, FeedForward, TransformerBlock, GPTModel |
| `src/train.py` | loss 계산, checkpoint, generation, pretraining loop |
| `src/finetune.py` | NSMC 감성 분류 Dataset과 classifier |

---

## 5. 데이터

기본 데이터는 **NAVER Sentiment Movie Corpus(NSMC)** 입니다.

- 원본 저장소: `https://github.com/e9t/nsmc`
- 라이선스: CC0 1.0
- 원본 파일:
  - `ratings_train.txt`
  - `ratings_test.txt`
- 컬럼:
  - `id`
  - `document`: 영화 리뷰 문장
  - `label`: 부정 `0`, 긍정 `1`

데이터 준비:

```bash
python download_data.py
```

생성되는 파일:

| 파일 | 용도 |
| --- | --- |
| `data/nsmc_lm_train.txt` | 사전 학습 train 텍스트 |
| `data/nsmc_lm_val.txt` | 사전 학습 validation 텍스트 |
| `data/nsmc_sentiment_train.jsonl` | 감성 분류 train 데이터 |
| `data/nsmc_sentiment_val.jsonl` | 감성 분류 validation 데이터 |
| `data/nsmc_sentiment_test.jsonl` | 감성 분류 test 데이터 |

데이터 파일은 `.gitignore`에 포함되어 있으므로 GitHub에 commit하지 않습니다.

---

## 6. 단계별 구현 안내

### 6.1 구현 전 확인 사항

- 작은 데이터로 먼저 실행해서 코드가 동작하는지 확인합니다.
- 각 단계의 TODO를 구현한 뒤 해당 테스트 파일만 먼저 실행합니다.
- 단계별 테스트를 모두 통과한 뒤 마지막에 전체 테스트를 실행합니다.
- Colab 런타임이 끊길 수 있으므로 오래 걸리는 학습 결과와 checkpoint는 저장합니다.
- 데이터 파일, checkpoint, token, 비밀번호는 GitHub에 commit하지 않습니다.

### 6.2 노트북 진행 방법

`gpt-lab.ipynb`는 개발 순서와 같은 순서로 구성되어 있습니다.

1. 환경설정 셀 실행
2. 데이터 준비 셀 실행
3. `src/bpe.py` TODO 구현
4. `pytest tests/test_bpe.py -v` 셀 실행
5. 통과하면 다음 단계로 이동
6. 마지막에 `pytest tests/ -v` 실행

노트북에서 어떤 셀이 `NotImplementedError`를 출력하면 아직 해당 단계 TODO가 남아 있다는 뜻입니다.

### 6.3 개발 순서 요약

| 순서 | 구현 대상 | 파일 | 테스트 |
| --- | --- | --- | --- |
| 1 | BPE tokenizer | `src/bpe.py` | `pytest tests/test_bpe.py -v` |
| 2 | Dataset / InputEmbedding | `src/dataset.py`, `src/embeddings.py` | `pytest tests/test_dataset.py -v` |
| 3 | MultiHeadAttention | `src/attention.py` | `pytest tests/test_attention.py -v` |
| 4 | GPT 모델 구성 요소 | `src/model.py` | `pytest tests/test_model.py -v` |
| 5 | 사전 학습 유틸리티 | `src/train.py` | `pytest tests/test_train.py -v` |
| 6 | 감성 분류 미세 조정 | `src/finetune.py` | `pytest tests/test_finetune.py -v` |
| 7 | 전체 테스트 | 전체 | `pytest tests/ -v` |

처음부터 `pytest tests/ -v`만 실행하면 어디가 문제인지 찾기 어렵습니다. 현재 구현 중인 단계의 테스트부터 실행하세요.

---

## 7. 추가 미션

필수 구현을 마친 뒤 선택적으로 진행합니다.

### 7.1 사전 학습 성능 향상

- 학습률 warmup
- cosine decay
- gradient clipping
- weight decay 실험

참고:

- 교재 부록 D
- 교재 소스의 `appendix-D.ipynb`

### 7.2 하이퍼파라미터 탐색

- `batch_size`: 2, 4, 8, 16
- `drop_rate`: 0.0, 0.1, 0.2
- `learning_rate`: 1e-4, 3e-4, 5e-4
- `context_length`: 64, 128
- `n_layers`: 1, 2, 4
- `emb_dim`: 64, 128, 192

### 7.3 더 나은 감성 분류

- backbone 일부 freeze
- classifier learning rate와 backbone learning rate 분리
- class imbalance 확인
- validation loss가 가장 낮은 checkpoint 선택

---

## 8. 제출물

- 동작하는 `src/` 소스 코드
- 실행 가능한 `gpt-lab.ipynb`
- `REPORT.md`