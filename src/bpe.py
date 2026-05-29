# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for token, token_id in SPECIAL_IDS.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()

        token_ids = [BYTE_OFFSET + byte for byte in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size and len(token_ids) >= 2:
            pair_counts = {}
            for left, right in zip(token_ids, token_ids[1:]):
                pair = (left, right)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            best_pair, best_count = max(pair_counts.items(), key=lambda item: item[1])
            if best_count < 2:
                break

            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            merged_ids = []
            idx = 0
            while idx < len(token_ids):
                if idx < len(token_ids) - 1 and (token_ids[idx], token_ids[idx + 1]) == best_pair:
                    merged_ids.append(new_id)
                    idx += 2
                else:
                    merged_ids.append(token_ids[idx])
                    idx += 1
            token_ids = merged_ids

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        def serialize_token(token):
            if isinstance(token, bytes):
                return {"type": "bytes", "value": list(token)}
            if isinstance(token, tuple):
                return {"type": "merge", "value": list(token)}
            if isinstance(token, str):
                return {"type": "special_tokens", "value": token}
            raise TypeError(f"Unsupported token type: {type(token)}")

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": {
                str(token_id): serialize_token(token)
                for token_id, token in sorted(self.id_to_token.items())
            },
            "merges": [list(pair) for pair in self.merges],
        }

        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        def deserialize_token(token_data):
            token_type = token_data["type"]
            value = token_data["value"]

            if token_type == "bytes":
                return bytes(value)
            if token_type == "merge":
                return tuple(value)
            if token_type in ("special", "special_tokens"):
                return value
            raise ValueError(f"Unsupported token type: {token_type}")

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}

        for token_id, token_data in data["id_to_token"].items():
            token_id = int(token_id)
            token = deserialize_token(token_data)
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        self.merges = [tuple(pair) for pair in data["merges"]]

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        token_ids = [BYTE_OFFSET + byte for byte in text.encode("utf-8")]

        for pair in self.merges:
            pair = tuple(pair)
            new_id = self.token_to_id[pair]
            merged_ids = []
            idx = 0

            while idx < len(token_ids):
                if idx < len(token_ids) - 1 and (token_ids[idx], token_ids[idx + 1]) == pair:
                    merged_ids.append(new_id)
                    idx += 2
                else:
                    merged_ids.append(token_ids[idx])
                    idx += 1
            token_ids = merged_ids

        if add_bos_eos:
            token_ids = [self.get_bos_id()] + token_ids + [self.get_eos_id()]

        return token_ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        byte_values = []

        def append_token_bytes(token_id):
            token = self.id_to_token[token_id]

            if isinstance(token, bytes):
                byte_values.extend(token)
                return

            if isinstance(token, tuple):
                left_id, right_id = token
                append_token_bytes(left_id)
                append_token_bytes(right_id)
                return

            if isinstance(token, str):
                if skip_special:
                    return
                byte_values.extend(token.encode("utf-8"))
                return

            raise TypeError(f"Unsupported token type: {type(token)}")

        for token_id in ids:
            append_token_bytes(token_id)

        return bytes(byte_values).decode("utf-8")
