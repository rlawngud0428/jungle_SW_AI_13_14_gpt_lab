# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json

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
        # raise NotImplementedError("_init_special_tokens를 구현하세요.")
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

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

    def _apply_merge(self, ids, left_id, right_id, new_id):
        result = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == left_id and ids[i + 1] == right_id:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1

        return result

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")
        self._init_special_tokens()

        ids = [byte + BYTE_OFFSET for byte in corpus.encode("utf-8")]
        next_id = BYTE_OFFSET + NUM_BYTES

        while len(self.id_to_token) < self.vocab_size:
            pair_counts = {}

            for pair in zip(ids, ids[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda item: item[1])

            if best_count < 2:
                break

            left_id, right_id = best_pair
            new_token = self.id_to_token[left_id] + self.id_to_token[right_id]
            new_id = next_id

            self.id_to_token[new_id] = new_token
            self.token_to_id[new_token] = new_id
            self.merges.append((left_id, right_id, new_id))

            ids = self._apply_merge(ids, left_id, right_id, new_id)

            next_id += 1

        return self

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        # raise NotImplementedError("BPETokenizer.save를 구현하세요.")
        path = Path(path)

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": {},
            "merges": self.merges,
        }

        for token_id, token in self.id_to_token.items():
            if isinstance(token, bytes):
                data["id_to_token"][str(token_id)] = {
                    "type": "bytes",
                    "value": list(token),
                }
            else:
                data["id_to_token"][str(token_id)] = {
                    "type": "str",
                    "value": token,
                }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        # raise NotImplementedError("BPETokenizer.load를 구현하세요.")
        path = Path(path)

        data = json.loads(path.read_text(encoding="utf-8"))
        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}

        for token_id_str, item in data["id_to_token"].items():
            token_id = int(token_id_str)

            if item["type"] == "bytes":
                token = bytes(item["value"])
            else:
                token = item["value"]

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        self.merges = [tuple(merge) for merge in data["merges"]]

        return self

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # raise NotImplementedError("BPETokenizer.encode를 구현하세요.")
        if not self.id_to_token:
            self._init_special_tokens()

        ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]

        # 나중에 train/load가 채운 self.merges를 순서대로 적용
        for left_id, right_id, new_id in self.merges:
            ids = self._apply_merge(ids, left_id, right_id, new_id)

        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]

        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        # raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
        if not self.id_to_token:
            self._init_special_tokens()

        byte_chunks = []

        for token_id in ids:
            token = self.id_to_token.get(token_id)

            if token is None:
                if skip_special:
                    continue
                token = self.id_to_token[self.get_unk_id()]

            if isinstance(token, str):
                if skip_special and token in SPECIAL_TOKENS:
                    continue
                continue

            byte_chunks.append(token)

        return b"".join(byte_chunks).decode("utf-8", errors="replace")
