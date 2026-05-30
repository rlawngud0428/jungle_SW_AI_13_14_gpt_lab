# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
from collections import Counter


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
        2. byte 0-255를 ID 4-259에 bytes([byte_value]) 형태로 등록합니다.
        """
        # raise NotImplementedError("_init_special_tokens를 구현하세요.")

        # 특수 토큰 4개를 고정 ID 0~3번에 등록
        for i, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[i] = token
            self.token_to_id[token] = i

        # byte 0~255를 id 4~259에 bytes([byte_value]) 형태로 등록
        for byte in range(256):
            token_id = byte + BYTE_OFFSET
            token = bytes([byte])
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
        # raise NotImplementedError("BPETokenizer.train을 구현하세요.")

        # self.id_to_token, token_to_id, merges를 초기화 해주기
        self._init_special_tokens()

        # corpus.encode를 통해서 byte를 얻고, 거기에 처음 special token offset 더해서 지금 id 구하기
        ids = [byte + BYTE_OFFSET for byte in corpus.encode("utf-8")]
        
        # 언제까지? 하나가 될 때까지? -> ai에게 물어보았다!
        # while 일단 vocab가 다 차면 그만둬야함. 그리고 단어가 1개 남게되면 그만둬야함. 그리고 최빈수가 2 이상이여야 함. 그래야 vocab에 추가하는 의미가 있지
        while ((len(self.id_to_token) < self.vocab_size) and (len(ids) > 1)):
            # 일단 2개씩 묶어보자
            temp_list = []
            for i in range(len(ids)-1):
                temp_list.append((ids[i], ids[i+1]))

            # 가장 자주 나온 놈을 찾는다.            
            # max함수 찾아볼것
            counter = Counter(temp_list)
            max_count = max(counter.values())
            if (max_count <= 1):
                break
            
            # 가장 많이 나온 페어들
            most_common = [key for key, value in counter.items() if value == max_count]
            # 새로 저장할 ids
            new_ids = []
            # 새로 추가할 페어에 대한 idx (id_to_token)
            idx = len(self.id_to_token)

            # ids를 돌면서 pair를 발견하면 new_ids에 저장 / 아니면 그냥 저장
            i = 0
            while i < len(ids):
                if (i == len(ids)-1):
                    new_ids.append(ids[i])
                    break

                temp = (ids[i], ids[i+1])
                if temp == most_common[0]:
                    new_ids.append(idx)
                    i += 2
                    continue
                new_ids.append(ids[i])
                i += 1

            # 반복문 돌면서 pair를 다 찾아서 new_ids에 만들었으니
            # merges, itt, tti에 등록하고, ids도 new_ids로 바꾸기
            self.merges.append((most_common[0], idx))
            self.id_to_token[idx] = most_common[0]
            self.token_to_id[most_common[0]] = idx

            ids = new_ids

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        import json

        def encode_value(value):
            if isinstance(value, bytes):
                return {"type": "bytes", "value": list(value)}
            if isinstance(value, tuple):
                return {"type": "tuple", "value": [encode_value(item) for item in value]}
            if isinstance(value, list):
                return {"type": "list", "value": [encode_value(item) for item in value]}
            if isinstance(value, str):
                return {"type": "str", "value": value}
            if isinstance(value, int):
                return {"type": "int", "value": value}
            raise TypeError(f"저장할 수 없는 타입입니다: {type(value).__name__}")

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": [
                {"id": token_id, "token": encode_value(token)}
                for token_id, token in self.id_to_token.items()
            ],
            "token_to_id": [
                {"token": encode_value(token), "id": token_id}
                for token, token_id in self.token_to_id.items()
            ],
            "merges": encode_value(self.merges),
        }

        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        import json

        def decode_value(value):
            value_type = value["type"]
            raw_value = value["value"]
            if value_type == "bytes":
                return bytes(raw_value)
            if value_type == "tuple":
                return tuple(decode_value(item) for item in raw_value)
            if value_type == "list":
                return [decode_value(item) for item in raw_value]
            if value_type == "str":
                return raw_value
            if value_type == "int":
                return raw_value
            raise ValueError(f"알 수 없는 타입입니다: {value_type}")

        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {
            item["id"]: decode_value(item["token"])
            for item in data["id_to_token"]
        }
        self.token_to_id = {
            decode_value(item["token"]): item["id"]
            for item in data["token_to_id"]
        }
        self.merges = decode_value(data["merges"])

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # raise NotImplementedError("BPETokenizer.encode를 구현하세요.")
        ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]

        merges = self.merges
        for merge in merges:
            pair, id = merge
            new_ids = []
            i = 0
            while (i < len(ids)):
                if (i == len(ids)-1):
                    new_ids.append(ids[i])
                    break
                
                temp = (ids[i], ids[i+1])
                if (temp == pair):
                    new_ids.append(id)
                    i += 2
                    continue

                new_ids.append(ids[i])
                i += 1

            ids = new_ids
        
        if (add_bos_eos == True):
            ids.insert(0, 2)
            ids.append(3)
        
        return ids

    def _extend(self, id):
        if (id >= 260):
            left, right = self.id_to_token[id]
        else:
            return [id - BYTE_OFFSET]
        left = self._extend(self, left)
        right = self._extend(self, right)
        return left + right

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        # raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
        new_ids = []

        for id in ids:
            if id <= 4:
                continue
            new_ids.extend(self._extend(id))

        return bytes(new_ids).decode("utf-8")
