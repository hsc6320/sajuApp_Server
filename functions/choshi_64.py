import json
import os
import random
from typing import Any, Dict, Optional

# 현재 파일 위치 기준으로 JSON 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CURRENT_DIR, "choshi_yi_lin_64_fullDetailed.json")

class GuaStore:
    def __init__(self):
        self._by_number: Dict[int, Dict[str, Any]] = {}
        self._loaded = False

    def init(self):
        if self._loaded:
            return
        if not os.path.exists(JSON_PATH):
            raise FileNotFoundError(f"64괘 JSON 파일이 없습니다: {JSON_PATH}")

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("초씨역림_64괘") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("64괘 JSON 구조가 리스트여야 합니다.")

        for obj in items:
            no = int(obj.get("번호"))
            self._by_number[no] = obj

        self._loaded = True
        print(f"✅ 64괘 로드 완료: {JSON_PATH} ({len(self._by_number)}개)")

    def get(self, number: int) -> Optional[Dict[str, Any]]:
        if not self._loaded:
            self.init()
        return self._by_number.get(number)

    def pick_random(self, seed: int | None = None) -> tuple[int, dict]:
        """본괘 1개 랜덤"""
        import random
        if not self._loaded:
            self.init()
        if not self._by_number:
            raise ValueError("64괘 데이터가 비어 있습니다.")
        rng = random.Random(seed)
        n = rng.choice(list(self._by_number.keys()))
        return n, self._by_number[n]

    def pick_two_random(self, seed: int | None = None) -> tuple[tuple[int, dict], tuple[int, dict]]:
        """본괘/변괘 2개를 서로 다르게 랜덤 선택"""
        import random
        if not self._loaded:
            self.init()
        keys = list(self._by_number.keys())
        if len(keys) < 2:
            raise ValueError("64괘 데이터가 2개 미만이라 본괘/변괘를 선택할 수 없습니다.")
        rng = random.Random(seed)
        ben = rng.choice(keys)
        bian = rng.choice([k for k in keys if k != ben])
        return (ben, self._by_number[ben]), (bian, self._by_number[bian])

GUA = GuaStore()
