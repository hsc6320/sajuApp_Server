# ===== JSON 대화저장 + 메타추출 (OpenAI + LangChain 0.2.x) =====
import json, time, uuid, os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI






#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#JSON_PATH = os.path.join(CURRENT_DIR, "converted.json")
# ---- 설정 ----

_OPENAI_MODEL = os.environ.get("EXTRACT_MODEL", "gpt-4o-mini")  # 저비용 추출용 모델
_TEMPERATURE  = float(os.environ.get("EXTRACT_TEMPERATURE", "0.0"))
# ---- LLM 체인 (메타 추출용) ----
# PromptTemplate: 사용자의 입력 문장을 받아 JSON 형태의 메타데이터로 변환하도록 LLM에 지시
# 사용 목적: 사용자 질문 한 문장에서
#  - msg_keywords / target_date / time / kind / notes 를 JSON으로 추출
#  - conversations.json에 함께 저장
_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "사용자 발화를 분석해 아래 스키마의 JSON만 출력하라(설명/주석/코드블록 금지).\n"
        "{{\"msg_keywords\": [\"string\"], "
        "\"target_date\": null, "
        "\"time\": null, "
        "\"kind\": null, "
        "\"notes\": \"string\"}}"
    )),
    ("user", "{text}")
])


def get_extract_chain():
    """
    OpenAI API 키가 설정되어 있을 때만 체인을 빌드합니다.
    @lru_cache를 사용하여 첫 호출 시에만 체인을 생성합니다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("[META] OPENAI_API_KEY not set — meta extraction will be skipped")
        return None

    llm = ChatOpenAI(
        model=os.environ.get("EXTRACT_MODEL", "gpt-4o-mini"),
        temperature=float(os.environ.get("EXTRACT_TEMPERATURE", "0.0")),
        max_tokens=500,
        timeout=45,
        max_retries=2,
        openai_api_key=api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    return _EXTRACT_PROMPT | llm


_CODEFENCE_RE = re.compile(r"```[a-zA-Z]*\s*([\s\S]*?)```", re.MULTILINE)

def _to_text(raw: Any) -> str:
    """LLM 응답을 문자열로 정규화"""
    if raw is None:
        return ""
    # langchain invoke/run의 결과가 객체이거나 dict 형태일 수 있음
    if isinstance(raw, dict) and "text" in raw:
        return raw["text"]
    if hasattr(raw, "content") and isinstance(raw.content, str):
        return raw.content
    if isinstance(raw, str):
        return raw
    return str(raw)

def _extract_json_block(raw_text: str) -> str:
    """
    1) ```json ... ``` 코드펜스 내부만 우선 추출
    2) 없으면 문자열 전체에서 첫 '{' ~ 마지막 '}' 구간만 슬라이스
    """
    if not raw_text:
        return "{}"
    m = _CODEFENCE_RE.search(raw_text)
    if m:
        return m.group(1).strip()
    s, e = raw_text.find("{"), raw_text.rfind("}")
    return raw_text[s:e+1] if (s != -1 and e != -1) else raw_text

def _safe_json_loads(maybe_json: str) -> Dict[str, Any]:
    try:
        return json.loads(maybe_json)
    except Exception:
        return {}





#def _db_save(db: Dict[str, Any]) -> None:

#def ensure_session(session_id: Optional[str], title: str = "사주 대화") -> str:




# ====== 대화 검색 유틸 ======
from typing import Tuple



def format_search_results(rows: list[dict]) -> list[dict]:
    """
    클라이언트로 내려주기 좋은 축약 필드만 추려서 반환.
    필요시 그대로 수정하세요.
    """
    out = []
    for r in rows:
        out.append({
            "session_id": r.get("_session_id"),
            "ts": r.get("ts"),
            "date": r.get("date"),
            "time": r.get("time"),
            "role": r.get("role"),
            "mode": r.get("mode"),
            "text": r.get("text"),
            "notes": r.get("notes"),
            "msg_keywords": r.get("msg_keywords", []),
            "target_date": r.get("target_date"),
            "event_time": r.get("event_time"),
            "kind": r.get("kind"),
        })
    return out


_gcs_client = None

def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client
