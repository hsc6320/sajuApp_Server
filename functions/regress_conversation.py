# regress_chat.py 내에 넣을 코드 (드롭인 교체용)
# -------------------------------------------------
# 목적 요약:
# 1) 회귀 판정은 LLM으로만(_llm_detect_regression), 키워드 기반이 아님.
# 2) 회귀=True면, 반드시 conversations.json을 열어 과거 턴을 직접 스캔해
#    (a) LLM에 붙일 "맥락 라인업"을 구성하고,
#    (b) '언제/날짜' 류 후속질의면 날짜 FACT를 복원하여 프롬프트에 주입.
# 3) 반환 debug에는 LLM 판정, JSON 검색 요약, FACT 주입 정보까지 모두 담아 로깅/저장 가능.
# -------------------------------------------------
from functools import lru_cache
import time
from langchain_core.prompts import ChatPromptTemplate
#from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import os, re, json
from datetime import date, datetime, timedelta
from langchain_openai import ChatOpenAI

from conv_store import _CUR_USER_ID, _db_load, _db_save, _is_gs_path, _max_turns, _parse_gs_path, _resolve_store_path_for_user, _trim_session_turns, get_current_user_id, get_current_app_uid, make_user_key, set_current_user_context, user_from_payload
try:
    from zoneinfo import ZoneInfo  # Py3.9+
except Exception:
    ZoneInfo = None  # 없으면 KST 변환 생략

KST = ZoneInfo("Asia/Seoul") if ZoneInfo else None
os.environ.setdefault("TZ", "Asia/Seoul")
try: time.tzset()
except: pass

# (A) LLM으로 회귀 의도 판단 + 주제 추정
_REG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
당신은 '대화 회귀(continuation)' 감지기입니다.

규칙(절대):
- has_history=false 또는 history_turns==0 이면 is_regression는 반드시 false.

정의:
- '회귀'란 사용자의 현재 발화가 과거 대화의 구체적 주제/사건/인물/문서/결과를 이어받아 후속 질의/지시를 하는 경우.

판정 가이드:
- True 조건(예시): "다시/이어서/아까/전에 말한" 등 명시적 지시가 있거나, summary에 나온 고유명/특정 사건을 구체적으로 이어감.
- False 조건(예시): '사주/운세/올해/내년/2025' 같은 일반 단어만 비슷하거나, 전혀 새로운 주제.

JSON 한 줄로만 응답:
{{\"is_regression\": <true|false>, \"confidence\": 0.0~1.0, \"topic_keywords\": [], \"explicit_markers\": [], \"reasons\": "짧은 이유"}}
"""),
    ("user", "has_history={has_history}\nhistory_turns={history_turns}\n\n## summary\n{summary}\n\n## utterance\n{question}\n\nJSON만 출력.")
])


# ───────── GCS 유틸 ─────────
from google.cloud import storage


# def _parse_gs_path(gs_path: str) -> tuple[str, str]:
#     # gs://bucket/name/with/slashes.json -> (bucket, name/with/slashes.json)
#     no_scheme = gs_path[len("gs://"):]
#     parts = no_scheme.split("/", 1)
#     bucket = parts[0]
#     name = parts[1] if len(parts) > 1 else ""
#     #print(f"_parse_gs_path({str}, name : {name})")
#     return bucket, name




# ------- 유틸: 안전 자카드 (키워드 겹침 점수) --------------------------------
def _jaccard_safe(a: List[str] | set, b: List[str] | set) -> float:
    A = set([x for x in (a or []) if x])
    B = set([x for x in (b or []) if x])
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def _get_regression_chain():
    """
    회귀 판정 체인을 지연 생성 + 1회 캐시.
    - 전역 초기화 대신 호출 시 초기화 → import 순서/헬스체크 안전
    - OPENAI_API_KEY 없으면 None 반환(폴백 처리)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[REG] OPENAI_API_KEY not set — regression detection disabled")
        return None

    model = os.environ.get("REG_MODEL", "gpt-4o-mini")
    temp  = float(os.environ.get("REG_TEMPERATURE", "0.0"))
    #print(f"[CHAIN] 회귀판정 체인 초기화: model={model}, temp={temp}")

    llm = ChatOpenAI(
        model=model,
        temperature=temp,
        max_tokens=350,
        timeout=45,
        max_retries=2,
        openai_api_key=api_key,  # ← 명시 전달(서버리스 환경에서 안전)
        model_kwargs={"response_format": {"type": "json_object"}},  # JSON only
    )
    return _REG_PROMPT | llm

_REG_CHAIN = _get_regression_chain()


# ------- 유틸: 세션 ID 추정 ---------------------------------------------------
def _guess_session_id(db: dict) -> Optional[str]:
    """명시 세션이 없다면 가장 먼저 보이는 세션을 고름(서비스 정책에 맞게 수정 가능)"""
    sessions = db.get("sessions") or {}
    for sid in sessions.keys():
        return sid
    return None

# ------- 유틸: 한 턴의 키워드/종류/텍스트 안전 추출 ---------------------------
def _turn_kws_kind_text(turn: dict) -> Tuple[List[str], Optional[str], str]:
    kws = turn.get("msg_keywords")
    if not kws:
        kws = (turn.get("meta") or {}).get("msg_keywords")
    if kws is None:
        kws = []
    kind = turn.get("kind") or (turn.get("meta") or {}).get("kind")
    text = turn.get("text") or ""
    return (list(kws), kind, text)

# ------- 유틸: 날짜 파싱 (절대/상대) ------------------------------------------
DATE_PATTERNS = [
    re.compile(r"(\d{4})[-./](\d{1,2})[-./](\d{1,2})"),
    re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
    re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
]
RELATIVE = {"오늘":0, "내일":1, "모레":2, "글피":3, "어제":-1, "그제":-2}

def _norm_date(y:int, m:int, d:int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"

def _parse_date_from_text(text: str, base_date_str: str) -> Optional[str]:
    """
    텍스트에서 날짜(절대/상대)를 추출한다.
    상대표현은 '그 턴의 날짜(base_date_str=YYYY-MM-DD)' 기준으로 환산.
    """
    # base_date 파싱
    try:
        base_dt = datetime.strptime(base_date_str, "%Y-%m-%d")
        if KST: base_dt = base_dt.replace(tzinfo=KST)
    except Exception:
        base_dt = datetime.now(KST) if KST else datetime.now()

    t = text or ""
    # 1) 절대 날짜
    for p in DATE_PATTERNS:
        m = p.search(t)
        if m:
            parts = [int(x) for x in m.groups()]
            if len(parts) == 3 and parts[0] >= 100:
                return _norm_date(parts[0], parts[1], parts[2])
            if len(parts) == 3 and parts[0] < 100:  # 2자리 연도 방어
                return _norm_date(parts[0] + 2000, parts[1], parts[2])
            if len(parts) == 2:
                y = base_dt.year
                return _norm_date(y, parts[0], parts[1])

    # 2) 상대 표현
    for token, delta in RELATIVE.items():
        if token in t:
            dt = (base_dt + timedelta(days=delta)).date()
            return _norm_date(dt.year, dt.month, dt.day)

    return None

# ------- 핵심: JSON에서 맥락 선택 --------------------------------------------
def _select_context_from_json(
    merged_kws: List[str],
    target_kind: Optional[str],
    *,
    limit_pick: int = 8,
    session_id: Optional[str] = None,
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    conversations.json에서 현재 주제에 '가까운' 과거 턴을 점수화하여 상위 N개 선택.
    - 점수: 키워드 자카드 + kind 보너스 + 약한 최신성 보정(최신턴 가중)
    - 반환 rows: LLM 프롬프트에 넣기 쉬운 요약 dict 목록
    - 반환 dbg : 로깅용 통계
    """
    db = _db_load()
    sessions = db.get("sessions") or {}
    sid = session_id or _guess_session_id(db)
    sess = sessions.get(sid) or {}
    turns = list(sess.get("turns") or [])
    total = len(turns)
    print(f"[JSON_SCAN] sid={session_id} total_turns={len(turns)}")
    
    now_set = set([x for x in (merged_kws or []) if x])
    rows_scored: List[Tuple[float, dict]] = []

    # 최신성 가중: 뒤로 갈수록(더 최근) 작은 페널티
    # (지수감쇠 같은 고급 기법도 가능하지만 간단히 0~0.1 범위로만 보정)
    for idx, t in enumerate(reversed(turns), start=1):
        # 최근 턴일수록 rec_boost가 커짐
        rec_boost = min(0.10, 0.10 * (1.0 - (idx / max(1, total)))) if total > 0 else 0.0

        prev_kws, prev_kind, _text = _turn_kws_kind_text(t)
        j = _jaccard_safe(now_set, prev_kws)
        score = j + rec_boost
        if target_kind and prev_kind and prev_kind == target_kind:
            score += 0.15

        # 너무 낮으면 제외(노이즈 컷)
        if score <= 0.0:
            continue

        rows_scored.append((score, t))

    rows_scored.sort(key=lambda x: x[0], reverse=True)
    picked_turns = [t for (s, t) in rows_scored[:limit_pick]]

    # 프롬프트용 요약 줄 만들기
    rows_fmt = []
    for t in picked_turns:
        rows_fmt.append({
            "date": t.get("date", ""),
            "time": t.get("time", ""),
            "role": t.get("role", ""),
            "text": t.get("text", "")
        })

    dbg = {
        "searched_total": total,
        "now_keywords": list(now_set),
        "now_kind": target_kind,
        "scored": len(rows_scored),
        "filtered_by_min_sim": max(0, len(rows_scored) - len(picked_turns)),
        "picked": len(picked_turns),
    }
    print(f"[JSON_SCAN] scored={len(rows_scored)} picked={len(picked_turns)}")
    return rows_fmt, dbg

# ------- 핵심: JSON에서 '여행 날짜' FACT 복원 ---------------------------------
def _find_last_trip_date_from_json(session_id: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    최근 '여행' 관련 사용자 턴에서 날짜를 복원.
    우선순위:
      1) turn.target_date / turn.meta.target_date
      2) turn.text의 절대 날짜
      3) turn.text의 상대표현을 turn.date 기준으로 환산
    """
    db = _db_load()
    sessions = db.get("sessions") or {}
    sid = session_id or _guess_session_id(db)
    sess = sessions.get(sid) or {}
    turns = list(sess.get("turns") or [])
    print(f"[DATE_LOOKUP] sid={session_id} scan_turns={len(turns)}")
    
    for t in reversed(turns):
        if t.get("role") != "user":
            continue
        kws = t.get("msg_keywords") or (t.get("meta") or {}).get("msg_keywords") or []
        if not any(k in ("여행", "여행운") for k in kws):
            continue

        td = t.get("target_date") or (t.get("meta") or {}).get("target_date")
        if isinstance(td, str) and td.strip():
            return td.strip(), {"source": "target_date", "turn": t}

        base_date = (t.get("date") or "").strip()  # turn.date 기준
        parsed = _parse_date_from_text(t.get("text", ""), base_date)
        if parsed:
            return parsed, {"source": "text_parsed", "turn": t}
    print(f"[DATE_LOOKUP] result={parsed or td or None}")
    return None, {"source": "not_found", "turn": None}


# 1) ```json ...``` 코드펜스 대응 정규식
#    - ```json { ... } ``` 또는 ``` { ... } ``` 모두 매칭
#    - DOTALL: 줄바꿈 포함, IGNORECASE: json/JSON 허용
_CODEFENCE_RE = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*?\})\s*```",
    re.IGNORECASE
)

# 2) 추출 프롬프트
#    - 주의: 예시 JSON의 { } 는 반드시 이중 중괄호로 이스케이프
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

def _safe_json_loads(maybe_json: Any) -> Dict[str, Any]:
    """
    JSON 파싱 안전판.
    - dict가 들어오면 그대로 반환
    - 문자열이면 json.loads 시도, 실패 시 {}
    - 그 외 타입이면 {} (LLM/체인 결과 방어)
    """
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}

def _extract_json_block(raw_text: str) -> str:
    """
    1) 코드펜스(```json ... ```) 내부 JSON만 우선 추출
    2) 없으면 문자열 전체에서 첫 '{' ~ 마지막 '}' 구간 슬라이스
    3) 그래도 없으면 "{}"
    """
    if not raw_text:
        return "{}"
    m = _CODEFENCE_RE.search(raw_text)
    if m:
        return m.group(1).strip()
    s, e = raw_text.find("{"), raw_text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return raw_text[s:e+1]
    return "{}"

def _to_text(raw: Any) -> str:
    """
    LangChain invoke 결과를 문자열로 정규화
    - AIMessage/LLMResult 등 content 속성 우선
    - dict/text 방어
    """
    if raw is None:
        return ""
    if isinstance(raw, dict) and "text" in raw:
        return str(raw["text"])
    if hasattr(raw, "content") and isinstance(raw.content, str):
        return raw.content
    if isinstance(raw, str):
        return raw
    return str(raw)

@lru_cache(maxsize=1)  # ← 체인 1회만 생성(Cold start 성능/안정성)
def get_extract_chain():
    """
    OpenAI API 키가 있을 때만 JSON-Only 응답을 강제하는 체인 생성.
    - JSON 모드: model_kwargs.response_format = {"type": "json_object"}
    - lru_cache로 동일 프로세스 내 재생성 방지(Cloud Run/Functions 권장)
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
        model_kwargs={"response_format": {"type": "json_object"}},  # ← JSON만
    )
    return _EXTRACT_PROMPT | llm



def _extract_meta(text: str) -> Dict[str, Any]:
    """
    OpenAI로 msg_keywords/target_date/time/kind/notes 추출.
    - get_extract_chain()은 JSON 응답(response_format=json_object) 강제된 체인이어야 함.
    - 키가 빠져도 기본값 세팅.
    """    
    data = {}

    try:
        extract_chain = get_extract_chain()     # ← 키 없으면 None 반환하도록 구현되어 있다면 체크
        if not extract_chain:
            print("[META] 추출 체인 없음(OPENAI_API_KEY 미설정 등) → 폴백")
            raise RuntimeError("no extract chain")
        res = extract_chain.invoke({"text": text})
        raw = _to_text(res)
        payload = _extract_json_block(raw)      # ← 이미 있던 헬퍼(없으면 raw 그대로)
        data = _safe_json_loads(payload)        # ← 안전 파싱(없으면 json.loads 예외 잡기)
        if data:
            print(f"[META] JSON 파싱 성공: {data}")
        else:
            print(f"[META] JSON 파싱 실패 → 폴백 사용")
    except Exception as e:
        print(f"[META] 예외 → 폴백: {e}")
        data = {}

    # 누락 키 보정(일관된 스키마 유지)
    data.setdefault("msg_keywords", [])
    data.setdefault("target_date", None)
    data.setdefault("time", None)
    data.setdefault("kind", None)
    data.setdefault("notes", "")

    # 중복/공백/대소문자 정규화 (한글엔 영향 없음)
    def _norm_kw_list(xs):
        out, seen = [], set()
        for x in xs or []:
            t = (x or "").strip().lower()
            if t and t not in seen:
                seen.add(t); out.append(t)
        return out

    data["msg_keywords"] = _norm_kw_list(data.get("msg_keywords"))
    if data.get("kind"):
        data["kind"] = str(data["kind"]).strip().lower()
    
    # ✅ [NEW] 시간 질문 구조화
    data["timing_structure"] = _structure_timing_question(text, data)

    return data


def _structure_timing_question(text: str, meta: dict) -> dict | None:
    """
    "언제쯤 수익이 날까?" 같은 시간 질문을 구조화
    
    Returns:
        {
            "type": "timing_question",
            "target": "business_income",  # 수익, 합격, 성사 등
            "granularity": "quarter",     # year, quarter, month
            "range": "2026",
            "urgency": "medium"            # high, medium, low
        }
    """
    if not any(tok in text for tok in ["언제", "시기", "타이밍", "몇월", "며칠"]):
        return None
    
    result = {
        "type": "timing_question",
        "target": None,
        "granularity": "quarter",
        "range": None,
        "urgency": "medium"
    }
    
    # 타겟 추출
    if "수익" in text or "돈" in text or "벌" in text:
        result["target"] = "income"
    elif "합격" in text or "붙을" in text:
        result["target"] = "pass"
    elif "성사" in text or "계약" in text:
        result["target"] = "deal"
    elif "만남" in text or "인연" in text:
        result["target"] = "relationship"
    else:
        result["target"] = "general_timing"
    
    # 세분도 추정
    if "년" in text or "올해" in text or "내년" in text:
        result["granularity"] = "year"
    elif "분기" in text or "쯤" in text:
        result["granularity"] = "quarter"
    elif "월" in text or "몇월" in text:
        result["granularity"] = "month"
    elif "일" in text or "며칠" in text:
        result["granularity"] = "day"
    
    # 연도 범위 (meta의 target_date에서 추출)
    if meta.get("target_date"):
        try:
            result["range"] = meta["target_date"][:4]  # YYYY 추출
        except:
            pass
    else:
        # 텍스트에서 연도 추출
        import re
        year_match = re.search(r"20\d{2}", text)
        if year_match:
            result["range"] = year_match.group(0)
    
    # 긴급도 (고민/불안 등의 단어로 판단)
    if any(word in text for word in ["급", "빨리", "불안", "걱정", "해야"]):
        result["urgency"] = "high"
    elif any(word in text for word in ["천천히", "여유", "궁금"]):
        result["urgency"] = "low"
    
    return result


    
def _llm_detect_regression(question: str, summary_text: str, hist: dict) -> dict:
    try:
        #print(f"_llm_detect_regression() Q : {question} : {summary_text}")
        res = _REG_CHAIN.invoke({
            # 프롬프트가 요구하는 입력만 필수로 넣자
            "summary": summary_text or "",
            "question": question,
            # (선택) 프롬프트에 반영한다면 같이 사용
            "has_history": hist.get("has_history", False),
            "history_turns": hist.get("history_turns", 0),
        })
        data = json.loads(_to_text(res))
    except Exception as e:
        print(f"[REG] 회귀 LLM 예외 → False 폴백: {e}")
        data = {
            "is_regression": False,
            "confidence": 0.0,
            "topic_keywords": [],
            "explicit_markers": [],
            "reasons": "exception",
        }

    # 누락 보정
    data.setdefault("is_regression", False)
    data.setdefault("confidence", 0.0)
    data.setdefault("topic_keywords", [])
    data.setdefault("explicit_markers", [])
    data.setdefault("reasons", "")
    return data


def _get_history_stats(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    회귀 게이트에 필요한 최소 통계만 리턴.
    - has_history: 세션에 저장된 턴이 1개 이상인지
    - history_turns: 턴 개수
    - session_id: 사용된 세션 ID (없으면 첫 세션을 자동 선택)

    주의: 여기서는 무거운 의존성 없이 conversations.json만 확인합니다.
    """
    db = _db_load()
    sessions = db.get("sessions") or {}

    # 세션이 지정되지 않았다면 첫 세션을 선택 (서비스 정책에 맞게 조정 가능)
    if session_id is None:
        session_id = next(iter(sessions), None)

    sess = sessions.get(session_id) or {}
    turns = (sess.get("turns") or [])
    print(f"turns {turns}")

    return {
        "has_history": len(turns) > 0,
        "history_turns": len(turns),
        "session_id": session_id,
    }
    
    
# ------- 외부 공개: 질문 + JSON맥락 결합 (회귀 대응) --------------------------
def build_question_with_regression_context(
    question: str,
    summary_text: str,
) -> Tuple[str, dict]:
    """
    반환:
      - (LLM에 보낼 질문 문자열, debug 메타)
    동작:
      1) 히스토리 존재 여부 확인(첫 턴 하드게이트)
      2) LLM으로 회귀 의도 판정(_llm_detect_regression); 키워드 규칙 X
      3) 회귀=True면 conversations.json을 실제로 열어:
         - 주제 유사 과거 턴 상위 N개를 뽑아 맥락 라인업 구성
         - '언제/날짜/며칠' 류면 '여행 날짜' FACT를 복원해 프롬프트에 주입
    """
    # 1) 히스토리 게이트
    hist = _get_history_stats()
    print(f"[REG] history: has_history={hist['has_history']}, turns={hist['history_turns']}")
    if not hist["has_history"]:
        print("[REG] no history → regression=False (first turn hard gate)")
        return question, {"llm": {"is_regression": False, "confidence": 0.0}, "reason": "first_turn_no_history"}

    # 2) 현재 발화 메타 (키워드, kind 등은 JSON 검색의 힌트로만 사용)
    meta_now = _extract_meta(question)
    print(f"[REG] meta_now: {meta_now}")
    #print(f"[REG] summary_text: {summary_text}")

    # 3) LLM 회귀 판정 (키워드 매칭이 아니라 LLM 판단)
    reg = _llm_detect_regression(question, summary_text, hist)
    print(f"[REG] LLM 판정: {reg}")

    conf_th = float(os.environ.get("REG_CONF_THRESH", "0.65"))
    is_reg = bool(reg.get("is_regression")) and float(reg.get("confidence", 0.0)) >= conf_th

    # 공통 debug 틀 (로그/저장용)
    debug: Dict[str, Any] = {
        "llm": {
            "is_regression": bool(reg.get("is_regression", False)),
            "confidence": float(reg.get("confidence", 0.0)),
            "topic_keywords": reg.get("topic_keywords", []),
            "explicit_markers": reg.get("explicit_markers", []),
            "reasons": reg.get("reasons", ""),
        },
        "search": {
            "context_used": 0, "searched_total": 0, "now_keywords": [],
            "now_kind": None, "scored": 0, "filtered_by_min_sim": 0, "picked": 0,
        },
        "facts": {}  # 날짜 등 구조화된 팩트가 들어감
    }

    if not is_reg:
        # 회귀 아님 → 원문 그대로
        return question, debug

    # 4) 회귀=True → JSON에서 '실제' 과거 맥락 뽑기
    merged_kws = list(set((meta_now.get("msg_keywords") or []) + (reg.get("topic_keywords") or [])))
    target_kind = meta_now.get("kind")
    context_rows, dbg = _select_context_from_json(
        merged_kws=merged_kws,
        target_kind=target_kind,
        limit_pick=8,
        session_id=None,  # 필요시 hist나 외부에서 세션 ID를 주입하도록 확장 가능
    )
    debug["search"].update({"context_used": len(context_rows), **dbg})

    # 5) '언제/날짜/며칠' 류 후속질의면 → JSON에서 '여행 날짜' FACT 복원
    is_when_like = any(tok in question for tok in ("언제", "날짜", "며칠"))
    fact_lines: List[str] = []
    if is_when_like:
        trip_date, fdbg = _find_last_trip_date_from_json(session_id=None)
        if trip_date:
            debug["facts"]["trip_date"] = {"value": trip_date, "source": fdbg.get("source")}
            # LLM이 정확히 기억하도록 FACT를 선행 문단으로 주입
            fact_lines.append(f"[FACT] 최근 여행 날짜: {trip_date}")
            debug["facts"]["injected"] = True
        else:
            debug["facts"]["injected"] = False

    # 6) 프롬프트 최종 구성
    lines = [f"- [{r.get('date','')} {r.get('time','')}] {r.get('role','')}: {r.get('text','')}" for r in context_rows]
    joined = "\n".join(lines)
    kw_str = ", ".join([k for k in merged_kws if k]) or "없음"

    header = [
        f"사용자가 과거 대화의 연속을 말하고 있습니다. (회귀 감지: True, 신뢰도={reg.get('confidence',0):.2f}, 키워드: {kw_str})",
        "다음 과거 대화 맥락을 참고하여 자연스럽게 이어 답하세요.",
    ]
    if fact_lines:
        # 날짜 같은 결정적 팩트는 맨 위에 명시하여 LLM이 틀리지 않게 함
        header.extend(fact_lines)

    if context_rows:
        # 과거 대화 라인업을 붙여서 LLM이 직접 참고하도록
        prompt = (
            "\n".join(header) + "\n" +
            f"과거 대화:\n{joined}\n\n" +
            f"현재 발화: {question}"
        )
        return prompt, debug

    # 컨텍스트가 0개여도 회귀는 유지. FACT만 있으면 그것만 붙여 원문 질문을 반환
    if fact_lines:
        prompt = "\n".join(header) + "\n" + f"현재 발화: {question}"
        return prompt, debug

    # 아무 맥락도 못 붙였으면 그냥 원문 반환(디버그로 회귀 흔적만 남음)
    return question, debug

from contextvars import ContextVar

#def _is_gs_path(p: str) -> bool:
#    return isinstance(p, str) and p.startswith("gs://")


def _norm(s: str) -> str:
    return (s or "").casefold()


# ---- JSON 저장소 유틸 ----
#CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
#_CONVO_STORE = os.path.join(CURRENT_DIR, "conversations.json")

def _now_utc_iso() -> str: 
    """현재 UTC 시각을 ISO 문자열로"""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _local_date() -> str:
    """로컬 날짜 YYYY-MM-DD"""
    return time.strftime("%Y-%m-%d", time.localtime())

def _local_time() -> str:
    """로컬 시간 HH:MM"""
    return time.strftime("%H:%M", time.localtime())

def _local_ts() -> str:
    """로컬 타임스탬프 YYYY-MM-DDTHH:MM:SS+TZ"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def record_turn_message(
    session_id: str,
    role: str,
    text: str,
    *,
    mode: str = "GEN",
    auto_meta: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
    user: Optional[Dict[str, Any]] = None,      # {"id","name","birth"} 직접 전달
    payload: Optional[Dict[str, Any]] = None,   # {"user":{"name","birth"}} 형태
) -> None:
    """
    한 턴(메시지)을 JSON DB에 기록.
    - auto_meta=True: OpenAI로 메타 자동 추출
    - extra_meta: 간지 등 추가 필드를 dict로 전달하면 turn에 합쳐 저장
    - N은 CONVO_MAX_TURNS 환경변수(기본 300)
    - GCS에 전체 JSON을 다시 업로드(덮어쓰기)하므로, 오래된 턴은 자연스럽게 사라진다(밀어내기).
    - user / payload를 주면 해당 호출 스코프에서만 사용자 컨텍스트를 임시 설정한다.
    - 설정하지 않으면 현재 컨텍스트(_CUR_USER_ID) 또는 전역 파일 사용.
    """
    
    # ── [A] 사용자 컨텍스트 임시 적용 (있으면 설정, 없으면 기존 유지) ─────────────
    _needs_reset = False
    # 기존 컨텍스트 확인 (main.py에서 이미 설정했을 수 있음)
    existing_user_id = get_current_user_id()
    existing_app_uid = get_current_app_uid()
    
    if user:
        new_user_id = user.get("id") or make_user_key(user.get("name"), user.get("birth"))
        # 기존 컨텍스트가 있고 user_id가 같으면 재설정하지 않음 (app_uid 유지)
        if existing_user_id and existing_user_id == new_user_id:
            # 컨텍스트가 이미 올바르게 설정되어 있음, app_uid만 확인/업데이트
            if not existing_app_uid and user.get("app_uid"):
                set_current_user_context(
                    name=user.get("name"),
                    birth=user.get("birth"),
                    user_id_override=user.get("id"),
                    app_uid=user.get("app_uid")
                )
                _needs_reset = True
        else:
            # ✅ user dict를 개별 인자로 전달
            set_current_user_context(
                name=user.get("name"),
                birth=user.get("birth"),
                user_id_override=user.get("id"),
                app_uid=existing_app_uid or user.get("app_uid")  # 기존 app_uid 유지 또는 user에서 가져오기
            )
            _needs_reset = True
    elif payload:
        up = user_from_payload(payload)
        if up:
            new_user_id = up.get("id")
            # ✅ payload에서 app_uid 추출 시도 (payload에 직접 있을 수 있음)
            payload_app_uid = payload.get("app_uid") or payload.get("appUid") or payload.get("uid") if payload else None
            # 기존 컨텍스트가 있고 user_id가 같으면 재설정하지 않음 (app_uid만 업데이트)
            if existing_user_id and existing_user_id == new_user_id:
                # 컨텍스트가 이미 올바르게 설정되어 있음, app_uid만 확인/업데이트
                if not existing_app_uid and payload_app_uid:
                    set_current_user_context(
                        name=up.get("name"),
                        birth=up.get("birth"),
                        user_id_override=up.get("id"),
                        app_uid=payload_app_uid
                    )
                    _needs_reset = True
            else:
                # ✅ user dict를 개별 인자로 전달, app_uid도 함께 전달
                set_current_user_context(
                    name=up.get("name"),
                    birth=up.get("birth"),
                    user_id_override=up.get("id"),
                    app_uid=existing_app_uid or payload_app_uid  # 기존 app_uid 우선, 없으면 payload에서 가져오기
                )
                _needs_reset = True
    try:
        db = _db_load()
        if session_id not in db["sessions"]:
            print(f"[WARN] 세션 {session_id} 없음 → 자동 생성")
            db["sessions"][session_id] = {
                "meta": {"session_id": session_id, "created_at": _now_utc_iso(), "title": "사주 대화"},
                "turns": []
            }

        turn = {
            "ts": _local_ts(),
            "date": _local_date(),
            "time": _local_time(),
            "role": role,
            "mode": mode,
            "text": text,
        }

        if auto_meta:
            meta = _extract_meta(text)
            #print(f"[META] 자동추출: {meta}")
            if meta.get("msg_keywords"): turn["msg_keywords"] = meta["msg_keywords"]
            if meta.get("target_date"):  turn["target_date"]  = meta["target_date"]
            if meta.get("time"):         turn["event_time"]   = meta["time"]
            if meta.get("kind"):         turn["kind"]         = meta["kind"]
            if meta.get("notes"):        turn["notes"]        = meta["notes"]

        if extra_meta:
            for k, v in extra_meta.items():
                if v is not None:
                    turn[k] = v
            print(f"[META-EXTRA] 추가 메타: {extra_meta}")

        # ── [D] append → 오래된 턴 컷 → 저장(덮어쓰기) ─────────────────────────
        db["sessions"][session_id]["turns"].append(turn)

        cur_len = len(db["sessions"][session_id]["turns"])
        print(f"[STORE] session='{session_id}' appended -> len={cur_len}")
        # removed = _trim_session_turns(db, session_id, max_turns=_max_turns())
        # if removed:
        #     kept = len(db["sessions"][session_id]["turns"])
        #     print(f"[TRIM] session='{session_id}' removed={removed} kept={kept} (limit={_max_turns()})")

        _db_save(db)  # GCS에 전체 JSON 재업로드 → 앞쪽(오래된) 턴은 파일에서 제거됨

    finally:
        # ── [C] 이 함수 내부에서 컨텍스트를 세팅했다면 끝나고 해제 ────────────────
        if _needs_reset:
            set_current_user_context(reset=True)
# ================== /JSON 저장 유틸 ==================

def ensure_session(sid: str, title: str = "사주 대화", *, user: dict | None = None, payload: dict | None = None) -> str:
#def ensure_session(sid: str, title: str = "사주 대화") -> str:
    """세션이 없으면 생성, 있으면 사용"""
    
    """
      - user/payload로 사용자 컨텍스트를 임시 설정 가능
      - 세션은 사용자 파일 내에 생성/확인
    """
    
    _needs_reset = False
    if user:
        set_current_user_context(user=user); _needs_reset = True
    elif payload:
        up = user_from_payload(payload)
        if up:
            set_current_user_context(user=up); _needs_reset = True
    try: 
        db = _db_load()
        if not isinstance(db.get("sessions"), dict):
            db["sessions"] = {}  # 방어
        if sid not in db["sessions"]:
            db["sessions"][sid] = {
                "meta": {"session_id": sid, "created_at": _now_utc_iso(), "title": title},
                "turns": []
            }
            _db_save(db)
            print(f"[SESSION] 새 세션 생성: {sid}")
        return sid
    finally:
        if _needs_reset:
            set_current_user_context(reset=True)



import re
from datetime import datetime, timedelta, timezone

# yyyy-mm-dd (ISO)
ISO_DATE_RE = re.compile(r"\b(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")

# (내후년|내년|올해|작년|재작년)? 10월 4일  → year는 지시어 기준으로 해석
KOR_ABS_DATE_RE = re.compile(
    r"(내후년|내년|올해|작년|재작년)?\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"
)

# 상대 '일' 단위
RELATIVE_DAY_TOKENS = {
    "오늘": 0,
    "내일": 1,
    "내일모레": 2,   # 붙여쓴 형태
    "내일 모레": 2,  # 띄어쓴 형태
    "모레": 2,
    "글피": 3,
    "어제": -1,
    "그저께": -2,
}

# 상대 '년' 지시어 (월/일이 함께 올 때만 실제 날짜 산출에 사용)
RELATIVE_YEAR_TOKENS = {
    "올해": 0,
    "내년": 1,
    "내후년": 2,
    "작년": -1,
    "재작년": -2,
}

def _now_kr():
    # GCF/Cloud Run 기본은 UTC. 한국 기준이 필요하면 이렇게 고정.
    # pytz 없이도 UTC+9 고정 오프셋으로 충분하면 아래처럼.
    return datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(
        timezone(timedelta(hours=9))
    )

def _safe_json_loads(s: str):
    import json
    try: return json.loads(s)
    except Exception: return {}

def _to_text(raw):
    if raw is None: return ""
    if hasattr(raw, "content"): return raw.content
    if isinstance(raw, str): return raw
    return str(raw)
    
def _today() -> date:
    """
    '내일/모레' 같은 상대표현을 정확히 계산하려면 기준 날짜가 중요함.
    - Cloud Run/Functions가 UTC라면 KST(+09:00)로 보정하는 게 보통 기대와 일치.
    - 환경이 이미 KST라면 use_kst=False로 바꿔도 됨.
    """
    return datetime.now(timezone(timedelta(hours=9))).date()  # 필요에 맞게 조정


def _maybe_override_target_date(question: str, parsed: dict, now: date) -> None:
    # 이미 ISO yyyy-mm-dd 라면 유지
    td = parsed.get("target_date")
    def _bad(td: str) -> bool:
        # 과거 3년보다 지나치게 멀거나(LLM 헛발), 포맷 비정상 → 덮어씀
        try:
            d = date.fromisoformat(td)
            return (d < now.replace(year=now.year-3)) or (d > now.replace(year=now.year+4))
        except Exception:
            return True

    # 상대 토큰이 포함되면 우선순위 높게 덮어씀
    for tok, off in RELATIVE_DAY_TOKENS.items():
        if tok in question:
            newd = (now + timedelta(days=off)).isoformat()
            if not td or _bad(td):
                parsed["target_date"] = newd
                parsed.setdefault("_facts", {})["deixis_anchor_date"] = {"value": newd, "source": f"relative:{tok}"}
                print(f"[DEIXIS][TIME] '{tok}' → target_date={newd}")
            break
        
        
