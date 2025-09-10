# ===== JSON 대화저장 + 메타추출 (OpenAI + LangChain 0.2.x) =====
import json, time, uuid, os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
_EXTRACT_PROMPT = PromptTemplate(
    input_variables=["text"],   # {text} 변수만 받음
    template=(
        "아래 문장에서 대화 저장용 메타데이터를 JSON으로 추출하라.\n"
        "출력은 반드시 JSON만, 키는 정확히 다음만 포함:\n"
        "- msg_keywords: 핵심 키워드 배열\n"
        "- target_date: 날짜가 있으면 ISO(YYYY-MM-DD), 없으면 null\n"
        "- time: 시/분이 있으면 HH:MM, 없으면 null\n"
        "- kind: 면접/시험/여행/회의/계약/결혼/병원/주식 중 하나 또는 null\n"
        "- notes: 문장 요약 (한 줄)\n\n"
        "문장: {text}"
    ),
)

def _get_extract_chain() -> LLMChain:
    """추출 전용 LLMChain 초기화"""
    print(f"[CHAIN] 메타추출 체인 초기화: model={os.getenv('EXTRACT_MODEL','gpt-4o-mini')}, temp={os.getenv('EXTRACT_TEMPERATURE','0.0')}")
    llm_extract = ChatOpenAI(
        model=os.environ.get("EXTRACT_MODEL", "gpt-4o-mini"),
        temperature=float(os.environ.get("EXTRACT_TEMPERATURE", "0.0")),
        max_tokens=300,
        timeout=25,
        # max_retries=2,  # 필요시
    )
    return LLMChain(llm=llm_extract, prompt=_EXTRACT_PROMPT)

_EXTRACT_CHAIN = _get_extract_chain()


_CODEFENCE_RE = re.compile(r"```[a-zA-Z]*\s*([\s\S]*?)```", re.MULTILINE)

def _to_text(raw: Any) -> str:
    """LLM 응답이 dict/obj/str 무엇이든 사람이 읽을 수 있는 문자열로 변환"""
    if raw is None:
        return ""
    # langchain invoke/run의 결과가 객체이거나 dict 형태일 수 있음
    if isinstance(raw, dict) and "text" in raw:
        return raw["text"]
    if hasattr(raw, "content") and isinstance(raw.content, str):
        return raw.content
    if hasattr(raw, "text") and isinstance(raw.text, str):
        return raw.text
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
    
def _extract_meta(text: str) -> Dict[str, Any]:
    """
    OpenAI 모델을 사용해 JSON 메타를 추출한다.
    실패 시 최소 스켈레톤으로 폴백한다.
    OpenAI로 msg_keywords/target_date/time/kind/notes 추출 (invoke + 견고한 파싱)
    """
    print(f"[META] 입력 문장: {text}")
    try:
        # run 또는 invoke 어떤 걸 쓰더라도 _to_text로 정규화
        res = _EXTRACT_CHAIN.run(text=text)  # ← 지금 run을 쓰는게 target 계산 OK라면 유지
        raw = _to_text(res)
        print(f"[META] 원시 응답: {raw}")
        payload = _extract_json_block(raw)
        data = _safe_json_loads(payload)
        if data:
            print(f"[META] JSON 파싱 성공: {data}")
        else:
            print(f"[META] JSON 파싱 실패 → 폴백 사용")
    except Exception as e:
        print(f"[META] 예외 → 폴백: {e}")
        data = {}
    
    # raw : LLM이 준 원시 응답 문자열 (예: json { "msg_keywords": [...], ... } 이런 형태)
    # raw.find("{") : 문자열에서 첫 번째 {의 위치(인덱스)를 찾음
    # raw.rfind("}") : 문자열에서 마지막 }의 위치를 찾음
    # raw[s:e+1] : 그 구간만 잘라서 JSON으로 파싱 시도
    # else raw : 만약 {}가 아예 없으면 그냥 원문 그대로 사용
    # 👉 요약:
    # LLM이 JSON 앞뒤에 불필요한 텍스트나 json 코드펜스를 붙여도, {...} 구간만 추출해서 json.loads()에 안전하게 넘기려는 보호 코드입니다.

    # 누락 키 보정
    data.setdefault("msg_keywords", [])
    data.setdefault("target_date", None)
    data.setdefault("time", None)
    data.setdefault("kind", None)
    data.setdefault("notes", "")

    return data
# ================== /메타 추출 체인 ==================


# ---- JSON 저장소 유틸 ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVO_STORE = os.path.join(CURRENT_DIR, "conversations.json")

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

def _mk_sid() -> str:
    """세션 ID 생성: UTC시간 + 랜덤 4hex"""
    sid = f"{_now_utc_iso()}#{uuid.uuid4().hex[:4]}"
    print(f"[SESSION-ID] 새 세션 ID: {sid}")
    return sid

def _db_load() -> dict:
    """
    conversations.json 읽기. 없거나 비어있으면 기본 구조(dict 기반) 생성.
    sessions가 list로 저장된 과거 포맷이면 dict로 마이그레이션.
    """
    try:
        with open(_CONVO_STORE, "r", encoding="utf-8") as f:
            db = json.load(f)
            #print(f"[JSON-LOAD] {_CONVO_STORE} 로드 성공")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[JSON-LOAD] {_CONVO_STORE} 없음/비어있음 → 새 DB 구조 생성")
        db = {"version": 1, "sessions": {}}   # ✅ dict 기반!

    # --- 안전 정규화(예전 포맷이 list였던 경우 자동 변환) ---
    sess = db.get("sessions")
    if isinstance(sess, list):
        #print("[JSON-LOAD] sessions가 list 포맷 → dict로 마이그레이션")
        new_sessions = {}
        for s in sess:
            sid = (s.get("meta") or {}).get("session_id") or s.get("id") or f"migrated_{len(new_sessions)+1}"
            new_sessions[sid] = s
        db["sessions"] = new_sessions

    if "sessions" not in db or not isinstance(db["sessions"], dict):
        db["sessions"] = {}  # 최종 방어

    return db

#def _db_save(db: Dict[str, Any]) -> None:
def _db_save(db: dict) -> None:
    """
    dict를 conversations.json으로 저장.
    """
    with open(_CONVO_STORE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    #print(f"[JSON-SAVE] {_CONVO_STORE} 저장 완료 (세션 수: {len(db.get('sessions', {}))})")  
    

#def ensure_session(session_id: Optional[str], title: str = "사주 대화") -> str:
def ensure_session(sid: str, title: str = "사주 대화") -> str:
    """세션이 없으면 생성, 있으면 사용"""
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

def record_turn_message(
    session_id: str,
    role: str,
    text: str,
    *,
    mode: str = "GEN",
    auto_meta: bool = True,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    한 턴(메시지)을 JSON DB에 기록.
    - auto_meta=True: OpenAI로 메타 자동 추출
    - extra_meta: 간지 등 추가 필드를 dict로 전달하면 turn에 합쳐 저장
    """
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

    db["sessions"][session_id]["turns"].append(turn)
    _db_save(db)
    #print(f"[TURN] 저장 완료: session={session_id}, role={role}")
# ================== /JSON 저장 유틸 ==================


# ====== 대화 검색 유틸 ======
from typing import Tuple

def _norm(s: str) -> str:
    return (s or "").casefold()


def search_messages(
    *,
    query: str | None = None,
    keywords: list[str] | None = None,
    keyword_match: str = "any",     # "any" | "all"
    date: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    target_date: str | None = None,
    session_id: str | None = None,
    role: str | None = None,
    mode: str | None = None,
    has_target: bool | None = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[list[dict], int]:
    """
    conversations.json 전체에서 조건으로 필터링하여 최신순 반환.
    - sessions가 dict(권장) 이든 list(과거 포맷) 이든 모두 지원
    return: (rows, total_count)
    """
    db = _db_load()
    rows: list[dict] = []

    # --- 입력 정규화 ---
    q = _norm(query) if query else None
    kwset = set((k or "").strip() for k in (keywords or []) if k and k.strip())

    sessions = db.get("sessions", {})
    # sessions가 list일 수도 있으므로 이터레이터 통일
    def _iter_sessions():
        if isinstance(sessions, dict):
            for sid, sess in sessions.items():
                yield sid, sess
        elif isinstance(sessions, list):
            for sess in sessions:
                sid = (sess.get("meta", {}) or {}).get("session_id") or sess.get("id") or "unknown"
                yield sid, sess
        else:
            return

    for sid, sess in _iter_sessions():
        if session_id and sid != session_id:
            continue

        turns = sess.get("turns", []) or []
        for turn in turns:
            # --- 공통 필터 ---
            if role and turn.get("role") != role:
                continue
            if mode and turn.get("mode") != mode:
                continue
            if date and turn.get("date") != date:
                continue
            if date_from and (turn.get("date", "") < date_from):
                continue
            if date_to and (turn.get("date", "") > date_to):
                continue
            if target_date and turn.get("target_date") != target_date:
                continue
            if has_target is True and not turn.get("target_date"):
                continue
            if has_target is False and turn.get("target_date"):
                continue

            # --- 키워드 필터(msg_keywords) ---
            msgs_kw = turn.get("msg_keywords") or []
            # 혹시 문자열로 저장된 경우 방어
            if isinstance(msgs_kw, str):
                msgs_kw = [msgs_kw]
            if kwset:
                msg_kw_set = set(msgs_kw)
                if keyword_match == "all":
                    if not kwset.issubset(msg_kw_set):
                        continue
                else:  # "any"
                    if kwset.isdisjoint(msg_kw_set):
                        continue

            # --- 자유 텍스트 필터(본문/요약/키워드 포함) ---
            if q:
                hay = " ".join([
                    _norm(turn.get("text", "")),
                    _norm(turn.get("notes", "")),
                    _norm(" ".join(msgs_kw)),
                ])
                if q not in hay:
                    continue

            # 매치된 항목 수집 (+ 세션 id 주입)
            row = dict(turn)
            row["_session_id"] = sid
            rows.append(row)

    # --- 정렬: ts(ISO) 우선, 없으면 date+time 사용 ---
    def _key(t: dict):
        ts = t.get("ts")
        if ts:
            return ts
        return f"{t.get('date','')}T{t.get('time','')}+0000"

    rows.sort(key=_key, reverse=True)

    total = len(rows)
    start = max(0, int(offset))
    end = start + max(0, int(limit))
    sliced = rows[start:end]

    #print(f"[SEARCH] total={total}, offset={offset}, limit={limit}, returned={len(sliced)}")
    return sliced, total


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
