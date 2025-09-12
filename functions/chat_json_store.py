# ===== JSON 대화저장 + 메타추출 (OpenAI + LangChain 0.2.x) =====
import json, time, uuid, os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI



# ───────── GCS 유틸 ─────────
from google.cloud import storage

def _is_gs_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith("gs://")

def _parse_gs_path(gs_path: str) -> tuple[str, str]:
    # gs://bucket/name/with/slashes.json -> (bucket, name/with/slashes.json)
    no_scheme = gs_path[len("gs://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    name = parts[1] if len(parts) > 1 else ""
    #print(f"_parse_gs_path({str}, name : {name})")
    return bucket, name


def _gcs_read_text(gs_path: str) -> str:
    bucket, name = _parse_gs_path(gs_path)
    client = storage.Client()
    bkt = client.bucket(bucket)
    blob = bkt.blob(name)
    if not blob.exists(client):
        raise FileNotFoundError(gs_path)
    return blob.download_as_text(encoding="utf-8")

def _gcs_write_text(gs_path: str, text: str) -> None:
    bucket, name = _parse_gs_path(gs_path)
    client = storage.Client()
    bkt = client.bucket(bucket)
    blob = bkt.blob(name)
    blob.cache_control = "no-store"
    blob.content_type = "application/json"
    blob.upload_from_string(text, content_type="application/json")
    #print(f"[JSON-SAVE] GCS {gs_path} 저장 완료 (size={len(text)} bytes)")


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


def _extract_meta(text: str) -> Dict[str, Any]:
    """
    OpenAI 모델을 사용해 JSON 메타를 추출한다.
    실패 시 최소 스켈레톤으로 폴백한다.
    OpenAI로 msg_keywords/target_date/time/kind/notes 추출 (invoke + 견고한 파싱)
    """
    print(f"[META] 입력 문장: {text}")
    data = {}

    try:
        # run 또는 invoke 어떤 걸 쓰더라도 _to_text로 정규화
        extract_chain = get_extract_chain()
        res = extract_chain.invoke({"text": text})
        raw = _to_text(res)
        #print(f"[META] 원시 응답: {raw}")  //사용자 질문
        payload = _extract_json_block(raw)
        data = _safe_json_loads(payload)
        if data:
            print(f"[META] JSON 파싱 성공: {data}")
        else:
            print(f"[META] JSON 파싱 실패 → 폴백 사용")
    except Exception as e:
        print(f"[META] 예외 → 폴백: {e}")
        data = {}


    # 누락 키 보정
    data.setdefault("msg_keywords", [])
    data.setdefault("target_date", None)
    data.setdefault("time", None)
    data.setdefault("kind", None)
    data.setdefault("notes", "")
    
    def _norm_kw_list(xs):
        out, seen = [], set()
        for x in xs or []:
            t = (x or "").strip().lower()
            if t and t not in seen:
                seen.add(t); out.append(t)
        return out

    data["msg_keywords"] = _norm_kw_list(data.get("msg_keywords"))
    if data.get("kind"):
        data["kind"] = (str(data["kind"]).strip().lower())

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

def _resolve_store_path() -> str:    
    # 1. 환경변수 우선
    override = os.getenv("CONVO_STORE_PATH")
    if override and override.startswith("gs://"):
        #print(f"[PATH] 환경변수(GCS) 경로 사용: {override}")
        return override

    # 2. 컨테이너/클라우드 환경 감지
    in_cloud = bool(
        os.getenv("K_SERVICE") or 
        os.getenv("FUNCTION_TARGET") or
        os.getenv("FIREBASE_CONFIG")
    )
    if in_cloud:
        # 여기에 Cloud Storage 경로를 직접 지정하거나 환경변수로 받아옵니다.
        # 예: gs://YOUR_BUCKET_NAME/conversations.json
        bucket_name = os.getenv("GCS_BUCKET") or 'your-project-id.appspot.com'
        path = f"gs://{bucket_name}/conversations.json"
        print(f"[PATH] Cloud 환경 감지 → GCS 사용: {path}")
        return path

    # 3. 로컬
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "conversations.json")
    #print(f"[PATH] 로컬 경로 사용: {path}") // 모든 대화 내용 출력
    return path



def _db_load() -> dict:
    print("_db_load()")
    path = _resolve_store_path()
    #print(f"_CONVO_STORE : {path}")
    
    try:
        if _is_gs_path(path):
            raw = _gcs_read_text(path)
            db = json.loads(raw)
        else:
            with open(path, "r", encoding="utf-8") as f:
                db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"[JSON-LOAD] {path} 없음/비어있음 → 새 DB 구조 생성")
        db = {"version": 1, "sessions": {}}

    # list → dict 마이그레이션 유지
    sess = db.get("sessions")
    if isinstance(sess, list):
        new_sessions = {}
        for s in sess:
            sid = (s.get("meta") or {}).get("session_id") or s.get("id") or f"migrated_{len(new_sessions)+1}"
            new_sessions[sid] = s
        db["sessions"] = new_sessions
    if "sessions" not in db or not isinstance(db["sessions"], dict):
        db["sessions"] = {}
    return db


#def _db_save(db: Dict[str, Any]) -> None:
def _db_save(db: dict) -> None:
    path = _resolve_store_path()
    payload = json.dumps(db, ensure_ascii=False, indent=2)
    #print(f"path : {path}, payload : {payload}") // payload: 모든 대화내용 출력 , path :  gs://chatsaju-5cd67-convos/conversations.json
    if _is_gs_path(path):
        _gcs_write_text(path, payload)
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(payload)

    # 진단용: 총 턴 수 출력
    sessions = db.get("sessions", {})
    turns = 0
    if isinstance(sessions, dict):
        for s in sessions.values():
            turns += len(s.get("turns", []))
    print(f"[JSON-SAVE] {path} 저장 완료 (세션 수: {len(sessions)}, 총 턴 수: {turns})")
    

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
    print(f"[TURN] 저장 완료: session={session_id}, role={role}")
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


_gcs_client = None

def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        _gcs_client = storage.Client()
    return _gcs_client

def load_conversations_gcs(bucket_name: str, blob_path: str) -> dict:
    """
    GCS에서 conversations.json 로드. import 시점이 아니라 '호출 시점'에만 접근.
    """
    try:
        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            print(f"[JSON-LOAD] gs://{bucket_name}/{blob_path} 없음/비어있음 → 새 DB 구조 생성")
            return {"sessions": {}, "total_turns": 0}
        content = blob.download_as_text(encoding="utf-8")
        return json.loads(content) if content.strip() else {"sessions": {}, "total_turns": 0}
    except Exception as e:
        print(f"[JSON-LOAD] 예외 → 새 구조 폴백: {e}")
        return {"sessions": {}, "total_turns": 0}

def save_conversations_gcs(bucket_name: str, blob_path: str, data: dict):
    try:
        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(json.dumps(data, ensure_ascii=False), content_type="application/json")
        print(f"[JSON-SAVE] gs://{bucket_name}/{blob_path} 저장 완료 (세션 수: {len(data.get('sessions', {}))}, 총 턴 수: {data.get('total_turns', 0)})")
    except Exception as e:
        print(f"[JSON-SAVE] 예외: {e}")