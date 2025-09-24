# ================== [NEW] 사용자별 파일 저장을 위한 컨텍스트 유틸 ==================
# 기존 코드와 최대한 어울리게, 함수명/스타일을 유지하면서 '현재 사용자' 개념만 주입합니다.

from typing import Optional
import os, re, json, unicodedata, hashlib
import os, os.path as p
from contextlib import contextmanager
from contextvars import ContextVar
from google.cloud import storage  # GCS 삭제용

chatJsonFilePath  = ""
def _parse_gs_path(gs_path: str) -> tuple[str, str]:
    # gs://bucket/name/with/slashes.json -> (bucket, name/with/slashes.json)
    no_scheme = gs_path[len("gs://"):]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    name = parts[1] if len(parts) > 1 else ""
    #print(f"_parse_gs_path({str}, name : {name})")
    return bucket, name

def _is_gs_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith("gs://")

def _norm_birth(birth: str | None) -> str:
    b = (birth or "").strip()
    if not b:
        return "unk"
    b = b.replace("-", "").replace("/", "").replace(".", "")
    if len(b) == 8 and b.isdigit():
        return b
    # YYYY-MM-DD 등은 위에서 제거됨, 그 외 형식은 전부 unk 처리
    return "unk"

def user_from_payload(payload: dict | None) -> dict | None:
    """
    payload 예)
      {"user": {"name": "홍", "birth": "1988-05-28"}}
    없으면 None 반환 (기존 전역 파일 경로로 저장됨)
    """
    if not payload:
        return None
    u = (payload.get("user") or {})
    if not u:
        return None
    name = u.get("name")
    birth = u.get("birth")
    if not name or not birth:
        return None
    return {
        "id": make_user_key(name, birth),
        "name": name,
        "birth": birth,
    }

# 요청별로 안전한 컨텍스트 저장(동시 처리 대비)
_CUR_USER_ID: ContextVar[str | None]  = ContextVar("_CUR_USER_ID",  default=None)
_CUR_USER_META: ContextVar[dict | None] = ContextVar("_CUR_USER_META", default=None)

# --- 파일키 유틸들 ---------------------------------------------------------

def _sanitize_name(name: str | None) -> str:
    """
    파일명으로 쓰일 '이름'을 안전하게 정규화.
    - 허용: 한글/영문/숫자/._-
    - 나머지는 '_'로 치환
    - 빈 값 방어: 'guest'로 대체
    """
    n = (name or "").strip() or "guest"
    n = re.sub(r"[^0-9A-Za-z가-힣_.-]", "_", n)
    return n[:64]  # 과도한 길이 방지

def make_user_id_from_name(name: str | None) -> str:
    """이름만으로 파일 키 생성 → <name>"""
    return _sanitize_name(name)

def make_user_key(name: str | None, birth: str | None) -> str:
    n = _sanitize_name(name)   # 한글/영문/숫자/._- 만 허용 + 길이 제한
    b = _norm_birth(birth)     # YYYYMMDD 정규화, 아니면 "unk"
    return f"{n}__{b}"

# --- 컨텍스트 세팅 ---------------------------------------------------------

def set_current_user_context(
    *,
    name: str | None = None,
    birth: str | None = None,
    reset: bool = False,
    user_id_override: Optional[str] = None,   # ← 필요 시 파일키를 직접 지정(최우선)
) -> None:
    """
    컨텍스트에 현재 요청의 '파일 키'를 저장한다.
    우선순위:
      1) user_id_override 가 있으면 그대로 사용 (예: name.json 강제 등)
      2) 둘 다 있으면: make_user_key(name, birth)  → <name>__<YYYYMMDD|unk>
      3) birth가 없으면: make_user_id_from_name(name) → <name>
      4) name도 없으면: 'guest'
    """
    if reset:
        _CUR_USER_ID.set(None)
        _CUR_USER_META.set(None)
        return

    
        # 입력 정규화
    name  = (name  or "").strip() or None
    birth = (birth or "").strip() or None

    # 1) 이름+생일 최우선
    if name and birth:
        uid = make_user_key(name, birth)
    # 2) override 차순위
    elif user_id_override and user_id_override.strip():
        uid = user_id_override.strip()
    # 3) 이름만 / 이름도 없으면 guest
    else:
        uid = make_user_id_from_name(name)

    _CUR_USER_ID.set(uid)
    _CUR_USER_META.set({"name": name, "birth": birth})

def get_current_user_id() -> str | None:
    """현재 요청 컨텍스트의 파일키 읽기(없으면 None)"""
    try:
        uid = _CUR_USER_ID.get()
    except LookupError:
        return None
    return (uid or "").strip() or None


@contextmanager
def user_context(*, user: dict | None = None, name: str | None = None, birth: str | None = None):
    """
    예)
      with user_context(name="홍", birth="1988-05-28"):
          ensure_session("single_global_session")
          record_turn_message(...)

    - 블록 안에서만 사용자 컨텍스트가 유지됩니다.
    """
    try:
        set_current_user_context(user=user, name=name, birth=birth)
        yield
    finally:
        set_current_user_context(reset=True)

def _resolve_store_path_for_user(user_id: str) -> str:    
    """
    Cloud: gs://<GCS_BUCKET>/<user_id>.json   (루트 저장)
    - Cloud(Firebase/Cloud Run 등): GCS 사용
      * 필수: GCS_BUCKET
    Local: <CONVO_BASE>/conversations/<user_id>.json
    - user_id 정규화(NFKC), '/' '\' 제거(치환), .json 확장자 중복 방지
    """
    
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    # 1) 키 안전화
    key = unicodedata.normalize("NFKC", user_id.strip())
    key = key.replace("/", "_").replace("\\", "_")  # GCS 객체명 안전
    if not key.lower().endswith(".json"):
        key += ".json"

    # 2) 클라우드/로컬 분기
    in_cloud = any(os.getenv(k) for k in ("K_SERVICE", "FUNCTION_TARGET", "FIREBASE_CONFIG"))
    if in_cloud:
        bucket = os.getenv("GCS_BUCKET")
        if not bucket:
            raise RuntimeError("GCS_BUCKET is required. e.g. GCS_BUCKET=chatsaju-5cd67-convos")

        # ★ 루트에 저장 (prefix 없음)
        path = f"gs://{bucket}/{key}"       
        #print(f"[PATH] cloud per-user → {path}")           //[PATH] cloud per-user → gs://chatsaju-5cd67-convos/김지은__19880716.json
        chatJsonFilePaht = path
        return path

    # local
    base = os.getenv("CONVO_BASE", "./data")
    conv_dir = os.path.join(os.path.abspath(base), "conversations")
    os.makedirs(conv_dir, exist_ok=True)
    path = os.path.join(conv_dir, key)
    print(f"[PATH] local per-user → {path}")
    
    return path

def _new_db_skeleton() -> dict:
    """
    새 JSON을 만들 때의 기본 스키마.
    - 기존 코드와의 호환을 위해 'sessions' 루트는 동일 구조 유지
    - 사용자 정보는 선택(있으면 저장)
    """
    db = {"version": 1, "sessions": {}}
    if _CUR_USER_ID and _CUR_USER_META:
        db["user"] = {"id": _CUR_USER_ID, **_CUR_USER_META}
    return db
# ================== [/NEW] ====================================================




def _gcs_delete(gs_path: str) -> bool:
    """지정 GCS 객체 삭제. 존재하지 않아도 False로만 리턴(예외 방지)."""
    bucket, name = _parse_gs_path(gs_path)
    client = storage.Client()
    bkt = client.bucket(bucket)
    blob = bkt.blob(name)
    try:
        if not blob.exists(client):
            print(f"[DEL] GCS object not found: {gs_path}")
            return False
        blob.delete()  # 성공 시 204
        print(f"[DEL] GCS deleted: {gs_path}")
        return True
    except Exception as e:
        print(f"[DEL][ERR] GCS delete failed: {gs_path} — {e}")
        return False

def _local_delete(path: str) -> bool:
    """로컬 파일 삭제. 없으면 False."""
    try:
        if p.exists(path):
            os.remove(path)
            print(f"[DEL] Local deleted: {path}")
            return True
        print(f"[DEL] Local file not found: {path}")
        return False
    except Exception as e:
        print(f"[DEL][ERR] Local delete failed: {path} — {e}")
        return False

def delete_user_store_by_id(user_id: str) -> bool:
    """
    이미 완성된 user_id(예: '김지은' 또는 '김지은__19880716')의 파일을 삭제.
    경로 규칙은 _resolve_store_path_for_user()와 동일하게 적용됨.
    """
    # _resolve_store_path_for_user는 .json 확장자 붙이고 안전화함
    path = _resolve_store_path_for_user(user_id)
    print(f"[DEL] target → {path}")
    return _gcs_delete(path) if _is_gs_path(path) else _local_delete(path)

def delete_current_user_store() -> bool:
    """
    현재 컨텍스트(_CUR_USER_ID)에 설정된 사용자 파일 삭제.
    ask_saju 등에서 set_current_user_context()가 먼저 호출되어 있어야 함.
    """
    uid = get_current_user_id()
    if not uid:
        print("[DEL] No current user context — skip")
        return False
    return delete_user_store_by_id(uid)

def _max_turns() -> int:
    raw = os.getenv("CONVO_MAX_TURNS", "10")  # 기본 10
    try:
        val = int(str(raw).strip())
    except Exception:
        print(f"[TRIM][CONF] invalid CONVO_MAX_TURNS='{raw}', fallback=10")
        val = 10
    # 최소 10 보장
    limit = max(10, val)
    #print(f"[TRIM][CONF] CONVO_MAX_TURNS raw='{raw}' -> resolved={limit}")
    return limit

    
def _trim_session_turns(db: dict, session_id: str, *, max_turns: int | None = None) -> int:
    """
    세션 내 turns를 뒤에서부터(최신 우선) max_turns만 남기고 앞부분(오래된 것) 삭제.
    반환: 삭제된 개수
    """
    limit = max_turns or _max_turns()
    sess = (db.get("sessions") or {}).get(session_id) or {}
    turns = list(sess.get("turns") or [])
    before = len(turns)

    # 방어: limit가 비정상(예: 0/음수/작게 들어옴)이어도 최소 10 유지
    if limit < 10:
        print(f"[TRIM][GUARD] limit<{10} detected ({limit}) -> force=10")
        limit = 10

    if before <= limit:
        print(f"[TRIM] session='{session_id}' before={before} <= limit={limit} -> no trim")
        return 0

    kept = turns[-limit:]              # ✅ '최근 limit개' 유지
    removed = before - len(kept)
    sess["turns"] = kept
    db["sessions"][session_id] = sess

    #print(f"[TRIM] session='{session_id}' before={before} removed={removed} kept={len(kept)} limit={limit}")
    return removed
