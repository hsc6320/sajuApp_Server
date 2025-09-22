# ================== [NEW] 사용자별 파일 저장을 위한 컨텍스트 유틸 ==================
# 기존 코드와 최대한 어울리게, 함수명/스타일을 유지하면서 '현재 사용자' 개념만 주입합니다.

import os, re, json, unicodedata, hashlib
from contextlib import contextmanager

# 현재 요청에서 사용할 "사용자 컨텍스트" (프로세스 전역; 요청 종료 시 반드시 해제)
_CUR_USER_ID: str | None = None
_CUR_USER_META: dict | None = None  # {"name":..., "birth":...}

def _norm_birth(birth: str | None) -> str:
    b = (birth or "").strip()
    if not b:
        return "unk"
    b = b.replace("-", "").replace("/", "").replace(".", "")
    if len(b) == 8 and b.isdigit():
        return b
    # YYYY-MM-DD 등은 위에서 제거됨, 그 외 형식은 전부 unk 처리
    return "unk"

def make_user_key(name: str | None, birth: str | None) -> str:
    n = (name or "익명").strip() or "익명"
    n = re.sub(r"[^0-9A-Za-z가-힣_.-]", "_", n)[:64]
    b = _norm_birth(birth)
    return f"{n}__{b}"

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

from contextvars import ContextVar

# 요청별로 안전한 컨텍스트 저장(동시 처리 대비)
_CUR_USER_ID: ContextVar[str | None] = ContextVar("_CUR_USER_ID", default=None)
_CUR_USER_META: ContextVar[dict | None] = ContextVar("_CUR_USER_META", default=None)

def set_current_user_context(*, name: str | None = None, birth: str | None = None, reset: bool = False):
    from conv_store import make_user_key  # 이미 있는 함수 재사용
    if reset:
        _CUR_USER_ID.set(None)
        _CUR_USER_META.set(None)
        return
    if name and birth:
        uid = make_user_key(name, birth)
        _CUR_USER_ID.set(uid)
        _CUR_USER_META.set({"name": name, "birth": birth})
    else:
        _CUR_USER_ID.set(None)
        _CUR_USER_META.set(None)

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
    현재 런타임 환경에 맞춰 '해당 사용자 전용' JSON 경로를 계산합니다.
    - Cloud(Firebase/Cloud Run 등): GCS 사용
      * 필수: GCS_BUCKET
      * 선택: GCS_PREFIX (기본 'conversations')
      → gs://<bucket>/<user_id>.json
    - 로컬: CONVO_BASE (기본 ./data)
      → <CONVO_BASE>/conversations/<user_id>.json
    """
    
    import os, os.path as p
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")
    user_id = user_id.strip()

    in_cloud = any(os.getenv(k) for k in ("K_SERVICE", "FUNCTION_TARGET", "FIREBASE_CONFIG"))
    if in_cloud:
        bucket = os.getenv("GCS_BUCKET")
        if not bucket:
            raise RuntimeError("GCS_BUCKET is required in cloud environment. 예: GCS_BUCKET=chatsaju-5cd67-convos")
        path = f"gs://{bucket}/{user_id}.json"   # ← prefix 제거, 루트에 저장
        print(f"[PATH] cloud per-user → {path}")
        return path

    base = os.getenv("CONVO_BASE", "./data")
    conv_dir = p.join(p.abspath(base), "conversations")
    os.makedirs(conv_dir, exist_ok=True)
    path = p.join(conv_dir, f"{user_id}.json")
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
