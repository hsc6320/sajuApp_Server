from curses import meta
from datetime import date, datetime
import hashlib
import logging
import os
import json
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from functools import lru_cache
import time

from conv_store import (
    set_current_user_context,
    make_user_id_from_name,
    delete_current_user_store,
    get_current_user_id,
    _resolve_store_path_for_user,
    trim_session_history,
    MAX_TURNS
)
from creativeBrief import build_creative_brief


from ganjiArray import extract_comparison_slices, format_comparison_block, parse_compare_specs
from ganji_converter import Scope

from regress_conversation import ISO_DATE_RE, KOR_ABS_DATE_RE, _db_load, _maybe_override_target_date, _today, ensure_session, record_turn_message, get_extract_chain, build_question_with_regression_context
from converting_time import extract_target_ganji_v2, convert_relative_time, parse_korean_date_safe
from regress_Deixis import _make_bridge, build_regression_and_deixis_context
from sip_e_un_sung import _branch_of, unseong_for, branch_for, pillars_unseong, seun_unseong
from Sipsin import _norm_stem, branch_from_any, get_sipshin, get_ji_sipshin_only, stem_from_any
from choshi_64 import GUA
from ganji_converter import get_ilju, get_wolju_from_date, get_year_ganji_from_json, JSON_PATH
#from langchain.chains import create_extraction_chain
import google.cloud.firestore

from langchain_openai import ChatOpenAI 

from core.services import (
    keyword_category,
    is_fortune_query,
    extract_meta_and_convert,
    make_saju_payload,
    category_to_korean,
    mirror_target_times_to_legacy,
    style_seed_from_payload
)
from prompts.saju_prompts import (
    DEV_MSG,
    counseling_prompt,
    SAJU_COUNSEL_SYSTEM
)


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory

from langchain.chains import LLMChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories  import ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage



# [START all]
# [START import]
# The Cloud Functions for Firebase SDK to create Cloud Functions and set up triggers.
from firebase_functions import firestore_fn, https_fn

# The Firebase Admin SDK to access Cloud Firestore.
from firebase_admin import initialize_app, firestore
import google.cloud.firestore

app = initialize_app()
# [END import]


import functions_framework
from firebase_functions import https_fn, options
import json
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from google.cloud import storage

# 1. Load API Key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
print("✅ OPENAI_API_KEY 로드 완료")

# ============================================================================
# 🔥 In-Memory 중복 추적 (빠른 중복 감지용)
# ============================================================================
# - GCS 로딩 없이 즉시 중복 체크 (0.001초)
# - 동시 요청도 감지 가능
_RECENT_REQUESTS = {}  # {"session:question": {"time": float, "status": str}}

# ============================================================================
# 📦 질문-답변 캐싱 시스템 (성능 최적화)
# ============================================================================
#
# 목적:
#   - 동일/유사 질문에 대해 OpenAI 재호출 없이 캐시된 답변 반환
#   - 응답 시간: 30초 → 1초 이하
#   - Flutter에서 캐시 여부 확인 가능 (UI 표시용)
#
# 구현:
#   - LRU 캐시 (메모리 기반, 최대 1000개)
#   - TTL: 1시간 (3600초)
#   - 질문 정규화: 공백/대소문자 제거 후 해시 생성
#
# 응답 형식:
#   {
#     "answer": "...",
#     "cached": true,           // 캐시 사용 여부 (Flutter UI용)
#     "cache_age_seconds": 120  // 캐시 생성 후 경과 시간
#   }
# ============================================================================

# 캐시 저장소: {question_hash: (answer, timestamp)}
_ANSWER_CACHE: dict[str, Tuple[str, float]] = {}
CACHE_TTL_SECONDS = 3600  # 1시간
CACHE_MAX_SIZE = 1000     # 최대 1000개 질문 캐시

def normalize_question(question: str, session_id: str = "") -> str:
    """
    질문을 정규화하여 캐시 키 생성
    
    Args:
        question: 사용자 질문
        session_id: 세션 ID (선택)
    
    Returns:
        str: 정규화된 질문의 SHA256 해시
    """
    # 공백/줄바꿈 제거, 소문자 변환
    normalized = "".join(question.lower().split())
    # 세션 ID 포함 (같은 질문이라도 세션별로 다른 캐시)
    key = f"{session_id}:{normalized}"
    return hashlib.sha256(key.encode('utf-8')).hexdigest()

def get_cached_answer(question: str, session_id: str = "") -> Optional[Tuple[str, int]]:
    """
    캐시된 답변 조회
    
    Args:
        question: 사용자 질문
        session_id: 세션 ID
    
    Returns:
        Optional[Tuple[str, int]]: (답변, 캐시 생성 후 경과 시간) 또는 None
    """
    cache_key = normalize_question(question, session_id)
    
    if cache_key not in _ANSWER_CACHE:
        return None
    
    answer, timestamp = _ANSWER_CACHE[cache_key]
    age = int(time.time() - timestamp)
    
    # TTL 체크
    if age > CACHE_TTL_SECONDS:
        # 만료된 캐시 삭제
        del _ANSWER_CACHE[cache_key]
        print(f"[CACHE] 만료된 캐시 삭제 (age={age}s)")
        return None
    
    print(f"[CACHE] ✅ 캐시 히트 (age={age}s, key={cache_key[:16]}...)")
    return (answer, age)

def save_to_cache(question: str, answer: str, session_id: str = "") -> None:
    """
    답변을 캐시에 저장
    
    Args:
        question: 사용자 질문
        answer: OpenAI 답변
        session_id: 세션 ID
    """
    global _ANSWER_CACHE
    
    # 캐시 크기 제한 (LRU 방식)
    if len(_ANSWER_CACHE) >= CACHE_MAX_SIZE:
        # 가장 오래된 항목 삭제
        oldest_key = min(_ANSWER_CACHE.keys(), key=lambda k: _ANSWER_CACHE[k][1])
        del _ANSWER_CACHE[oldest_key]
        print(f"[CACHE] 캐시 용량 초과 → 가장 오래된 항목 삭제")
    
    cache_key = normalize_question(question, session_id)
    _ANSWER_CACHE[cache_key] = (answer, time.time())
    print(f"[CACHE] 💾 답변 저장 (key={cache_key[:16]}..., total={len(_ANSWER_CACHE)})")


# ============================================================================
# 2. LLM 정의 (사주 + 점괘 응답용)
# ============================================================================
llm = ChatOpenAI(
    temperature=1.2,
    #model_kwargs={"top_p": 1.0},  # ✅ 이렇게
    top_p=0.9, 
    openai_api_key=openai_key,
    model="gpt-4o-mini",
    timeout=25,
    max_retries=2,
)#"gpt-3.5-turbo" 
print("✅ LLM 초기화 완료")

# ============================================================================
# 3. Memory 저장소 (성능 최적화)
# ============================================================================
# 
# ✅ 최적화 이전:
#    - ConversationSummaryBufferMemory 사용
#    - 매 턴마다 LLM 호출하여 요약 생성 (16초 소요)
#    - 29턴 hydration 시 각 턴마다 요약 업데이트 (71초 소요)
#
# ✅ 최적화 이후:
#    - ChatMessageHistory 사용 (단순 메시지 저장)
#    - LLM 호출 없이 메시지만 저장 (~0초)
#    - OpenAI는 전체 메시지 히스토리를 직접 받아 맥락 이해
#    - 요약 없이도 더 정확한 맥락 제공 (정보 손실 없음)
#
# 📊 성능 개선:
#    - 메모리 저장: 16초 → 0초
#    - Hydration: 71초 → 2초
#    - 총 개선: ~85초 절감
# ============================================================================

global_memory = ChatMessageHistory()
print("✅ Memory 설정 완료 (ChatMessageHistory, 요약 생성 없음)")

# ✅ fortune 전용 프롬프트

# ✅ fortune 전용 프롬프트 및 시스템 설정은 prompts/saju_prompts.py에서 로드됨
print("✅ Prompt & System loaded from modules")




# ============================================================================
# 세션 히스토리 반환 함수 (RunnableWithMessageHistory용)
# ============================================================================
#
# 📌 역할:
#    - RunnableWithMessageHistory가 대화 이력을 가져올 때 호출
#    - 모든 세션이 동일한 전역 메모리 공유 (단일 사용자 가정)
#
# ✅ 최적화:
#    - 이전: global_memory.chat_memory (ConversationSummaryBufferMemory의 내부 객체)
#    - 이후: global_memory (ChatMessageHistory 직접 반환)
# ============================================================================

def get_session_history_func(session_id: str) -> ChatMessageHistory:
    """
    세션 ID에 대한 메시지 히스토리 반환
    
    Args:
        session_id: 세션 식별자 (현재는 사용하지 않음, 전역 메모리 공유)
    
    Returns:
        ChatMessageHistory: 메시지 히스토리 객체
    """
    return global_memory  # ✅ ChatMessageHistory 직접 반환

print("✅ Chain 구성 완료")

# 1. 키워드 기반 카테고리 분류 함수



# ============================================================================
# Hydration 함수 (대화 이력 복원) - 성능 최적화
# ============================================================================
#
# 📌 역할:
#    - GCS/JSON에 저장된 과거 대화 턴을 LangChain 메모리로 로드
#    - 세션당 1회만 실행 (중복 방지)
#
# ⚡ 최적화 전 (71초 소요):
#    - 각 턴마다 global_memory.save_context() 호출
#    - save_context()가 LLM을 호출하여 요약 생성
#    - 29턴 × 2.5초 = 71초
#
# ⚡ 최적화 후 (2초 소요):
#    - 메시지만 history.add_user_message() / add_ai_message()로 추가
#    - LLM 호출 없음
#    - 단순 메모리 추가만 수행
#
# 📊 성능 개선:
#    - 71초 → 2초 (35배 빠름)
#    - 맥락 이해도는 동일 (전체 메시지는 OpenAI가 직접 받음)
# ============================================================================

# 이미 하이드레이션 했는지(중복 방지) 추적
_HYDRATED_SESSIONS: set[str] = set()

def hydrate_history_from_store(session_id: str) -> int:
    """
    per-user JSON에 저장된 turns를 LangChain 히스토리에 주입 (성능 최적화).
    
    Args:
        session_id: 복원할 세션 ID
    
    Returns:
        int: 주입된 턴 수
    
    최적화 세부사항:
        - 요약 생성 제거 (save_context 미사용)
        - 메시지만 히스토리에 추가 (add_user_message / add_ai_message)
        - 중복 주입 방지 (프로세스 생명주기 동안 1회만)
    """
    # ──────────────────────────────────────────────────────────────
    # 1. 중복 방지: 이미 주입된 세션은 스킵
    # ──────────────────────────────────────────────────────────────
    if session_id in _HYDRATED_SESSIONS:
        return 0

    # ──────────────────────────────────────────────────────────────
    # 2. GCS/JSON에서 세션 데이터 로드
    # ──────────────────────────────────────────────────────────────
    try:
        db = _db_load()  # ★ 현재 user 컨텍스트 기반 파일을 로드함
    except Exception as e:
        print(f"[HYDRATE][ERR] load failed: {e}")
        return 0

    sess = (db.get("sessions") or {}).get(session_id)
    if not sess:
        print(f"[HYDRATE] no session '{session_id}' in store")
        _HYDRATED_SESSIONS.add(session_id)  # 없다는 사실도 캐시해 재시도 낭비 방지
        return 0

    turns = list(sess.get("turns") or [])
    if not turns:
        _HYDRATED_SESSIONS.add(session_id)
        return 0

    # ──────────────────────────────────────────────────────────────
    # 3. LangChain 히스토리 객체 획득
    # ──────────────────────────────────────────────────────────────
    history = get_session_history_func({"configurable": {"session_id": session_id}})

    # ──────────────────────────────────────────────────────────────
    # 4. 턴을 메시지로 변환하여 주입 (⚡ 최적화: LLM 호출 없음)
    # ──────────────────────────────────────────────────────────────
    injected = 0
    for t in turns:
        role = (t.get("role") or "").strip().lower()
        text = t.get("text") or ""
        if not text:
            continue

        # ✅ 최적화: add_user_message / add_ai_message만 사용
        # ❌ 제거: global_memory.save_context() (LLM 호출하여 요약 생성)
        if role == "user":
            history.add_user_message(text)
        elif role == "assistant":
            history.add_ai_message(text)
        else:
            # 기타 role은 무시
            continue
        
        injected += 1

    # ──────────────────────────────────────────────────────────────
    # 5. 중복 방지 플래그 설정 및 로그
    # ──────────────────────────────────────────────────────────────
    _HYDRATED_SESSIONS.add(session_id)
    print(f"[HYDRATE] ⚡ injected={injected} turns into session='{session_id}' (요약 없음, LLM 호출 0회)")
    return injected



# ============================================================================
# 요약 텍스트 가져오기 함수 (레거시 호환성)
# ============================================================================
#
# 📌 주의:
#    - ChatMessageHistory는 moving_summary_buffer가 없음
#    - 실제로는 get_session_brief_summary()가 사용됨 (JSON에서 직접 읽기)
#
# ✅ 최적화:
#    - 요약 생성 없음 (LLM 호출 제거)
#    - 필요시 JSON에서 최근 턴만 직접 읽기
# ============================================================================

def get_summary_text() -> str:
    """
    레거시 호환성을 위한 함수 (실제로는 사용되지 않음)
    
    Returns:
        str: 빈 문자열 (ChatMessageHistory에는 요약 기능 없음)
    """
    # ChatMessageHistory에는 moving_summary_buffer가 없음
    # 실제로는 get_session_brief_summary()를 사용하여 JSON에서 직접 읽음
    return ""


def get_session_brief_summary(session_id: str, n: int = 6) -> str:
    db = _db_load()
    sess = (db.get("sessions") or {}).get(session_id) or {}
    turns = sess.get("turns") or []
    return "\n".join(f"{t.get('role','')}: {(t.get('text') or '').strip().replace('\n',' ')}"
                     for t in turns[-n:])



    


# ============================================================================
# 상태 로그 함수 (디버깅용)
# ============================================================================
#
# ✅ 최적화: ChatMessageHistory에 맞게 수정
#    - moving_summary_buffer 제거 (존재하지 않음)
#    - 메시지 수만 출력
# ============================================================================

def print_summary_state():
    """현재 메모리 상태를 한 번에 로그 (성능 최적화 버전)"""
    try:
        msg_count = len(global_memory.messages) if hasattr(global_memory, 'messages') else 0
        print(f"\n🧠 메모리 내 메시지 수: {msg_count}")
    except Exception as e:
        print(f"\n🧠 메모리 상태 확인 실패: {e}")


# ============================================================================
# 대화 턴 기록 함수 (성능 최적화)
# ============================================================================
#
# ⚡ 최적화 전 (16초 소요):
#    - global_memory.save_context() 호출
#    - save_context()가 LLM을 호출하여 요약 업데이트
#
# ⚡ 최적화 후 (0초):
#    - 메모리 조작 없음
#    - JSON 저장은 record_turn_message()에서 처리
#    - LLM 호출 제거
#
# 📊 성능 개선:
#    - 16초 → 0초 (완전 제거)
# ============================================================================

def record_turn(user_text: str, assistant_text: str, payload: dict | None = None):
    """
    대화 1턴 기록 (성능 최적화 버전)
    
    Args:
        user_text: 사용자 메시지
        assistant_text: 어시스턴트 응답
        payload: 추가 메타데이터 (사용하지 않음)
    
    최적화:
        - LangChain 메모리 업데이트 제거 (save_context 미사용)
        - JSON 저장은 record_turn_message()에서 별도 처리
        - LLM 호출 제거로 16초 절감
    
    참고:
        - 실제 저장은 record_turn_message()에서 수행
        - 이 함수는 레거시 호환성을 위해 유지
    """
    # ✅ 최적화: LangChain 메모리 업데이트 제거
    # ❌ 기존: global_memory.save_context() → LLM 호출하여 요약 생성 (16초)
    # ✅ 개선: 아무것도 하지 않음 (JSON 저장은 record_turn_message에서 처리)
    
    # 상태 로그 (옵션)
    print_summary_state()
    print("================== record_turn end (최적화: LLM 호출 없음) ==================\n")

    
    

# 5. Firebase 함수 엔드포인트
@https_fn.on_request(memory=4096, timeout_sec=300)
def ask_saju(req: https_fn.Request) -> https_fn.Response:
    global _RECENT_REQUESTS  # ✅ 전역 변수 선언 (UnboundLocalError 방지)
    _ctx = False
    try:
        print("📥 요청 수신")
        # ✅ JSON 파싱을 안전하게 처리 (빈 요청 또는 잘못된 형식 대응)
        try:
            data = req.get_json(silent=True) or {}
        except Exception as e:
            print(f"[WARN] JSON 파싱 실패: {e}")
            # 요청 본문을 직접 읽어서 확인
            try:
                raw_data = req.get_data(as_text=True)
                print(f"[DEBUG] 요청 본문 (raw): {raw_data[:200] if raw_data else '(empty)'}")
                if raw_data:
                    import json
                    data = json.loads(raw_data)
                else:
                    data = {}
            except Exception as e2:
                print(f"[ERROR] 요청 본문 파싱 실패: {e2}")
                return https_fn.Response(
                    response=json.dumps({
                        "error": "잘못된 요청 형식입니다. JSON 형식으로 요청해주세요.",
                        "detail": str(e2)
                    }, ensure_ascii=False),
                    status=400,
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )
        
        if not isinstance(data, dict):
            return https_fn.Response(
                response=json.dumps({
                    "error": "요청 데이터가 올바른 형식이 아닙니다. JSON 객체를 전송해주세요."
                }, ensure_ascii=False),
                status=400,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
        
        # --- 안전한 입력 파싱 ---
        question = (data.get("question") or "").strip()
        user_name = data.get("name") or ""
        sajuganji = data.get("sajuganji") or {}   # ✅ dict 기본값
        session_id = data.get("session_id") or "single_global_session"
        
        # ✅ [NEW] 모드 구분 (saju / fortune)
        mode = (data.get("mode") or "saju").strip().lower()

        # 사주 원국 기둥 (키 없을 수 있음)
        year  = sajuganji.get("년주", "") or ""
        month = sajuganji.get("월주", "") or ""
        day         = sajuganji.get("일주", "") or ""
        pillar_hour = sajuganji.get("시주", "") or ""      # ❗ time 변수명 피함

        # ✅ [NEW] 대운 정보 (배열 형태 지원)
        daewoon_raw = data.get("daewoon")
        if isinstance(daewoon_raw, list):
            daewoon = daewoon_raw  # 배열 그대로 사용
            daewoon_str = ", ".join(daewoon_raw)  # 로그/표시용 문자열
        else:
            daewoon = daewoon_raw or ""  # 기존 문자열 형태
            daewoon_str = daewoon_raw or ""
        
        current_daewoon = data.get("currentDaewoon") or ""
        
        # ✅ [NEW] 대운 시작 나이
        first_luck_age = data.get("firstLuckAge")
        if first_luck_age is not None:
            try:
                first_luck_age = int(first_luck_age)
            except (ValueError, TypeError):
                first_luck_age = None

        # ✅ 십성 정보 (sipseong_info 객체 또는 개별 필드 지원)
        sipseong_info = data.get("sipseong_info") or {}
        
        # sipseong_info 객체가 있으면 우선 사용, 없으면 기존 개별 필드 사용
        yinYang = sipseong_info.get("yinYang") or data.get("yinYang", "") or ""
        fiveElement = sipseong_info.get("fiveElement") or data.get("fiveElement", "") or ""
        
        # 년간/년지
        yearGan = sipseong_info.get("yearGan") or sipseong_info.get("년주") or data.get("yearGan") or ""
        yearJi  = sipseong_info.get("yearJi") or sipseong_info.get("년주") or data.get("yearJi") or ""
        
        # 월간/월지
        wolGan  = sipseong_info.get("wolGan") or sipseong_info.get("월간") or data.get("wolGan") or ""
        wolJi   = sipseong_info.get("wolJi") or sipseong_info.get("월지") or data.get("wolJi") or ""
        
        # 일간/일지
        ilGan   = sipseong_info.get("ilGan") or sipseong_info.get("일주") or data.get("ilGan") or ""
        ilJi    = sipseong_info.get("ilJi") or sipseong_info.get("일주") or data.get("ilJi") or ""
        
        # 시간/시지
        siGan   = sipseong_info.get("siGan") or sipseong_info.get("시간") or data.get("siGan") or ""
        siJi    = sipseong_info.get("siJi") or sipseong_info.get("시지") or data.get("siJi") or ""
        
        # 대운간/대운지
        currDaewoonGan = sipseong_info.get("currDaewoonGan") or sipseong_info.get("대운간") or data.get("currDaewoonGan", "") or ""
        currDaewoonJi  = sipseong_info.get("currDaewoonJi") or sipseong_info.get("대운지") or data.get("currDaewoonJi", "") or ""
        
        # [ADD] 생년월일(YYYY-MM-DD 또는 YYYYMMDD). 앱에서 'birth' 또는 'birthday' 어느 키든 허용
        user_birth = (data.get("birth") or data.get("birthday") or "").strip()
        
        # [ADD] 앱 UID (새로운 경로 구조용)
        app_uid = (data.get("app_uid") or data.get("appUid") or data.get("uid") or "").strip()
        
        # [NEW] 이 요청 동안만 '해당 사용자' 파일로 라우팅되도록 켠다
        #       (Cloud Run/Functions 재사용 프로세스 대비, 요청 끝나면 반드시 해제)
        
        # ★ 파일명을 'user_name.json'으로 강제하려면:
        user_id = make_user_id_from_name(user_name)    # "홍길동" → "홍길동"        
        set_current_user_context(
            name=user_name,
            birth=user_birth or "19880716",     # 생일은 파일명에 반영하지 않음
            user_id_override=user_id,          # ★ 이름만 파일키로 고정 (프로필 ID로 사용됨)
            app_uid=app_uid,                    # ★ 앱 UID (새 경로 구조용)
        )
        _ctx = True
        
        # ✅ reset 플래그를 유연하게 파싱 (문자/숫자/불리언 모두 허용)
        raw_reset = data.get("reset", False)
        reset_flag = False
        if isinstance(raw_reset, bool):
            reset_flag = raw_reset
        else:
            reset_flag = str(raw_reset).strip().lower() in ("1", "true", "t", "yes", "y")

        print(f"[RESET] raw={raw_reset!r} → flag={reset_flag}")

        if reset_flag:
            # 현재 컨텍스트의 파일을 지운다 (gs://.../<user_id>.json 또는 로컬 파일)
            
            uid = get_current_user_id()
            target_path = _resolve_store_path_for_user(uid) if uid else "(no-uid)"
            ok = delete_current_user_store()
            print(f"[RESET] delete {uid} → {target_path} → ok={ok}")

            # 컨텍스트 정리 후 바로 종료(중요)
            set_current_user_context(reset=True)
            return https_fn.Response(
                response=json.dumps({"reset": bool(ok), "user_id": uid, "path": target_path}, ensure_ascii=False),
                status=200,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
        
        # (옵션) 클라이언트가 'history'만 요청하는 경우
        # ✅ 사용자 컨텍스트가 설정된 후에 처리 (올바른 파일 로드)
        if str(data.get("fetch_history", "")).lower() in ("1","true","yes","y"):
            # 저장소에서 그대로 읽어 반환 (세션 생성/LLM 미실행)
            try:
                db = _db_load()
                sess_id = (data.get("session_id") or "single_global_session")
                sess = (db.get("sessions") or {}).get(sess_id) or {"meta": {"session_id": sess_id}, "turns": []}
                uid = get_current_user_id() or ""   # ← 안전하게 호출
                path = _resolve_store_path_for_user(uid) if uid else "unknown"
                print(f"[FETCH_HISTORY] user_id={uid}, session_id={sess_id}, path={path}, turns={len(sess.get('turns', []))}")
                return https_fn.Response(
                    response=json.dumps(
                        {
                            "user_id": uid,
                            "session_id": sess_id,
                            "path": path,
                            "meta": sess.get("meta") or {},
                            "turns": sess.get("turns") or [],
                        }, ensure_ascii=False
                    ),
                    status=200,
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )
            except Exception as e:
                print(f"[FETCH_HISTORY][ERROR] {e}")
                import traceback
                traceback.print_exc()
                return https_fn.Response(
                    response=json.dumps({
                        "error": f"히스토리 로드 실패: {str(e)}",
                        "user_id": get_current_user_id() or "",
                    }, ensure_ascii=False),
                    status=500,
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )


        # --- 세션 보장 (hydration은 중복 체크 후로 이동) ---
        session_id = data.get("session_id") or "single_global_session"
        session_id = ensure_session(session_id, title="사주 대화")

        # ⭐ [FAST DEDUP] 메모리 기반 중복 체크 (GCS 로딩 전, 0.001초)
        # - 동시 요청, 빠른 재시도 모두 감지
        # - 60초 윈도우
        from time import time
        
        request_key = f"{session_id}:{question}"
        now = time()
        
        # 1) 오래된 항목 정리 (60초 초과)
        _RECENT_REQUESTS = {k: v for k, v in _RECENT_REQUESTS.items() if now - v.get("time", 0) <= 60}
        
        # 2) 중복 체크
        if request_key in _RECENT_REQUESTS:
            last_req = _RECENT_REQUESTS[request_key]
            delta = now - last_req["time"]
            
            if delta <= 60:  # 60초 이내 중복
                status = last_req.get("status", "processing")
                
                if status == "done":
                    # 이미 완료된 요청 → 캐시 반환 (실제론 GCS에서 가져와야 하지만, 여기선 처리중 반환)
                    print(f"[DEDUP-MEMORY] ✅ 중복 감지 (완료됨, {delta:.1f}초 전, hydration skip)")
                    return https_fn.Response(
                        response=json.dumps({
                            "answer": "이전 요청을 처리 완료했습니다. 잠시 후 다시 시도해주세요.",
                            "status": "duplicate_done"
                        }, ensure_ascii=False),
                        status=200,
                        headers={"Content-Type": "application/json; charset=utf-8"}
                    )
                else:
                    # 처리 중인 요청 → 대기 메시지
                    print(f"[DEDUP-MEMORY] ⚠️ 중복 감지 (처리 중, {delta:.1f}초 전, hydration skip)")
                    return https_fn.Response(
                        response=json.dumps({
                            "answer": "이전 요청을 처리 중입니다. 잠시만 기다려주세요.",
                            "status": "processing"
                        }, ensure_ascii=False),
                        status=202,
                        headers={"Content-Type": "application/json; charset=utf-8"}
                    )
        
        # 3) 새 요청 기록
        _RECENT_REQUESTS[request_key] = {"time": now, "status": "processing"}
        print(f"[DEDUP-MEMORY] 새 요청 기록: {request_key[:50]}...")

        # [ENHANCED] 중복 요청 방지 (Client Retry 방어 강화)
        # ⭐ Hydration 전에 먼저 체크 → 중복이면 30초 절약!
        try:
            _db_dedup = _db_load()  # 한 번만 로드
            _sess_dedup = (_db_dedup.get("sessions") or {}).get(session_id) or {}
            _turns_dedup = _sess_dedup.get("turns") or []
            
            # 1) 마지막 User 턴 찾기 (역순 검색)
            last_user_turn = None
            last_asst_turn = None
            for t in reversed(_turns_dedup):
                if t.get("role") == "user" and last_user_turn is None:
                    last_user_turn = t
                elif t.get("role") == "assistant" and last_asst_turn is None:
                    last_asst_turn = t
                if last_user_turn and last_asst_turn:
                    break
            
            if last_user_turn:
                last_q = (last_user_turn.get("text") or "").strip()
                last_q_updated = (last_user_turn.get("updated_question") or "").strip()
                
                # 2) 같은 질문인지 확인 (원본 또는 변환된 질문 비교)
                is_duplicate = (
                    last_q == question or
                    (last_q_updated and last_q_updated == question) or
                    (last_q_updated and last_q == question)
                )
                
                if is_duplicate:
                    # 3) 시간 윈도우 체크 (60초 이내 중복 감지)
                    last_ts_str = last_user_turn.get("ts") or ""
                    try:
                        from datetime import datetime, timedelta, timezone
                        if last_ts_str:
                            if last_ts_str.endswith("+0900"):
                                last_ts_str = last_ts_str[:-5] + "+09:00"
                            last_dt = datetime.fromisoformat(last_ts_str)
                            now_dt = datetime.now(timezone(timedelta(hours=9)))
                            delta_sec = (now_dt - last_dt).total_seconds()
                        else:
                            delta_sec = 0
                    except Exception as te:
                        print(f"[DEDUP] 시간 파싱 실패: {te}")
                        delta_sec = 0
                    
                    # 4) 60초 이내 중복이면 처리
                    if delta_sec <= 60:
                        # 4-1) 이미 응답이 있는가?
                        if (last_asst_turn and 
                            _turns_dedup.index(last_asst_turn) > _turns_dedup.index(last_user_turn)):
                            cached_answer = last_asst_turn.get("text") or ""
                            print(f"[DEDUP] ✅ 중복 감지 (캐시 반환, hydration skip): {delta_sec:.1f}초 전")
                            return https_fn.Response(
                                response=json.dumps({"answer": cached_answer}, ensure_ascii=False),
                                status=200,
                                headers={"Content-Type": "application/json; charset=utf-8"}
                            )
                        else:
                            # 4-2) 응답이 아직 없음 → 처리 중
                            print(f"[DEDUP] ⚠️  중복 감지 (처리 중, hydration skip): {delta_sec:.1f}초 전")
                            return https_fn.Response(
                                response=json.dumps({
                                    "answer": "이전 요청을 처리 중입니다. 잠시만 기다려주세요.",
                                    "status": "processing"
                                }, ensure_ascii=False),
                                status=202,
                                headers={"Content-Type": "application/json; charset=utf-8"}
                            )
                    else:
                        print(f"[DEDUP] ℹ️  동일 질문이지만 시간 초과 ({delta_sec:.1f}초) → 새 요청으로 처리")
                        
        except Exception as e:
            print(f"[DEDUP] 체크 실패 (무시): {e}")
            import traceback
            traceback.print_exc()

        # ⭐ [CACHE CHECK] 캐시된 답변 확인 (OpenAI 호출 전)
        # - 동일 질문 재요청 시 30초 → 1초로 단축
        # - Flutter에서 cached 필드로 UI 차별화 가능
        print(f"[CACHE] 캐시 확인 중... (question={question[:30]}...)")
        cached_result = get_cached_answer(question, session_id)
        if cached_result:
            cached_answer, cache_age = cached_result
            print(f"[CACHE] ✅ 캐시된 답변 반환 (age={cache_age}s, saved ~30s)")
            
            # 컨텍스트 정리 후 응답
            set_current_user_context(reset=True)
            _ctx = False
            
            return https_fn.Response(
                response=json.dumps({
                    "answer": cached_answer,
                    "cached": True,              # ✅ Flutter UI 표시용
                    "cache_age_seconds": cache_age
                }, ensure_ascii=False),
                status=200,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
        else:
            print(f"[CACHE] 캐시 미스 → OpenAI 호출 진행")

        # ⭐ 중복이 아닐 때만 Hydration 실행 (30초 걸림)
        print(f"[HYDRATE] Starting hydration for session={session_id}")
        hydrate_history_from_store(session_id)
                
                
        # ---------- (A) 메타 추출 체인 실행 ----------
        # 프롬프트는 가벼운 템플릿만(외부 I/O 금지)       

        question_for_llm = None       
                    
        # 2. 메타 추출 및 시간 변환: 재사용 가능한 함수로 분리
        #parsed_meta = extract_meta_and_convert(question)
        #updated_question = parsed_meta.get("updated_question", question) #"updated_question" 값이 없다면 원래 질문 "question"을 리턴함
        
        # 2. 메타 추출 및 시간 변환
        parsed_meta, updated_question = extract_meta_and_convert(question)  # ✔ 튜플 언팩

        # updated_question이 비어오면 안전하게 원문으로 폴백
        updated_question = updated_question or parsed_meta.get("updated_question") or question

        print(f"[CRT] abs={parsed_meta.get('absolute_keywords')} / updated='{updated_question}'")
        
        # ============================================================================
        # 📋 Flutter에서 받은 사주 정보 전체 로그
        # ============================================================================
        print("=" * 80)
        print("📥 [FLUTTER 요청 데이터 전체 로그]")
        print("=" * 80)
        print(f"🧑 이름: {user_name}")
        print(f"📅 생년월일: {user_birth}")
        print(f"🆔 앱 UID: {app_uid}")
        print(f"🔑 세션 ID: {session_id}")
        print(f"🎯 모드: {mode}")
        print("-" * 80)
        print(f"🌿 간지 정보 (sajuganji):")
        print(f"   년주: {year}")
        print(f"   월주: {month}")
        print(f"   일주: {day}")
        print(f"   시주: {pillar_hour}")
        print(f"   전체 객체: {json.dumps(sajuganji, ensure_ascii=False)}")
        print("-" * 80)
        print(f"📊 대운 정보:")
        print(f"   대운 배열: {daewoon}")
        print(f"   대운 문자열: {daewoon_str}")
        print(f"   현재 대운: {current_daewoon}")
        print(f"   대운 시작 나이: {first_luck_age}")
        # 나이대별 대운 계산 및 출력 (년도, 십성, 십이운성 포함)
        if isinstance(daewoon, list) and first_luck_age is not None:
            from core.services import calculate_daewoon_by_age, _extract_birth_year
            from Sipsin import _norm_stem
            birth_year = _extract_birth_year(user_birth)
            # 일간 정보 추출 (십성 계산용)
            day_stem_hj = None
            if ilGan:
                try:
                    day_stem_hj = _norm_stem(ilGan)
                except Exception:
                    pass
            daewoon_by_age = calculate_daewoon_by_age(daewoon, first_luck_age, birth_year, day_stem_hj)
            if daewoon_by_age:
                print(f"   나이대별 대운:")
                for item in daewoon_by_age:
                    year_range = item.get('year_range', '')
                    age_range = item.get('age_range', '')
                    daewoon_ganji = item.get('daewoon', '')
                    sipseong = item.get('sipseong', '')
                    sipseong_branch = item.get('sipseong_branch', '')
                    sibi_unseong = item.get('sibi_unseong', '')
                    
                    # 기본 정보
                    if year_range:
                        line = f"     {year_range}년: {age_range}세: {daewoon_ganji}"
                    else:
                        line = f"     {age_range}세: {daewoon_ganji}"
                    
                    # 십성과 십이운성 정보 추가
                    sipseong_parts = []
                    if sipseong:
                        sipseong_parts.append(f"천간 십성={sipseong}")
                    if sipseong_branch:
                        sipseong_parts.append(f"지지 십성={sipseong_branch}")
                    if sibi_unseong:
                        sipseong_parts.append(f"십이운성={sibi_unseong}")
                    
                    if sipseong_parts:
                        line += f" ({', '.join(sipseong_parts)})"
                    
                    print(line)
        print("-" * 80)
        print(f"☯️ 십성 정보:")
        print(f"   음양: {yinYang}")
        print(f"   오행: {fiveElement}")
        print(f"   년간/년지: {yearGan}/{yearJi}")
        print(f"   월간/월지: {wolGan}/{wolJi}")
        print(f"   일간/일지: {ilGan}/{ilJi}")
        print(f"   시간/시지: {siGan}/{siJi}")
        print(f"   대운간/대운지: {currDaewoonGan}/{currDaewoonJi}")
        print(f"   전체 sipseong_info 객체: {json.dumps(sipseong_info, ensure_ascii=False)}")
        print("-" * 80)
        print(f"❓ 질문:")
        print(f"   원본: {question}")
        print(f"   변환 후: {updated_question}")
        print("=" * 80)
        
        {
            # print("===========================테스트 코드 ===============================")

            # pu = pillars_unseong('壬', pillars)
            # print(f"🧪 예시) {ilGan}'壬' 일간에게 2025년 巳(사)는 어떤 운성?")
            # print(f"👉 결과: {unseong_for('임', '사')}")   # '관대'       
            

            # # 2) 내 사주 기둥 운성 일괄
            # pillars = {'year':'辰', 'month':'巳', 'day':'申', 'hour':'酉'}
            # pu = pillars_unseong('壬', pillars)
            # print(f"🧩 기둥 운성: {pu}")
            
            # print(f"🧪 예시) 갑목(甲) 일간의 '제왕' 지지는?")
            # print(f"👉 결과: {branch_for('갑', '제왕')}")   # '묘'

            # # 3) 세운만 빠르게
            # print(f"📆 세운(巳) 운성: {seun_unseong('壬', '巳')}")
            
            # input_date = datetime(1988, 7, 16)  # 예: 양력 2025년 5월 28일
            # year_ganji = get_year_ganji_from_json(input_date, JSON_PATH)
            # print(f"년주: {year_ganji}")

            # wolju_ = get_wolju_from_date(input_date, JSON_PATH)
            # print(f"월주: {wolju_}")

            # ilju_ = get_ilju(input_date, JSON_PATH)
            # print(f"일주: {ilju_}")
            
            # tempDaewoon = data.get("currentDaewoon", "").strip().strip('"')
            # print(f"일간 :{ilju_[0]},  현재 대운 일간 : {tempDaewoon}/{tempDaewoon[0]}")


            # sipshin_result = get_sipshin(ilju_[0], tempDaewoon[0])  # 예: 일간=甲, 타간=丙
            # print(f"'{ilju_[0]}' 기준으로 '{tempDaewoon[0]}'의 십신은 → {sipshin_result}")
            # print(f"십신: {sipshin_result}")  # 결과: 겁재 또는 비견

            # sipshin_Jiresult = get_ji_sipshin_only(ilju_[0], tempDaewoon[1])  # 일간=甲, 지지=午 → 지장간의 마지막 '丁'
            # print(f"'{ilju_[0]}' 기준으로 '{tempDaewoon[1]}'의 십신은 → {sipshin_Jiresult}")
            # print(f"지지 기반 십신: {sipshin_Jiresult}")  # 결과: 편인 (예시)

            print("===============================================================")
        }      

          
        # 0) 세션 먼저 보장
        session_id = ensure_session(session_id, title="사주 대화")

        # ✅ 요약 텍스트 가져오기 (이미 쓰는 전역 메모리 그대로)
        #summary_text = global_memory.moving_summary_buffer or ""
        summary_text = get_summary_text()
        summary_text = get_session_brief_summary(session_id)
        #print(f"summary_text : {summary_text}")
        
        
        # --- 회귀(이전 대화 회수) ---
        # ✅ 회귀 판단 + 맥락 결합 (키워드 리스트 따로 만들 필요 없음)
        reg_prompt, reg_dbg = build_regression_and_deixis_context(
                                        question=updated_question,
                                        summary_text=summary_text,
                                        session_id=session_id,   # ★ 반드시 전달 → [JSON_SCAN] sid=None 방지
                                    )
        print(f"[REG] 최종 회귀 상태: {reg_dbg}")

        #1차 분류
        #category = classify_question(updated_question)
        #print(f"📂 최종 분류 결과: {category}")
        # ──────────────────────────────── fortune(점괘) 분기 ────────────────────────────────
        # ✅ mode가 명시적으로 'fortune'이면 우선 사용, 아니면 키워드 기반 판단
        is_fortune = (mode == "fortune") or is_fortune_query(updated_question)
        print(f"🔮 모드 판단: mode={mode}, is_fortune={is_fortune}")
        
        if is_fortune:
            try:
                # ✅ 최적화: ChatMessageHistory에는 moving_summary_buffer 없음
                # 최근 대화 요약은 get_session_brief_summary()로 대체
                summary_text = get_session_brief_summary(session_id)

                # 1) 본괘/변괘 서로 다르게 선택
                (ben_n, ben_item), (bian_n, bian_item) = GUA.pick_two_random()
                print(f"🎲 본괘 #{ben_n}, 변괘 #{bian_n} 선택")

                # 2) JSON 필드 안전 추출
                def take(item: dict, primary: str, fallbacks: tuple[str, ...] = ()) -> str:
                    for k in (primary, *fallbacks):
                        v = item.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                    return ""

                ben_name_ko     = take(ben_item,  "괘이름_한글")
                ben_name_hanja  = take(ben_item,  "괘이름_한자")
                ben_summary_txt = take(ben_item,  "요지", ("해석", "translate_summary"))
                ben_detail_txt  = take(ben_item,  "풀이")

                bian_name_ko     = take(bian_item, "괘이름_한글")
                bian_name_hanja  = take(bian_item, "괘이름_한자")
                bian_summary_txt = take(bian_item, "요지", ("해석", "translate_summary"))
                bian_detail_txt  = take(bian_item, "풀이")

                # 3) 화면에 고정으로 뿌릴 헤더(LLM이 수정하지 않음)
                fixed_header = (
                    f"[본괘]\n"
                    f"번호: {ben_n}\n"
                    f"이름: {ben_name_ko} ({ben_name_hanja})\n"
                    f"요지: {ben_summary_txt}\n"
                    f"풀이: {ben_detail_txt}\n\n"
                    f"[변괘]\n"
                    f"번호: {bian_n}\n"
                    f"이름: {bian_name_ko} ({bian_name_hanja})\n"
                    f"요지: {bian_summary_txt}\n"
                    f"풀이: {bian_detail_txt}\n\n"
                )

                llm_only_prompt = ChatPromptTemplate.from_template("""
        너는 초씨역림(주역) 보조 해석가다.
        아래 본괘/변괘의 '요지'만 참고해 사용자의 질문에 맞춘 **추가 풀이/조언**만 작성해라.

        규칙:
        - 본괘/변괘의 번호·이름·요지·풀이를 다시 쓰지 마라(화면에 이미 표기됨).
        - 답변은 **질문과 선택된 괘의 요지**에 **직접 연결**한다. 일반론/성향 일반화 금지.
        - [풀이] 섹션 3~6문장.
        - 마지막 줄은 '🔎 포인트: ...' 한 줄 요약.

        [본괘 요지]
        {ben_summary}

        [변괘 요지]
        {bian_summary}

        [대화 요약]
        {summary}

        [사용자 질문]
        {question}
        """)

                base_chain = llm_only_prompt | llm
                chat_with_memory = RunnableWithMessageHistory(
                    base_chain,
                    get_session_history_func,
                    input_messages_key="question",
                    history_messages_key="history"
                )

                result = chat_with_memory.invoke(
                    {
                        "ben_summary": ben_summary_txt,
                        "bian_summary": bian_summary_txt,
                        "summary": summary_text,
                        "question": updated_question,
                    },
                    config={"configurable": {"session_id": session_id}},
                )

                final_text = f"{fixed_header}[풀이]\n{result.content}"

                # ✅ 최적화: 메모리 저장 제거 (LLM 호출 제거)
                # 실제 저장은 JSON 파일에서 처리됨

                
                # 상태 로그
                print_summary_state()
                return https_fn.Response(
                    response=json.dumps({
                        "answer_type": "fortune",
                        "ben_number": ben_n,
                        "bian_number": bian_n,
                        # 필요하면 구조화 필드도 함께 내려주기 좋음
                        "ben": {
                            "number": ben_n,
                            "name_ko": ben_name_ko,
                            "name_hanja": ben_name_hanja,
                            "summary": ben_summary_txt,
                            "detail": ben_detail_txt,
                        },
                        "bian": {
                            "number": bian_n,
                            "name_ko": bian_name_ko,
                            "name_hanja": bian_name_hanja,
                            "summary": bian_summary_txt,
                            "detail": bian_detail_txt,
                        },
                        "answer": final_text,
                    }, ensure_ascii=False),
                    status=200,
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )

            except Exception as e:
                import traceback; traceback.print_exc()
                return https_fn.Response(
                    response=json.dumps({
                        "answer_type": "fortune",
                        "error": f"점괘 처리 중 오류: {str(e)}"
                    }, ensure_ascii=False),
                    status=500,
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )
        else :
            print(f"*******SAJU_COUNSEL_SYSTEM 분기")
            # ✅ 최적화: ChatMessageHistory에는 moving_summary_buffer 없음
            # 최근 대화 요약은 get_session_brief_summary()로 대체
            summary_text = get_session_brief_summary(session_id)

            focus = data.get("focus") or "종합운"


            user_payload = make_saju_payload(data, focus, updated_question)
            # ✅ app_uid를 payload에 추가 (record_turn_message에서 사용)
            if app_uid:
                user_payload["app_uid"] = app_uid
            print(json.dumps(user_payload.get("meta", {}).get("daewoon_by_age"), ensure_ascii=False))
            # → prompt 호출 시 {comparison_block}에 주입

            #비교 블록 만들기
            #    - target_times가 존재하면 우선 사용
            #    - 없으면 legacy(resolved.flow_now.target 또는 target_time)에서 1건이라도 가져와 최소 비교/근거 형태 유지
            try:
                slices = extract_comparison_slices(user_payload)  # 내부에서 payload["target_times"] 우선 사용하도록 구현됨
            except Exception as e:
                print(f"[WARN] extract_comparison_slices 실패: {e}")
                slices = []

            if not slices:
                print("not slices")
                # ---- Fallback: legacy 단일 타겟에서 한 건이라도 꺼내서 최소 정보 구성 ----
                legacy = (user_payload.get("resolved", {})
                                        .get("flow_now", {})
                                        .get("target", {}))
                if not legacy:
                    legacy = user_payload.get("target_time", {}) or {}
                picked = None
                for scope in ("year","month","day","hour"):
                    slot = legacy.get(scope)
                    if slot and any(slot.get(k) for k in ("ganji","sipseong","sipseong_branch","sibi_unseong")):
                        picked = {
                            "label": {"year":"연운","month":"월운","day":"일운","hour":"시운"}.get(scope, scope),
                            "scope": scope,
                            "ganji": slot.get("ganji"),
                            "stem": slot.get("stem"),
                            "branch": slot.get("branch"),
                            "sipseong": slot.get("sipseong"),
                            "sipseong_branch": slot.get("sipseong_branch"),
                            "sibi_unseong": slot.get("sibi_unseong"),
                        }
                        break
                slices = [picked] if picked else []

            # 문자열 블록 (프롬프트에 바로 꽂기)
            comparison_block = format_comparison_block(slices) if slices else ""

            # [NEW] payload에 사용자 정보가 없으면 주입
            if "user" not in user_payload:
                user_payload["user"] = {"name": user_name, "birth": user_birth}
            
            creative_brief = build_creative_brief(user_payload, updated_question)
            style_seed = style_seed_from_payload(user_payload)

            chain = counseling_prompt | ChatOpenAI(
                temperature=1.2, 
                #model_kwargs={"top_p": 0.9},
                top_p = 0.9,
                openai_api_key=openai_key,                
                model="gpt-4o-mini",
                max_tokens=600,
                timeout=20,           # 25초 내 못 받으면 예외
                max_retries=2,        # 재시도 안 함 (지연 방지)
            )

            chat_with_memory = RunnableWithMessageHistory(
                chain,
                get_session_history_func,
                input_messages_key="question",
                history_messages_key="history",
            )
            
            # [중요] 사용자 메시지 기록(+메타 자동추출) — 같은 사용자 파일에 기록됨
            session_id = ensure_session(session_id, title="사주 대화")

            #max_history 결정 (클라이언트에서 보내면 그 값, 아니면 기본값)
            try:
                max_history = int(data.get("max_history") or MAX_TURNS)
            except (TypeError, ValueError):
                max_history = MAX_TURNS
            
            # [중요] 사용자 메시지 기록(+메타 자동추출)
            # [OPTIMIZED] 메타 재사용 (LLM 호출 절약)
            # 이미 extract_meta_and_convert에서 추출한 메타를 그대로 사용합니다.
            meta_reuse = {
                "msg_keywords": parsed_meta.get("msg_keywords"),
                "target_date": parsed_meta.get("target_date"),
                "event_time": parsed_meta.get("time"),  # DB 필드명 매핑 (time -> event_time)
                "kind": parsed_meta.get("kind"),
                "notes": parsed_meta.get("notes"),
                "updated_question": updated_question,  # [DEDUP] 정규화된 질문 저장 (간지 변환 후)
            }

            # [중요] 사용자 메시지 기록(+메타 자동추출)
            record_turn_message(
                session_id=session_id,
                role="user",
                text=question,
                mode="GEN",
                auto_meta=False,   # [FIX] 중복 LLM 호출 방지 (약 3~5초 절약)
                extra_meta=meta_reuse,
                payload=user_payload,
            )
            
            # 회귀 빌더에서 만든 질문(맥락 포함) 사용; 없으면 updated_question
            effective_question = (question_for_llm or parsed_meta.get("updated_question") or updated_question or question)
            bridge_text = _make_bridge(reg_dbg.get("facts", {}))
            facts_json   = json.dumps(reg_dbg.get("facts", {}), ensure_ascii=False)
            
            # ✅ [NEW] 나이대별 대운 정보 포맷팅 (년도, 십성, 십이운성 포함)
            daewoon_by_age = user_payload.get("meta", {}).get("daewoon_by_age", [])
            daewoon_age_text = ""
            if daewoon_by_age:
                daewoon_lines = []
                for item in daewoon_by_age:
                    year_range = item.get("year_range", "")
                    age_range = item.get("age_range", "")
                    daewoon_ganji = item.get("daewoon", "")
                    sipseong = item.get("sipseong", "")
                    sipseong_branch = item.get("sipseong_branch", "")
                    sibi_unseong = item.get("sibi_unseong", "")
                    
                    # 기본 정보
                    if year_range:
                        line = f"  - {year_range}년: {age_range}세: {daewoon_ganji}"
                    else:
                        line = f"  - {age_range}세: {daewoon_ganji}"
                    
                    # 십성과 십이운성 정보 추가
                    sipseong_parts = []
                    if sipseong:
                        sipseong_parts.append(f"천간 십성={sipseong}")
                    if sipseong_branch:
                        sipseong_parts.append(f"지지 십성={sipseong_branch}")
                    if sibi_unseong:
                        sipseong_parts.append(f"십이운성={sibi_unseong}")
                    
                    if sipseong_parts:
                        line += f" ({', '.join(sipseong_parts)})"
                    
                    daewoon_lines.append(line)
                daewoon_age_text = "\n".join(daewoon_lines)
            
            # 나이대별 대운 정보를 context에 추가
            daewoon_context = ""
            if daewoon_age_text:
                daewoon_context = f"\n\n[나이대별 대운 정보]\n{daewoon_age_text}\n"
            
            # comparison_block을 context에 추가 (있으면)
            comparison_context = ""
            if comparison_block:
                comparison_context = f"\n\n[비교 입력]\n{comparison_block}\n"
            
            # context에 나이대별 대운 정보와 comparison_block 추가
            enhanced_context = reg_prompt + daewoon_context + comparison_context
            
            result = chat_with_memory.invoke(
                {
                    "context": enhanced_context,                        # 회귀/컨텍스트 전문 + 나이대별 대운
                    "facts": facts_json,                                # 구조화 FACT
                    "summary": summary_text,                            # moving_summary_buffer
                    "question": effective_question,         # 히스토리 키
                    "bridge": bridge_text,                             # ★ 첫 문장 강제
                    "payload": json.dumps(user_payload, ensure_ascii=False),
                    # ★ 비교 전용 추가 파라미터
                    "comparison_block": comparison_block,               # 사람이 읽을 요약 문자열
                    "target_times": user_payload.get("target_times", []),# 원본 배열(모델이 표/비교 생성용으로 사용)

                    "creative_brief": json.dumps(creative_brief, ensure_ascii=False),  # ★ 추가
                    "style_seed": style_seed, 
                },
                config={"configurable": {"session_id": session_id}},
            )
            answer_text = getattr(result, "content", str(result))
            #print(f"counseling_prompt : {counseling_prompt}")
            #print(f"result: {result}") openAI 응답 출력
            
            # 메모리 저장(옵션)
            record_turn(updated_question, result.content, payload=user_payload)
            
            
            # [중요] 어시스턴트 메시지 기록(메타 추출 불필요)
            record_turn_message(
                session_id=session_id,
                role="assistant",
                text=answer_text,
                mode="SAJU",
                auto_meta=False,
                payload=user_payload,
            )

            # 👇 여기서 세션 히스토리를 max_history 개까지만 유지
            try:
                print(f"[DBG] trim-call: data.max_history={data.get('max_history')} MAX_TURNS={MAX_TURNS} → using max_history={max_history}")
                trimmed = trim_session_history(session_id, max_history)
                if trimmed:
                    print(f"[TRIM] session_id={session_id} 에 대해 히스토리 잘라냄 (max={max_history})")
            except Exception as te:
                print(f"[TRIM] trim_session_history 예외: {te}")
                
            # ⭐ 요청 완료 - 메모리 상태 업데이트
            _req_key = f"{session_id}:{question}"
            if _req_key in _RECENT_REQUESTS:
                _RECENT_REQUESTS[_req_key]["status"] = "done"
                print(f"[DEDUP-MEMORY] 요청 완료 표시")
            
            # ⭐ [CACHE SAVE] 답변을 캐시에 저장 (다음 요청을 위해)
            answer_text = result.content
            save_to_cache(question, answer_text, session_id)
            
            return https_fn.Response(
                response=json.dumps({
                    "answer": answer_text,
                    "cached": False,           # ✅ 새로 생성된 답변
                    "cache_age_seconds": 0     # ✅ 방금 생성
                }, ensure_ascii=False),
                status=200,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        # 상세한 에러 로그 (Stack trace)를 찍는 것이 디버깅에 매우 유용합니다.
        import traceback
        traceback.print_exc() # 에러의 전체 스택 트레이스를 출력합니다.
        return https_fn.Response(
            response=json.dumps({"error": f"서버 처리 중 오류가 발생했습니다: {str(e)}"}),
            status=500,
            headers={"Content-Type": "application/json"}
        )
    finally:
        # [NEW] 이 요청 동안 켜둔 사용자 컨텍스트 해제(프로세스 재사용 대비)
        if _ctx:
            set_current_user_context(reset=True)

# [END askSaju]
# [END all]
