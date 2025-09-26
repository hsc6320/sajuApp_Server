from curses import meta
from datetime import date, datetime
import logging
import os
import json 
from dotenv import load_dotenv

from conv_store import (
    set_current_user_context,
    make_user_id_from_name,
    delete_current_user_store,
    get_current_user_id,
    _resolve_store_path_for_user
)

from regress_conversation import ISO_DATE_RE, KOR_ABS_DATE_RE, _db_load, _maybe_override_target_date, _today, ensure_session, record_turn_message, get_extract_chain, build_question_with_regression_context
from converting_time import extract_target_ganji_v2, convert_relative_time
from regress_Deixis import _make_bridge, build_regression_and_deixis_context
from sip_e_un_sung import _branch_of, unseong_for, branch_for, pillars_unseong, seun_unseong
from Sipsin import _norm_stem, branch_from_any, get_sipshin, get_ji_sipshin_only, stem_from_any
from choshi_64 import GUA
from ganji_converter import get_ilju, get_wolju_from_date, get_year_ganji_from_json, JSON_PATH
from langchain.chains import create_extraction_chain
import google.cloud.firestore

from langchain_openai import ChatOpenAI 


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

# [START addMessage]
# [START addMessageTrigger]
@https_fn.on_request()
def addmessage(req: https_fn.Request) -> https_fn.Response:
    """Take the text parameter passed to this HTTP endpoint and insert it into
    a new document in the messages collection."""
# [END addMessageTrigger]
    # Grab the text parameter.
    original = req.args.get("text")
    if original is None:
        return https_fn.Response("No text parameter provided", status=400)

    # [START adminSdkPush]
    firestore_client: google.cloud.firestore.Client = firestore.client()

    # Push the new message into Cloud Firestore using the Firebase Admin SDK.
    _, doc_ref = firestore_client.collection("messages").add({"original": original})

    # Send back a message that we've successfully written the message
    return https_fn.Response(f"Message with ID {doc_ref.id} added.")
    # [END adminSdkPush]
# [END addMessage]



# [START makeUppercase]
@firestore_fn.on_document_created(document="messages/{pushId}")
def makeuppercase(event: firestore_fn.Event[firestore_fn.DocumentSnapshot | None]) -> None:
    """Listens for new documents to be added to /messages. If the document has
    an "original" field, creates an "uppercase" field containg the contents of
    "original" in upper case."""

    # Get the value of "original" if it exists.
    if event.data is None:
        return
    try:
        original = event.data.get("original")
    except KeyError:
        # No "original" field, so do nothing.
        return

    # Set the "uppercase" field.
    print(f"Uppercasing {event.params['pushId']}: {original}")
    upper = original.upper()
    event.data.reference.update({"uppercase": upper})
# [END makeUppercase]
# [END all]

import functions_framework
from firebase_functions import https_fn, options
import json
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from google.cloud import storage

# 1. Load API Key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
print("✅ OPENAI_API_KEY 로드 완료")


# 2. LLM 정의  (사주 + 점괘 응답용)
llm = ChatOpenAI(
    temperature=0.7,
    model_kwargs={"top_p": 1.0},  # ✅ 이렇게
    openai_api_key=openai_key,
    model="gpt-4o-mini",
    timeout=25,
    max_retries=2,
)#"gpt-3.5-turbo" 
print("✅ LLM 초기화 완료")

# 3. Memory 저장소 (글로벌 메모리 사용)
# ConversationSummaryBufferMemory는 내부적으로 ChatMessageHistory를 가집니다.
global_memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="history", # 이 history_messages_key와 일치해야 합니다.
    max_token_limit=1000,    # 적절한 요약 토큰 제한
    max_tokens=220,         # 또는 max_output_tokens
    return_messages=True, # 메시지 리스트로 반환
    verbose=True
)
print("✅ Memory 설정 완료")

# ✅ fortune 전용 프롬프트

DEV_MSG = """
너는 사주명리학 상담가다. 아래 JSON(이미 계산된 값)만 신뢰해 해석한다.

[JSON 필드 정의]
- saju: 출생 원국 간지 {year, month, day, hour}
- natal.sipseong_by_pillar: 원국 각 기둥의 십성
- current_daewoon: 현재 대운(큰 흐름)
  · ganji: 현재 대운 간지
  · sipseong: (선택) 대운의 십성
  · sibi_unseong: (선택) 대운의 십이운성
- target_time: 이번 질문의 관측 시점(세운/월운)
  · year: {ganji, sipseong, sibi_unseong}
  · month: {ganji, sipseong, sibi_unseong}
- focus: 질문 초점 (직업/재물/연애/건강/일운/월운/세운 등)

[규칙]
- 새 간지/십신/운성 생성 금지, 내부 재계산 금지, JSON 외 추정 금지.
- focus에 맞춰 해석하며, 답변은 자유롭게 하되 반드시 십신(주제 직결: 재성/관성/인성/비겁/식상)과
  십이운성(시기 흐름/강약/심리적 기세)을 모두 1회 이상 명시적으로 반영하라.
- 말투는 따뜻하고 현실적. 사전식 정의/장황한 이론 금지.

[상황에 따라 효율적으로 사용할 출력 구조]
1) 핵심 흐름: 2~3문장 요약(십신·십이운성 각각 최소 1회 언급)
2) 기회: • 불릿 2~3개 (실제 상황 예시 포함)
3) 실행 팁: • 불릿 2~3개 (구체 행동)
4) 주의점: • 불릿 1~2개 (오해/리스크)
5) 한 줄 정리: 1문장
"""


#counseling_prompt = ChatPromptTemplate.from_template("""
SAJU_COUNSEL_SYSTEM = """
너는 따뜻하고 현실적인 조언가다. 입력 JSON(이미 계산된 값)과 질문만 신뢰해 답한다.

[입력 JSON]
- saju: 출생 원국 간지 {year, month, day, hour}
- natal.sipseong_by_pillar: 원국 각 기둥 십성(예: year, month, day, hour)
- current_daewoon: 현재 대운 {ganji, sipseong?, sibi_unseong?}
- target_time: 관측 시점(연/월/일/시) {ganji, sipseong?, sibi_unseong?}
- resolved: (서버에서 정규화해 제공하는 읽기 전용 섹션)
  - pillars.{year|month|day|hour}: {ganji, stem, branch, sipseong, sibi_unseong?}
  - flow_now.daewoon: {ganji, sipseong?, sibi_unseong?}
  - flow_now.target.{year|month|day|hour}: {ganji, sipseong?, sibi_unseong?}
  - canon:
    - sipseong_vocab: 허용 십성 라벨 집합
    - sibi_vocab: 허용 십이운성 라벨 집합
- meta: {focus?, question, summary}

[모드 자동 판별(상호 배타)]
- SAJU: 질문이 사주/간지/대운/세운/월운/일운/십성 등 해석 의도를 띠거나, saju 데이터가 주어져 있고 질문이 운세·시점·재물·직업 등 “해석 대상”일 때.
- COUNSEL: 고민/감정/일반적 조언·정보 요청이며, 사주 해석이 적절치 않거나 JSON 근거가 부족할 때(또는 사용자가 해석이 아닌 조언을 원할 때).
- LOOKUP(간단 조회): “내 사주 간지 알려줘/보여줘”처럼 값 확인만 요청.

[금지]
- JSON에 없는 간지/대운/세운/십성/십이운성 새로 계산·추정·생성 금지.
- COUNSEL/LOOKUP 모드에서는 사주 이론(간지·십성·운성·대운 등) 언급 금지.

[데이터 사용 규약 - 매우 중요]
- 계산/추정 금지. 입력 JSON만 사용.
- 십성(sipseong) 읽기 우선순위:
  1) $.resolved.flow_now.target.{year|month|day|hour}.sipseong
  2) $.current_daewoon.sipseong
  3) $.resolved.pillars.{year|month|day|hour}.sipseong
  (없으면 "데이터 없음"이라고 쓰고 넘어간다.)
- 십이운성(sibi_unseong) 읽기 우선순위:
  1) $.resolved.flow_now.target.{year|month|day|hour}.sibi_unseong
  2) $.current_daewoon.sibi_unseong
  3) $.resolved.pillars.{year|month|day|hour}.sibi_unseong
  (없으면 "데이터 없음"이라고 쓰고 넘어간다.)
- 간지(stem/branch) 언급은 입력 값만 사용. 새로 계산 금지.
- 용어 고정: 십성/십신 혼용 금지 → 항상 “십성”. 12운성은 “십이운성/운성” 표현 사용.
- 라벨 집합 고정: 아래 집합 밖 단어 사용 금지.
  - 십성: $.resolved.canon.sipseong_vocab
  - 십이운성: $.resolved.canon.sibi_vocab
  
[출력 형식]
- 첫 줄에 모드 태그를 반드시 출력: [MODE: SAJU] | [MODE: COUNSEL] | [MODE: LOOKUP]
- SAJU:
  1) 핵심 흐름(2~3문장; 십신·십이운성 각 1회 이상 언급)
  2) 기회 • 2~3개
  3) 실행 팁 • 2~3개
  4) 주의점 • 1~2개
  5) 한 줄 정리(1문장)
  6) 근거
     - (참조 경로와 값을 2~4줄, 실제 사용한 것만 명시. 예시)
     - 십성(월운): $.resolved.flow_now.target.month.sipseong = "편재"
     - 12운성(월운): $.resolved.flow_now.target.month.sibi_unseong = "건록"
     - 대운 십성: $.current_daewoon.sipseong = "편인"
- COUNSEL: 공감 1문장 + 현실적 제안 2~3문장(사주 언급 금지)
- LOOKUP: 요청한 값만 간단히 나열(불필요한 해석 없음)

[어조]
- 따뜻하고 현실적. 장황한 이론 설명 금지. 결과를 바로 쓸 수 있게 간결하게.

[대화 맥락 연결]
- {summary}를 확인하고 직전 응답이 SAJU 모드였다면, 후속 질문이 일상적/가벼워도 사주 해석 맥락과 연결해 자연스럽게 이어서 답한다.
- COUNSEL 모드로 보내야 하는 질문이어도, 사주 기반 조언을 부드럽게 덧붙일 수 있다면 함께 제공해라.

# ───────── 맥락 강화 규칙(추가) ─────────
- 반드시 **첫 문장**은 그대로 출력한다: "{bridge}"
- 아래 [FACTS]의 정보(날짜/장소/인물 등)가 있으면 **첫 1~2문장**에 자연스럽게 명시해라.
- [CONTEXT]의 과거 대화에 근거해 '맥락 브릿지'를 만든 뒤, 그 범위를 벗어나 **새 대주제(결혼/승진/재물 등)로 비약하지 마라**.
- [CONTEXT]에 없는 사실을 단정하지 마라. 중복 일반론 나열 금지.


[대화 요약]
{summary}

[사용자 질문]
{question}
"""

SAJU_COUNSEL_SYSTEM = SAJU_COUNSEL_SYSTEM + """

[표현 금지 / 시작 규칙]
- 첫 문장은 바로 **핵심 요약**으로 시작. 불필요한 서두 금지.
- '출생 원국 년주/월주/일주/시주'는 참조용으로만 사용하고, 제목/첫 문장/첫 문단에는 절대 넣지 말 것. (반드시 타겟 시점 기준으로 작성)
- 아래 [CONTEXT]/[FACTS]/[BRIDGE]는 참고용으로만 사용하고, **문구를 그대로 답변에 쓰지 말 것**.
"""

counseling_prompt = ChatPromptTemplate.from_messages([
    # 기존 시스템 규칙
    SystemMessage(content=SAJU_COUNSEL_SYSTEM),

    # 기존 요약 주입(유지)
    ("system", "이전 대화 요약:\n{summary}"),

    #  🔥브릿지/컨텍스트/팩트 사용 규칙 (문구 복붙 금지 + 비약 금지 재강조)
    ("system",
     '출력 규칙(중요):\n'
     '- 반드시 **첫 문장**은 그대로 출력한다: "{bridge}" (bridge가 비어 있으면 생략)\n'
     '- [FACTS]에 날짜/장소/인물 등이 있으면 **첫 1~2문장**에 자연스럽게 명시한다.\n'
     '- [CONTEXT] 범위를 벗어나 새로운 대주제로 비약하지 말 것.\n'
     '- [CONTEXT]에 없는 사실을 단정하지 말 것. 중복 일반론 나열 금지.\n'
     '- 답변 마지막에 반드시 "근거" 블록을 포함한다.'
    ),

    # 기존 휴먼 입력에 CONTEXT/FACTS/JSON/질문만 확장
    ("human",
     "[CONTEXT]\n{context}\n\n"
     "[FACTS]\n{facts}\n\n"
     "[입력 데이터(JSON)]\n{payload}\n\n"
     "[사용자 질문]\n{question}"
    ),
])

print("✅ Prompt 정의 완료")


# 세션 ID에 따라 다른 ChatMessageHistory 객체를 반환하는 함수
# 현재는 `global_memory`를 반환하므로, 어떤 session_id가 들어와도 동일한 메모리를 참조합니다.
# 사용자 요구사항에 맞게 전역 메모리를 반환하도록 유지합니다.
def get_session_history_func(session_id: str) -> ChatMessageHistory:
   # print(f"🔄 세션 '{session_id}'의 기록 가져오기 (전역 메모리 사용)")
    return global_memory.chat_memory

print("✅ Chain 구성 완료")

# 1. 키워드 기반 카테고리 분류 함수
def keyword_category(question: str) -> str | None:
    keyword_map = {
        "saju": ["사주", "팔자", "대운", "십신", "지장간", "운세", "명리", "일주", "시주"],
        "fortune": ["초씨역림", "점괘", "점", "괘", "육효", "점치다", "괘상", "효"],
        "life_decision": ["이직", "퇴사", "사업", "진로", "선택", "결단", "도전", "변화", "창업"],
        "relationship": ["연애", "결혼", "이혼", "짝사랑", "소개팅", "헤어짐", "재회", "궁합"],
        "self_reflection": ["나", "내가", "자아", "성격", "성향", "고민", "불안", "혼란", "위로"],
        "timing": ["언제", "시기", "올해", "내년", "몇월", "좋은날", "기회", "시점"],
        "academic": ["학업", "시험", "성적", "공부", "수능", "입시"],
        "job": ["취업", "면접", "합격", "지원", "이력서"],
    }
    for category, keywords in keyword_map.items():
        if any(k in question for k in keywords):
            return category
    return None


# 4. 영어 → 한글 매핑
# ① 카테고리별 기본 focus (사용자가 data["focus"]로 덮어쓸 수 있음)
category_to_korean = {
    "saju": "사주",
    "fortune": "초씨역림",
    "life_decision": "인생 결정",
    "relationship": "연애/인간관계",
    "self_reflection": "자기 성찰",
    "timing": "시기 판단",
    "academic": "학업",
    "job": "취업",
    "etc": "기타"
}
valid_categories = category_to_korean.keys()



llm2 = ChatOpenAI(temperature=0.2)
schema = {
    "properties": {
        "대상_시간표현": {
            "type": "array",
            "items": {"type": "string"},
            "description": "질문에서 언급된 대상 시점. 상대적/절대적 표현 모두 포함됨 (예: '올해', '내후년', '오늘','2026년', '11월' 등등)"
        },
        "간지_명시여부": {
            "type": "string",
            "description": "질문에 병오년, 을해일 등의 간지가 명시되어 있으면 '예', 없으면 '아니오'"
        }
    },
    "required": ["대상_시간표현"]
}
#ext_chain = create_extraction_chain(schema=schema, llm=llm2)

# ====== 메모리/히스토리 하이드레이터 ======

# 이미 하이드레이션 했는지(중복 방지) 추적
_HYDRATED_SESSIONS: set[str] = set()

def hydrate_history_from_store(session_id: str) -> int:
    """
    per-user JSON에 저장된 turns를 LangChain 히스토리/요약 메모리에 주입.
    - 중복 주입 방지: 동일 session_id는 프로세스 생명주기 동안 1회만 수행
    - 반환: 주입된 턴 수
    """
    # 중복 방지
    if session_id in _HYDRATED_SESSIONS:
        return 0

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

    # 1) LangChain의 메시지 히스토리 객체 획득
    history = get_session_history_func({"configurable": {"session_id": session_id}})

    # 2) 턴을 그대로 “리플레이”
    injected = 0
    last_user_text = None
    for t in turns:
        role = (t.get("role") or "").strip().lower()
        text = t.get("text") or ""
        if not text:
            continue

        if role == "user":
            history.add_user_message(text)
            last_user_text = text
        elif role == "assistant":
            history.add_ai_message(text)
            # 3) (선택) 요약 메모리에도 반영
            try:
                # 사용자→어시스턴트 페어일 때만 요약에 넣는게 깔끔
                global_memory.save_context({"input": last_user_text or ""}, {"output": text})
                _ = global_memory.load_memory_variables({})  # 요약 즉시 업데이트
            except Exception:
                pass
        else:
            # 기타 role은 무시
            pass
        injected += 1

    _HYDRATED_SESSIONS.add(session_id)
    print(f"[HYDRATE] injected={injected} turns into session='{session_id}'")
    return injected


# ✅ 전용 FACTS 요약 슬롯: global_memory.facts_summary
def get_summary_text() -> str:
    try:
        return getattr(global_memory, "moving_summary_buffer", "") or ""
    except Exception:
        return ""

def get_session_brief_summary(session_id: str, n: int = 6) -> str:
    db = _db_load()
    sess = (db.get("sessions") or {}).get(session_id) or {}
    turns = sess.get("turns") or []
    return "\n".join(f"{t.get('role','')}: {(t.get('text') or '').strip().replace('\n',' ')}"
                     for t in turns[-n:])


def set_summary_text(text: str) -> None:
    """요약을 메모리에만 저장(교체)."""
    
    safe = text or ""
    try:
        # ConversationSummaryBufferMemory가 가진 정식 필드만 사용
        global_memory.moving_summary_buffer = safe
    except Exception as e:
        print(f"[summary] moving_summary_buffer set 실패: {e}")
    

def print_summary_state():
    """현재 요약/버퍼 상태를 한 번에 로그"""
    print("\n🧠 현재 global_memory.moving_summary_buffer (요약) 내용:")
    print(f"메모리 내 메시지 수: {len(global_memory.chat_memory.messages)}")
    #print(f"현재 토큰 수 (추정): {len(str(global_memory.chat_memory.messages)) // 4}")
    print(f"요약 버퍼 내용: {global_memory.moving_summary_buffer}")

def record_turn(user_text: str, assistant_text: str, payload: dict | None = None): 
    """대화 1턴 저장 + LangChain 요약 갱신 + FACTS 병합""" 
    
    #1) LangChain 메모리에 원문 저장(요약 자동 업데이트 트리거) 
    try: 
        global_memory.save_context({"input": user_text}, {"output": assistant_text}) 
        _ = global_memory.load_memory_variables({}) 
    except Exception as e: 
        print(f"[memory] save_context 실패: {e}") 

    # 3) 상태 로그 
    print_summary_state()
    print("================== record_turn end ==================\n")
    
    
def _sipseong_for_target(day_stem_hj: str, target_ganji: str | None) -> str | None:
    if not target_ganji:
        return None
    t_stem_hj   = stem_from_any(target_ganji)    # '戊' 등
    t_branch_hj = branch_from_any(target_ganji)  # '戌' 등
    if t_stem_hj:
        v = get_sipshin(day_stem_hj, t_stem_hj)
        return None if v == "미정" else v
    if t_branch_hj:
        v = get_ji_sipshin_only(day_stem_hj, t_branch_hj)
        return None if v in ("미정", "없음") else v
    return None

def make_saju_payload(data: dict, focus: str, updated_question: str) -> dict:
    """
    요청 data에서 사주 관련 정보를 추출해 표준 스키마(JSON)로 변환
    - 입력: data(dict), focus(str), updated_question(str)
    - 출력: payload(dict)
    """
    # 기본 정보 (기본값 안전화)
    question   = data.get("question", "") or ""
    user_name  = data.get("name", "") or ""
    sajuganji  = data.get("sajuganji") or {}          # ❗ 기본값은 dict
    daewoon    = data.get("daewoon", "") or ""
    current_dw = data.get("currentDaewoon", "") or "" # 문자열/간지표현일 수 있음
    session_id = data.get("session_id") or "single_global_session"  # 필요 시 요청에서 받기

    # 사주 원국 기둥 (키가 없을 수 있으니 dict.get 사용)
    year        = sajuganji.get("년주", "") or ""
    month       = sajuganji.get("월주", "") or ""
    day         = sajuganji.get("일주", "") or ""
    pillar_hour = sajuganji.get("시주", "") or ""      # ❗ time 변수명 피함

    # 십성 참고 정보 (없을 수 있음)
    yinYang        = data.get("yinYang", "") or ""
    fiveElement    = data.get("fiveElement", "") or ""
    yearGan        = data.get("yearGan", "") or ""
    yearJi         = data.get("yearJi", "") or ""
    wolGan         = data.get("wolGan", "") or ""
    wolJi          = data.get("wolJi", "") or ""
    ilGan          = data.get("ilGan", "") or ""
    ilJi           = data.get("ilJi", "") or ""
    siGan          = data.get("siGan", "") or ""
    siJi           = data.get("siJi", "") or ""
    currDwGan      = data.get("currDaewoonGan", "") or ""
    currDwJi       = data.get("currDaewoonJi", "") or ""

    # 질문에서 타겟 간지 추출 (에러 가드)
    try:
        t_year_ganji, t_month_ganji, t_day_ganji, t_hour_ganji = extract_target_ganji_v2(updated_question)
    except Exception as e:
        print(f"[make_saju_payload] ⚠️ extract_target_ganji_v2 실패: {e}")
        t_year_ganji = t_month_ganji = t_day_ganji = t_hour_ganji = None

    print(
        f"[make_saju_payload] 🎯 타겟 간지 → "
        f"year={t_year_ganji}, month={t_month_ganji}, day={t_day_ganji}, hour={t_hour_ganji}"        
    )

    # 요약/엔티티 단계에서 쉽게 이용하도록 표준화
    target_ganji_list = [g for g in [t_year_ganji, t_month_ganji, t_day_ganji, t_hour_ganji] if g]

    # 현재 대운 보조 필드: 없으면 None
    curr_dw_sipseong = f"{currDwGan}/{currDwJi}" if (currDwGan or currDwJi) else None
    #print(f"target_ganji_list :{target_ganji_list}, curr_dw_sipseong : {curr_dw_sipseong}")   

    
    # === 1) 타겟 간지 파싱 후, pillars_unseong로 일괄 계산 ===
    pillars_branches = {
        "year":  branch_from_any(t_year_ganji),
        "month": branch_from_any(t_month_ganji),
        "day":   branch_from_any(t_day_ganji),
        "hour":  branch_from_any(t_hour_ganji),
    }

    #  일간(천간) 표준화: 한글/혼합 → 한자(예: '임'→'壬') (★)
    day_stem_hj = _norm_stem(ilGan)  # ilGan 예: '임' 또는 '壬'

    # None이 섞여 있어도 pillars_unseong 내부에서 처리됨
    target_sibi_map = pillars_unseong(day_stem_hj, pillars_branches)
    # 예: {'year': '관대', 'month': '절', 'day': None, 'hour': '장생'}

    # === [B] 현재 대운 십이운성 (★ _branch_of → branch_from_any)
    current_dw_branch = branch_from_any(current_dw)  # 예: '亥' 또는 None
    curr_dw_sibi = unseong_for(day_stem_hj, current_dw_branch) if current_dw_branch else None
    
    print(f"day_stem_hj : {day_stem_hj}, current_dw_branch : {current_dw_branch}, curr_dw_sibi : {curr_dw_sibi}")
    
    # 타겟(연/월/일/시) 십성 산출 (★ 추가)
    year_sip  = _sipseong_for_target(day_stem_hj, t_year_ganji)
    month_sip = _sipseong_for_target(day_stem_hj, t_month_ganji)
    day_sip   = _sipseong_for_target(day_stem_hj, t_day_ganji)
    hour_sip  = _sipseong_for_target(day_stem_hj, t_hour_ganji)
    print(f"year_sip : {year_sip}, month_sip : {month_sip}, day_sip : {day_sip}")
    # 따옴표 오류 수정(내부 키는 작은따옴표로)
    print(f"target_sibi_map.get(year/month/day) : {target_sibi_map.get('year')}, {target_sibi_map.get('month')}, {target_sibi_map.get('day')}")

    
    # 최종 스키마 구성
    payload = {
        "saju": {
            "year": year,
            "month": month,
            "day": day,          # NOTE: 필요 시 일간/일지 분리 구조로 확장 가능
            "hour": pillar_hour
        },
        "natal": {
            "sipseong_by_pillar": {
                "year": yearGan or None,
                "month": wolGan or None,
                "day": ilGan or None,
                "hour": siGan or None,
            }
        },
        "current_daewoon": {
            "ganji": current_dw or None,          # 문자열(예: '辛酉'), 없으면 None
            "sipseong": curr_dw_sipseong,        # "간/지" 조합, 없으면 None
            "sibi_unseong": curr_dw_sibi,          # 계산된 12운성 (없으면 None)
        },
        "target_time": {
            "year":  {
                "ganji": t_year_ganji,
                "sipseong": year_sip,
                "sibi_unseong": target_sibi_map.get("year")   # ← ✅ 연 운성
            },
            "month": {
                "ganji": t_month_ganji,
                "sipseong": month_sip,
                "sibi_unseong": target_sibi_map.get("month")  # ← ✅ 월 운성
            },
            "day":   {
                "ganji": t_day_ganji,
                "sipseong": day_sip,
                "sibi_unseong": target_sibi_map.get("day")    # ← ✅ 일 운성
            },
            "hour":  {
                "ganji": t_hour_ganji,
                "sipseong": hour_sip,
                "sibi_unseong": target_sibi_map.get("hour")   # ← ✅ 시 운성
            },
        },
        "focus": focus,
        "meta": {
            "user_name": user_name,
            "daewoon": daewoon,
            "yinYang": yinYang,
            "fiveElement": fiveElement,
            "session_id": session_id,           # 필요 시 상위에서 실제 세션 주입
            "question": question,
            # 🔥 요약 엔진에서 바로 읽어갈 수 있는 엔티티 블록
            "entities": {
                "간지": target_ganji_list,
                "타겟_연도": t_year_ganji,
                "타겟_월": t_month_ganji,
                "타겟_일": t_day_ganji,
                "타겟_시": t_hour_ganji,
                "키워드": [],     # 별도 키워드 추출기로 채울 예정이라면 유지
                "이벤트": []
            }
        }        
    }
    
     # === E) 정규화 블록(resolved) 추가: 모델은 여기만 보면 됨 ===
    def _stem(g):   return g[0] if isinstance(g, str) and len(g) >= 1 else None
    def _branch(g): return g[1] if isinstance(g, str) and len(g) >= 2 else None

    resolved_pillars = {
        "year":  {"ganji": year or None,  "stem": _stem(year),  "branch": _branch(year),  "sipseong": None, "sibi_unseong": None},
        "month": {"ganji": month or None, "stem": _stem(month), "branch": _branch(month), "sipseong": None, "sibi_unseong": None},
        "day":   {"ganji": day or None,   "stem": _stem(day),   "branch": _branch(day),   "sipseong": None, "sibi_unseong": None},
        "hour":  {"ganji": pillar_hour or None, "stem": _stem(pillar_hour), "branch": _branch(pillar_hour), "sipseong": None, "sibi_unseong": None},
    }

    payload["resolved"] = {
        "pillars": resolved_pillars,
        "flow_now": {
            "daewoon": {
                "ganji": current_dw or None,
                "sipseong": None,             # 현재 대운의 십성 라벨이 없으므로 None
                "sibi_unseong": curr_dw_sibi, # 계산된 12운성
            },
            "target": {
                "year":  {"ganji": t_year_ganji,  "sipseong": year_sip,  "sibi_unseong": target_sibi_map.get("year")},
                "month": {"ganji": t_month_ganji, "sipseong": month_sip, "sibi_unseong": target_sibi_map.get("month")},
                "day":   {"ganji": t_day_ganji,   "sipseong": day_sip,   "sibi_unseong": target_sibi_map.get("day")},
                "hour":  {"ganji": t_hour_ganji,  "sipseong": hour_sip,  "sibi_unseong": target_sibi_map.get("hour")},
            }
        },
        # 앱 표준에 맞춰 필요 시 조정
        "canon": {
            "sipseong_vocab": ["비견","겁재","식신","상관","편재","정재","편관","정관","편인","정인","본체(편인)"],
            "sibi_vocab": ["양","욕","대극","건록","제왕","쇠","병","사","묘","절","태","양(재생)"]
        }
    }
    

    return payload


FORTUNE_KEYS = ["초씨역림", "주역", "점괘", "괘", "육효", "괘상", "점쳐", "점치"]

def is_fortune_query(text: str) -> bool:
    t = (text or "").strip()
    return any(k in t for k in FORTUNE_KEYS)


# # --- (B) 메타 추출 및 시간 변환 로직 함수 ---
def extract_meta_and_convert(question: str) -> tuple[dict, str]:
    """메타 추출 + (프롬프트는 그대로) 상대시간 → 절대/간지 치환까지 한 번에.
    반환: (parsed_meta(dict), updated_question(str))
    """
    # 1) LLM 메타 추출
    parsed: dict = {}
    extract_chain = get_extract_chain()
    if not extract_chain:
        print("[META] skip: OPENAI_API_KEY not set")
        parsed = {}
    else:
        try:
            ext_res = extract_chain.invoke({"text": question})
            raw = ext_res.content if hasattr(ext_res, "content") else str(ext_res)
            parsed = json.loads(raw)
            print(f"[META] JSON 파싱 성공: {parsed}")
        except Exception as e:
            print(f"[META] 예외 → 빈 메타 사용: {e}")
            parsed = {}

    # 2) 누락 보정
    parsed.setdefault("msg_keywords", [])
    parsed.setdefault("target_date", None)
    parsed.setdefault("time", None)
    parsed.setdefault("kind", None)
    parsed.setdefault("notes", "")
    parsed.setdefault("_facts", {})

    # 3) target_date 보강(프롬프트 수정 없이 여기서만 처리)
    #    - LLM이 넣어주면 그대로 둠
    #    - 없으면 질문에서 ISO 또는 한글 절대일 추출
    print(f"parsed[\"target_date\"] : {parsed["target_date"]}")
    if not parsed["target_date"]:
        m_iso = ISO_DATE_RE.search(question)
        if m_iso:
            parsed["target_date"] = m_iso.group(0)
            parsed["_facts"]["deixis_anchor_date"] = {
                "value": parsed["target_date"], "source": "iso_in_text"
            }
            print(f"[DEIXIS] ISO 날짜 감지 → target_date={parsed['target_date']}")
        else:
            now = _today()
            print(f"[TIME] today={now.isoformat()}")
            m_kor = KOR_ABS_DATE_RE.search(question)
            if m_kor:
                mm, dd = int(m_kor.group(1)), int(m_kor.group(2))
                # 연도 추정은 필요한 정책으로 보강하세요(올해 기준 등)
                yyyy = now.year
                try:
                    parsed["target_date"] = date(yyyy, mm, dd).isoformat()
                    parsed["_facts"]["deixis_anchor_date"] = {
                        "value": parsed["target_date"], "source": "korean_abs"
                    }
                    print(f"[DEIXIS] 한글 절대일 감지 → target_date={parsed['target_date']}")
                except Exception as e:
                    print(f"[DEIXIS] 한글 절대일 보정 실패: {e}")

    # 4) 상대시간 치환: expressions에 **항상 질문 원문을 포함**
    #    (이 한 줄이 핵심입니다)
    today = _today()
    cy, cm, cd = today.year, today.month, today.day
    # msg_keywords + 질문 원문(중복 제거)
    expressions = list(dict.fromkeys((parsed.get("msg_keywords") or []) + [question]))
    _maybe_override_target_date(question, parsed, today)
    
    
    try:
        abs_kws, updated_q = convert_relative_time(
            question=question,
            expressions=expressions,   # ← 여기!
            current_year=cy,
            current_month=cm,
            current_day=cd,
        )
    except Exception as e:
        print(f"[CRT] convert_relative_time 예외: {e}")
        abs_kws, updated_q = (parsed.get("msg_keywords") or []), question

    parsed["absolute_keywords"] = abs_kws
    parsed["updated_question"] = updated_q
    
    return parsed, updated_q


# 5. Firebase 함수 엔드포인트
@https_fn.on_request(memory=4096, timeout_sec=120)
def ask_saju(req: https_fn.Request) -> https_fn.Response:
    _ctx = False
    try:
        print("📥 요청 수신")
        data = req.get_json()       
        # --- 안전한 입력 파싱 ---
        question = (data.get("question") or "").strip()
        user_name = data.get("name") or ""
        sajuganji = data.get("sajuganji") or {}   # ✅ dict 기본값
        daewoon = data.get("daewoon") or ""
        current_daewoon = data.get("currentDaewoon") or ""
        session_id = data.get("session_id") or "single_global_session"

        # 사주 원국 기둥 (키 없을 수 있음)
        year  = sajuganji.get("년주", "") or ""
        month = sajuganji.get("월주", "") or ""

        # 십성 참고
        yinYang = data.get("yinYang", "") or ""
        fiveElement = data.get("fiveElement", "") or ""
        yearGan = data.get("yearGan") or ""
        yearJi  = data.get("yearJi") or ""
        wolGan  = data.get("wolGan") or ""
        wolJi   = data.get("wolJi") or ""
        ilGan   = data.get("ilGan") or ""
        ilJi    = data.get("ilJi") or ""
        siGan   = data.get("siGan") or ""
        siJi    = data.get("siJi") or ""
        currDaewoonGan = data.get("currDaewoonGan", "") or ""
        currDaewoonJi  = data.get("currDaewoonJi", "")  or ""
        
        # [ADD] 생년월일(YYYY-MM-DD 또는 YYYYMMDD). 앱에서 'birth' 또는 'birthday' 어느 키든 허용
        user_birth = (data.get("birth") or data.get("birthday") or "").strip()
        
        # [NEW] 이 요청 동안만 '해당 사용자' 파일로 라우팅되도록 켠다
        #       (Cloud Run/Functions 재사용 프로세스 대비, 요청 끝나면 반드시 해제)
        
        # ★ 파일명을 'user_name.json'으로 강제하려면:
        user_id = make_user_id_from_name(user_name)    # "홍길동" → "홍길동"        
        set_current_user_context(
            name=user_name,
            birth="19880716",                  # 생일은 파일명에 반영하지 않음
            user_id_override=user_id,    # ★ 이름만 파일키로 고정
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
        if str(data.get("fetch_history", "")).lower() in ("1","true","yes","y"):
            # 저장소에서 그대로 읽어 반환 (세션 생성/LLM 미실행)
            db = _db_load()
            sess_id = (data.get("session_id") or "single_global_session")
            sess = (db.get("sessions") or {}).get(sess_id) or {"meta": {"session_id": sess_id}, "turns": []}
            uid = get_current_user_id() or ""   # ← 안전하게 호출
            return https_fn.Response(
                response=json.dumps(
                    {
                        "user_id": uid,
                        "session_id": sess_id,
                        "meta": sess.get("meta") or {},
                        "turns": sess.get("turns") or [],
                    }, ensure_ascii=False
                ),
                status=200,
                headers={"Content-Type": "application/json; charset=utf-8"}
            )

        # --- 세션 보장 & 하이드레이션(앱 재실행 시 과거 대화 복원) ---
        session_id = data.get("session_id") or "single_global_session"
        session_id = ensure_session(session_id, title="사주 대화")
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
        

        print(f"🧑 이름: {user_name}, 🌿 간지: {sajuganji}, 📊 대운: {daewoon}, 현재: {current_daewoon}")
        print(f"십성정보 : 년간 {yearGan}/{yearJi} 월간{wolGan}/{wolJi} 일간{ilGan}/{ilJi} 대운{currDaewoonGan}/{currDaewoonJi}")
        print(f"년주: {year} 월주: {month}")
        print(f"❓ 질문: {question} {updated_question}")
        
        
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
        
         # 현재 연도/월 기준으로 변환
       
        {
        # current_year = datetime.now().year
        # current_month = datetime.now().month
        # current_day = datetime.now().day
        # print(f"오늘 날짜 : {current_year} : {current_month} : {current_day}")
                
        # if not parsed:
        #     absolute_keywords, updated_question = convert_relative_time(question, parsed, current_year, current_month, current_day)
        #     print(f"사용자 입력 키워드: {parsed} ")
        # print(f"변환된 키워드: {absolute_keywords}")
        # print(f"🟡 갱신된 질문: {updated_question}")
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
        #if category == "fortune":
        if is_fortune_query(updated_question):
            try:
                summary_text = global_memory.moving_summary_buffer or ""

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

                # 메모리 저장(옵션)
                try:
                    global_memory.save_context({"input": updated_question},
                                            {"output": final_text})
                    _ = global_memory.load_memory_variables({})   # ✅ 요약 즉시 업데이트
                except Exception:
                    pass

                
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
            summary_text = global_memory.moving_summary_buffer or ""

            # 기존 데이터에서 테스트용 스키마 구성 (target_time 값 있으면 채워서 전달)
            # 병오/기축 테스트가 필요한 경우, req JSON에 아래 키들을 함께 넣어보면 됨:
            # target_year_ganji, target_year_sipseong, target_year_unseong,
            # target_month_ganji, target_month_sipseong, target_month_unseong
            focus = data.get("focus") or "종합운"

            # ── 사용자 페이로드 구성 (3인자 버전 권장) ─────────────────────────────
            # make_saju_payload 시그니처가 4인자(absolute_keywords 포함)라면 여기에 absolute_keywords를 추가하거나,           
            user_payload = make_saju_payload(data, focus, updated_question)

            # [NEW] payload에 사용자 정보가 없으면 주입
            if "user" not in user_payload:
                user_payload["user"] = {"name": user_name, "birth": user_birth}
                
            chain = counseling_prompt | ChatOpenAI(
                temperature=0.6, 
                #model_kwargs={"top_p": 0.9},
                top_p = 0.9,
                openai_api_key=openai_key,                
                model="gpt-4o-mini",
                max_tokens=400,
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
            
            # [중요] 사용자 메시지 기록(+메타 자동추출)
            record_turn_message(
                session_id=session_id,
                role="user",
                text=question,
                mode="GEN",
                auto_meta=True,
                extra_meta={
                    # 간지 결과 등 추가 필드가 있다면 여기에 붙이세요(없으면 생략)
                    # "ganji": {"year": "...", "month": "...", "day": "...", "hour": "..."}
                },
                payload=user_payload,
            )
            
            # 회귀 빌더에서 만든 질문(맥락 포함) 사용; 없으면 updated_question
            effective_question = (question_for_llm or parsed_meta.get("updated_question") or updated_question or question)
            bridge_text = _make_bridge(reg_dbg.get("facts", {}))
            facts_json   = json.dumps(reg_dbg.get("facts", {}), ensure_ascii=False)            
          
            result = chat_with_memory.invoke(
                {
                    "context": reg_prompt,                              # 회귀/컨텍스트 전문
                    "facts": facts_json,                                # 구조화 FACT
                    "summary": summary_text,                            # moving_summary_buffer
                    "question": effective_question,         # 히스토리 키
                    "bridge": bridge_text,                              # ★ 첫 문장 강제
                    "payload": json.dumps(user_payload, ensure_ascii=False),
                },
                config={"configurable": {"session_id": session_id}},
            )
            answer_text = getattr(result, "content", str(result))
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
            
            return https_fn.Response(
                response=json.dumps({"answer": result.content}, ensure_ascii=False),
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
