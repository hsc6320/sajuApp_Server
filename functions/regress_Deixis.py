# -*- coding: utf-8 -*-
"""
대화 회귀 + 지시어(이때/거기/그 사람...) 앵커링 통합 빌더 (V2)
- 기존 build_question_with_regression_context 와 이름 충돌 방지: build_regression_and_deixis_context 로 제공
- 핵심 아이디어:
  1) LLM으로 회귀 의도 판정(키워드 규칙 X)
  2) 회귀=True면 conversations.json에서 과거 맥락을 실제로 '선택'
  3) 질문에 지시어가 있으면 회귀 여부와 무관하게 JSON에서 시간/장소 앵커를 복원해 [FACT]로 프롬프트 상단에 주입
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import os, re, json
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from regress_conversation import _extract_meta, _llm_detect_regression, _db_load

# ─────────────────────────────────────────────────────────────
# 외부 제공/기존 함수(이미 프로젝트에 있는 것으로 가정)
# - _db_load(): conversations.json 로드
# - _extract_meta(text): msg_keywords/kind/notes 등 메타 추출(OpenAI 사용)
# - _llm_detect_regression(question, summary_text, hist): 회귀 여부/키워드/이유 등
# - _to_text(): LangChain/OpenAI 응답을 문자열로 정규화
# ※ 없으면 기존 구현 임포트하세요.
# ─────────────────────────────────────────────────────────────
# from your_modules import _db_load, _extract_meta, _llm_detect_regression, _to_text

# ─────────────────────────────────────────────────────────────
# Deixis(지시어) 토큰: 시간/장소/인물
# ─────────────────────────────────────────────────────────────
DEIXIS_TIME_TOKENS = (
    "이때", "그때", "그 날", "그날", "이날", "그즈음", "그 무렵", "그 시기",
)
DEIXIS_PLACE_TOKENS = (
    "그곳", "이곳", "거기", "저기", "그 장소", "그 위치", "그 지역",
    "그 호텔", "그 리조트", "그 카페", "그 식당", "그 여행지", "그 도시", "그 나라",
)
DEIXIS_PERSON_TOKENS = (
    "그 사람", "이 사람", "그분", "그 여자", "그 남자", "그 친구", "그 애",
)

def _has_deixis(q: str) -> bool:
    """질문에 시간/장소/인물 지시어가 하나라도 있으면 True"""
    if not q: return False
    qs = " ".join(str(q).split())
    toks = DEIXIS_TIME_TOKENS + DEIXIS_PLACE_TOKENS + DEIXIS_PERSON_TOKENS
    return any(tok in qs for tok in toks)

# ─────────────────────────────────────────────────────────────
# 세션 히스토리 유무/길이
# ─────────────────────────────────────────────────────────────
def _get_history_stats(*, session_id: str) -> dict:
    """
    현재 세션의 과거 턴 수를 기준으로 '히스토리 존재 여부' 판단.
    - 반드시 session_id를 받아서 sid=None 문제를 원천 차단.
    """
    db = _db_load()
    sess = (db.get("sessions") or {}).get(session_id) or {}
    turns = (sess.get("turns") or [])
    return {"has_history": len(turns) > 0, "history_turns": len(turns)}

# ─────────────────────────────────────────────────────────────
# 과거 맥락 선택: 키워드 Jaccard + kind 보너스
# ─────────────────────────────────────────────────────────────
def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _select_context_from_json(
    *, merged_kws: List[str], target_kind: Optional[str], limit_pick: int, session_id: str
) -> Tuple[List[dict], dict]:
    """
    conversations.json → sessions[session_id].turns에서 최근→과거로 스캔,
    키워드 겹침/Jaccard + kind 일치 보너스로 스코어링하여 상위 N개 픽.
    반환: (LLM 프롬프트용 포맷 리스트, 디버그)
    """
    db = _db_load()
    sess = (db.get("sessions") or {}).get(session_id) or {}
    turns = (sess.get("turns") or [])[:]
    total = len(turns)
    turns.reverse()  # 최신→과거

    now_kws = set([k.strip().lower() for k in (merged_kws or []) if k])
    scored: List[Tuple[float, dict]] = []

    for t in turns:
        prev_kws = set([k.strip().lower() for k in (t.get("msg_keywords") or []) if k])
        s = _jaccard(now_kws, prev_kws)
        if target_kind and (t.get("kind") or "").strip().lower() == target_kind:
            s += 0.15
        if s > 0:
            scored.append((s, t))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [t for _, t in scored[:limit_pick]]

    # 포맷(LLM에 보여줄 단문 라인)
    rows_fmt = [
        {
            "date": t.get("date",""),
            "time": t.get("time",""),
            "role": t.get("role",""),
            "text": (t.get("text") or "").strip().replace("\n"," "),
        } for t in picked
    ]
    dbg = {
        "searched_total": total,
        "now_keywords": list(now_kws),
        "now_kind": target_kind,
        "scored": len(scored),
        "filtered_by_min_sim": len(scored),  # (간단화)
        "picked": len(picked)
    }
    return rows_fmt, dbg

# ─────────────────────────────────────────────────────────────
# 절대날짜(YYYY년 M월 D일) 파싱
# ─────────────────────────────────────────────────────────────
_DATE_KR_RE = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")

def _parse_abs_kr_date(text: str) -> Optional[str]:
    """텍스트에서 'YYYY년 M월 D일' → 'YYYY-MM-DD'"""
    if not text: return None
    m = _DATE_KR_RE.search(text)
    if not m: return None
    y, mth, d = map(int, m.groups())
    try:
        _ = datetime(y, mth, d)  # 유효성 검사
        return f"{y:04d}-{mth:02d}-{d:02d}"
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────────
# 시간 앵커(날짜) 복원
# ─────────────────────────────────────────────────────────────
def _find_temporal_anchor_from_json(
    session_id: str, *, topic_hints: Tuple[str,...] = ("여행","만남")
) -> Tuple[Optional[str], dict]:
    """
    최신→과거로 스캔하며 시간 앵커를 복원:
      1) assistant 텍스트에 절대날짜 + 주제 힌트 → 최고 신뢰
      2) turn.target_date 필드 (user/assistant)
      3) 절대날짜만 있는 텍스트
    """
    db = _db_load()
    sess = (db.get("sessions") or {}).get(session_id) or {}
    turns = (sess.get("turns") or [])[:]
    turns.reverse()

    print(f"[DEIXIS][TIME] scan sid={session_id} n={len(turns)}")
    searched = 0

    for t in turns:
        searched += 1
        role = t.get("role","")
        txt  = (t.get("text") or "")

        d1 = _parse_abs_kr_date(txt)
        if d1 and any(h in txt for h in topic_hints):
            return d1, {"source":"assistant_text" if role=="assistant" else "text_with_hint", "searched":searched}

        td = t.get("target_date")
        if td and any(h in (txt or "") for h in topic_hints + ("여행운","일정","날짜")):
            return td, {"source":"turn.target_date", "searched":searched}

        if d1:
            return d1, {"source":"text_abs_date", "searched":searched}

    return None, {"source":"none", "searched":searched}

# ─────────────────────────────────────────────────────────────
# 장소 앵커(휴리스틱) 복원
# ─────────────────────────────────────────────────────────────
_PLACE_AFTER_HEAD_RE = re.compile(
    r"(?:여행지|장소|위치|도시|국가|호텔|리조트|공원|해변|카페|식당)\s*(?:은|는|이|가|으로|로|에서|에)?\s*([가-힣A-Za-z0-9·\- ]{2,30})"
)
_PLACE_BEFORE_JOSA_RE = re.compile(
    r"([가-힣A-Za-z0-9·\- ]{2,30})(?:에서|으로|로|에)\s*(?:만났|여행|출발|간|왔다|머문|묵었|봤|예약|찍었)"
)

def _extract_place_candidate(text: str) -> Optional[str]:
    """문장에서 장소 단서를 가볍게 추출(휴리스틱)"""
    if not text: return None
    m = _PLACE_AFTER_HEAD_RE.search(text)
    if m: return m.group(1).strip()
    m = _PLACE_BEFORE_JOSA_RE.search(text)
    if m: return m.group(1).strip()
    return None

def _find_place_anchor_from_json(
    session_id: str, *, topic_hints: Tuple[str,...] = ("여행","만남","장소","호텔","카페","도시","국가")
) -> Tuple[Optional[str], dict]:
    """
    최신→과거로 스캔하며 장소 단서를 복원.
    - 구조화 필드가 없다는 전제에서 텍스트 휴리스틱만 사용(가벼움)
    - topic_hints 가 포함된 문장을 우선 채택
    """
    db = _db_load()
    sess = (db.get("sessions") or {}).get(session_id) or {}
    turns = (sess.get("turns") or [])[:]
    turns.reverse()

    print(f"[DEIXIS][PLACE] scan sid={session_id} n={len(turns)}")
    searched = 0
    fallback = None

    for t in turns:
        searched += 1
        txt = (t.get("text") or "")
        cand = _extract_place_candidate(txt)
        if not cand:
            continue
        if any(h in txt for h in topic_hints):
            return cand, {"source":"text_place_with_hint", "searched":searched}
        if not fallback:
            fallback = cand

    if fallback:
        return fallback, {"source":"text_place_fallback", "searched":searched}
    return None, {"source":"none", "searched":searched}

# ─────────────────────────────────────────────────────────────
# Step A: LLM 회귀 판정 (의미 기반, 룰/마커 제거)
# ─────────────────────────────────────────────────────────────

_CONTINUATION_DETECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """너는 "대화 회귀 여부 판정기"다.
아래의 "직전 assistant 답변"과 "현재 user 질문"을 보고,
현재 질문이 직전 답변을 전제로 의미적으로 이어지는 질문인지 판정하라.

규칙:
- 추측 금지. 텍스트 근거가 약하면 False.
- "이전 답변을 전제(그 답변의 결론/내용/선택지/설명을 바탕으로)" 하면 True.
- 완전히 새 주제면 False.
- 출력은 JSON만. 다른 텍스트 금지.

출력 스키마:
{{
  "is_continuation": true/false,
  "confidence": 0.0-1.0,
  "reason": "한 문장"
}}
"""),
    ("user", """직전 assistant 답변:
<<<
{prev_assistant_text}
>>>

현재 user 질문:
<<<
{current_question}
>>>

JSON만 출력.""")
])


def _llm_detect_continuation_v2(question: str, prev_assistant_text: str) -> dict:
    """
    LLM으로 회귀 여부 판정 (의미 기반, 마커/룰 없음)
    
    Args:
        question: 현재 user 질문
        prev_assistant_text: 직전 assistant 답변 (최근 300자)
    
    Returns:
        {
            "is_continuation": bool,
            "confidence": float,
            "reason": str
        }
    """
    try:
        import os
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[REG][STEP-A] OPENAI_API_KEY not set")
            return {
                "is_continuation": False,
                "confidence": 0.0,
                "reason": "api_key_missing"
            }
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=200,
            timeout=15,
            openai_api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        chain = _CONTINUATION_DETECT_PROMPT | llm
        result = chain.invoke({
            "prev_assistant_text": prev_assistant_text[:300],  # 최근 300자만
            "current_question": question
        })
        
        import json
        data = json.loads(result.content if hasattr(result, "content") else str(result))
        
        # 기본값 보정
        data.setdefault("is_continuation", False)
        data.setdefault("confidence", 0.0)
        data.setdefault("reason", "")
        
        return data
        
    except Exception as e:
        print(f"[REG][STEP-A] LLM 판정 실패: {e}")
        return {
            "is_continuation": False,
            "confidence": 0.0,
            "reason": f"exception: {e}"
        }



# ─────────────────────────────────────────────────────────────
# Step B: 이전 결론 LLM 정제 (회귀일 때만, 토픽 키워드 없이)
# ─────────────────────────────────────────────────────────────

_REFINE_CONCLUSIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """너는 대화 요약기다. 새 판단을 만들지 말고, 이미 나온 판단만 정제/요약한다.

🚨 핵심 규칙 (절대 준수):
1. **텍스트에 없는 내용은 만들지 마라.**
2. **가능하면 빈 배열로 두어라.** (불확실하면 비우기)
3. 추정/해석 금지. 명시적으로 나온 결론만 추출.

출력 JSON:
{{
  "decisions": ["이전에 도출된 핵심 결론 (명확한 것만)"],
  "key_points": ["중요 판단 요지"],
  "open_questions": ["아직 답 안 된 핵심 질문"],
  "constraints": ["보수적 접근", "리스크 회피" 등 조건/제약],
  "confidence": 0.0~1.0
}}

confidence 가이드:
- 명확한 결론이 여러 개 → 0.8~1.0
- 일부 결론만 명확 → 0.5~0.7
- 애매하거나 추정 필요 → 0.3 이하
- 결론 없음 → 0.0 (빈 배열)
"""),
    ("user", """최근 답변들:
{assistant_messages}

현재 질문:
{current_question}

JSON만 출력.""")
])

def _refine_conclusions_with_llm(rows_fmt: List[dict], current_question: str) -> dict:
    """
    LLM으로 이전 결론 정제 (토픽 키워드 없이 의미 기반)
    
    Returns:
        {
            "decisions": [...],
            "key_points": [...],
            "open_questions": [...],
            "constraints": [...],
            "confidence": 0.0~1.0
        }
    """
    if not rows_fmt:
        return {"decisions": [], "key_points": [], "open_questions": [], "constraints": [], "confidence": 0.0}
    
    # 최근 assistant 답변 2~4개만 (비용 최소화)
    assistant_msgs = [r for r in rows_fmt if r.get("role") == "assistant"][-4:]
    if not assistant_msgs:
        return {"decisions": [], "key_points": [], "open_questions": [], "constraints": [], "confidence": 0.0}
    
    assistant_text = "\n\n".join([f"[답변{i+1}] {m.get('text', '')[:200]}" for i, m in enumerate(assistant_msgs)])
    
    try:
        # LLM 호출
        import os
        from langchain_openai import ChatOpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("[REFINE] OPENAI_API_KEY not set")
            return {"decisions": [], "key_points": [], "open_questions": [], "constraints": [], "confidence": 0.0}
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=400,
            timeout=15,
            openai_api_key=api_key,
            model_kwargs={"response_format": {"type": "json_object"}}
        )
        
        chain = _REFINE_CONCLUSIONS_PROMPT | llm
        result = chain.invoke({
            "assistant_messages": assistant_text,
            "current_question": current_question
        })
        
        import json
        data = json.loads(result.content if hasattr(result, "content") else str(result))
        
        # 기본값 보정
        data.setdefault("decisions", [])
        data.setdefault("key_points", [])
        data.setdefault("open_questions", [])
        data.setdefault("constraints", [])
        data.setdefault("confidence", 0.0)
        
        return data
        
    except Exception as e:
        print(f"[REFINE] LLM 정제 실패: {e}")
        return {"decisions": [], "key_points": [], "open_questions": [], "constraints": [], "confidence": 0.0}


# ─────────────────────────────────────────────────────────────
# 지시어 해석 → FACT 생성
# ─────────────────────────────────────────────────────────────
def _resolve_deixis_and_make_facts(question: str, *, session_id: str, meta_now: dict) -> dict:
    """
    질문에 지시어가 있으면:
      - 시간 앵커(날짜)
      - 장소 앵커(장소명)
      - 인물 지시 고정("이때 만난 사람" → 해당 시점/장소의 만남)
    을 FACT로 구성해 반환.
    """
    facts: dict = {}
    if not _has_deixis(question):
        return facts

    hints = tuple(meta_now.get("msg_keywords") or []) + ("여행","만남","일정","장소","호텔","도시")

    # 시간
    anchor_date, tdbg = _find_temporal_anchor_from_json(session_id, topic_hints=hints)
    if anchor_date:
        facts["deixis_anchor_date"] = {"value": anchor_date, "source": tdbg.get("source")}

    # 장소
    anchor_place, pdbg = _find_place_anchor_from_json(session_id, topic_hints=hints)
    if anchor_place:
        facts["deixis_anchor_place"] = {"value": anchor_place, "source": pdbg.get("source")}

    # 인물(사람 이름이 없으므로 "그 시점/장소의 만남"으로 고정)
    q = question
    if any(tok in q for tok in DEIXIS_PERSON_TOKENS) or "만난" in q or "만남" in q:
        val = "해당 시점의 만남(최근 대화)"
        if anchor_date and anchor_place:
            val = f"{anchor_date} {anchor_place}에서의 만남(최근 대화)"
        elif anchor_date:
            val = f"{anchor_date}의 만남(최근 대화)"
        elif anchor_place:
            val = f"{anchor_place}에서의 만남(최근 대화)"
        facts["deixis_person"] = {"value": val, "source": "inferred_from_anchor"}
        
    print(f"[DEIXIS] facts={facts}")        #
    return facts

# ─────────────────────────────────────────────────────────────
# 메인 빌더(V2): 회귀 + 지시어 FACT 통합
# ─────────────────────────────────────────────────────────────
from typing import Dict, Any, Tuple, List

def build_regression_and_deixis_context(
    question: str,
    summary_text: str,
    *,
    session_id: str,
) -> Tuple[str, dict]:
    """
    🎯 하이브리드 회귀 처리 (Rule + LLM 정제)
    
    Pipeline:
      Step A: Rule-based 회귀 판정 (대화 구조만)
      Step B: LLM 이전 결론 정제 (의미 기반, 토픽 키워드 ❌)
      Step C: 조건부 memory_summary 주입 (confidence ≥ 임계치)
    
    Returns:
        (프롬프트, 디버그메타)
    """
    def _brief(s: str, n: int = 140) -> str:
        if not s:
            return ""
        s = str(s).replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s
    
    print(f"[REG][IN] session_id={session_id}")
    print(f"[REG][IN] question='{_brief(question)}'")
    
    # ─────────────────────────────────────────────────────────
    # 1) 히스토리 게이트 (첫 턴이면 회귀 불가)
    # ─────────────────────────────────────────────────────────
    hist = _get_history_stats(session_id=session_id)
    if not hist.get("has_history"):
        dbg = {"step": "A", "is_continuation": False, "confidence": 0.0, "reason": "first_turn"}
        return question, dbg
    
    # ─────────────────────────────────────────────────────────
    # Step A: LLM 회귀 판정 (의미 기반)
    # ─────────────────────────────────────────────────────────
    # 직전 assistant 답변 가져오기
    from regress_conversation import _db_load
    db = _db_load()
    sessions = db.get("sessions") or {}
    sess = sessions.get(session_id) or {}
    turns = list(sess.get("turns") or [])
    
    prev_assistant_text = ""
    for t in reversed(turns):
        if t.get("role") == "assistant":
            prev_assistant_text = t.get("text", "")[:500]  # 최근 500자
            break
    
    continuation_result = _llm_detect_continuation_v2(question, prev_assistant_text)
    print(f"[REG][STEP-A] is_continuation={continuation_result['is_continuation']} "
          f"confidence={continuation_result['confidence']:.2f} "
          f"reason='{continuation_result['reason']}'")
    
    CONTINUATION_THRESHOLD = 0.75  # ✅ 보수적 임계치
    
    # 회귀 아니면 즉시 종료
    if not continuation_result["is_continuation"] or continuation_result["confidence"] < CONTINUATION_THRESHOLD:
        dbg = {
            "step": "A",
            "is_continuation": continuation_result["is_continuation"],
            "confidence": continuation_result["confidence"],
            "reason": continuation_result["reason"],
            "below_threshold": continuation_result["confidence"] < CONTINUATION_THRESHOLD
        }
        return question, dbg
    
    # ─────────────────────────────────────────────────────────
    # Step B: LLM 이전 결론 정제 (회귀일 때만 호출)
    # ─────────────────────────────────────────────────────────
    # 최근 대화 컨텍스트 가져오기
    from regress_conversation import _extract_meta
    meta_now = _extract_meta(question)
    
    # 과거 맥락 검색 (키워드 기반, 상위 4개만)
    merged_kws = meta_now.get("msg_keywords", [])
    try:
        rows_fmt, scan_dbg = _select_context_from_json(
            merged_kws=merged_kws,
            target_kind=meta_now.get("kind"),
            limit_pick=4,  # 비용 최소화
            session_id=session_id
        )
    except Exception as e:
        print(f"[REG][STEP-B] context scan failed: {e}")
        rows_fmt, scan_dbg = [], {}
    
    # LLM 정제
    refined = _refine_conclusions_with_llm(rows_fmt, question)
    decisions_count = len(refined.get("decisions", []))
    print(f"[REG][STEP-B] refined_confidence={refined['confidence']:.2f} decisions_count={decisions_count}")
    
    # ─────────────────────────────────────────────────────────
    # Step C: 조건부 주입 (보수적)
    # ─────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD = 0.6
    
    if refined["confidence"] < CONFIDENCE_THRESHOLD:
        print(f"[REG][STEP-C] injected=False reason=low_confidence ({refined['confidence']:.2f} < {CONFIDENCE_THRESHOLD})")
        dbg = {
            "step": "C_skipped",
            "is_continuation": True,
            "llm_confidence": continuation_result["confidence"],
            "llm_reason": continuation_result["reason"],
            "refined_confidence": refined["confidence"],
            "decisions_count": decisions_count,
            "injected": False,
            "reason": "low_confidence"
        }
        return question, dbg
    
    # ─────────────────────────────────────────────────────────
    # 프롬프트 구성 (최소화, 자연스러운 문체)
    # ─────────────────────────────────────────────────────────
    header = []
    header.append(f"사용자가 이전 대화를 이어서 질문하고 있습니다. (회귀 신뢰도={continuation_result['confidence']:.2f})")
    
    # memory_summary (1~3줄 제한, 태그 형식 피하기)
    context_lines = []
    if refined.get("decisions"):
        decisions_text = ", ".join(refined['decisions'][:2])
        context_lines.append(f"이전 대화 요약: {decisions_text}")
    
    if refined.get("open_questions") and len(context_lines) < 3:
        context_lines.append(f"현재 질문은 이전 결론을 전제로 함: {refined['open_questions'][0]}")
    
    if refined.get("constraints") and len(context_lines) < 3:
        constraints_text = ", ".join(refined['constraints'][:2])
        context_lines.append(f"참고: {constraints_text}")
    
    # ✅ 최대 3줄까지만 주입
    if context_lines:
        header.extend(context_lines[:3])
    
    # 지시어 FACT (있으면)
    try:
        facts = _resolve_deixis_and_make_facts(question, session_id=session_id, meta_now=meta_now)
    except Exception as e:
        print(f"[REG][DEIXIS] failed: {e}")
        facts = {}
    
    if "deixis_anchor_date" in facts:
        header.append(f"[FACT] '이때'는 {facts['deixis_anchor_date']['value']}")
    
    # 과거 대화 라인업 (간략, 최대 3개)
    lines = [f"- {r.get('role','')}: {r.get('text','')[:80]}..." for r in rows_fmt[:3]]
    
    body = "\n".join(header)
    if lines:
        body += f"\n\n과거 대화 요약:\n" + "\n".join(lines)
    
    prompt = f"{body}\n\n현재 질문: {question}"
    
    print(f"[REG][STEP-C] injected=True prompt_length={len(prompt)} chars")
    
    dbg = {
        "step": "C_injected",
        "is_continuation": True,
        "llm_confidence": continuation_result["confidence"],
        "llm_reason": continuation_result["reason"],
        "refined_confidence": refined["confidence"],
        "decisions_count": decisions_count,
        "injected": True,
        "reason": "confidence_ok",
        "refined": refined,
        "facts": facts
    }
    
    return prompt, dbg


# ─────────────────────────────────────────────────────────────
# 브릿지 텍스트 생성 (기존 호환)
# ─────────────────────────────────────────────────────────────
def _make_bridge(facts: dict | None) -> str:
    """회귀 시 내부 참고 메모. 답변에 노출 금지 가정."""
    facts = facts or {}
    bits = []

    d = (facts.get("deixis_anchor_date") or {}).get("value")
    if d:
        bits.append(f"사용자 질문의 '그날/이때'는 {d}를 가리킴")

    trip = (facts.get("trip_date") or {}).get("value")
    if trip and trip != d:
        bits.append(f"최근 회수된 여행 날짜는 {trip}")

    # 사람/장소 같은 추가 팩트가 있으면 같은 방식으로 붙이세요.
    # place = (facts.get("place") or {}).get("value")
    # if place: bits.append(f"여행 장소: {place}")

    return " / ".join(bits)  # ← '이어서...' 같은 서두 없음

    
    
    
# ChatPromptTemplate는 이미 상단에서 import됨

counseling_prompt = ChatPromptTemplate.from_messages([
    ("system", """너는 맥락을 정확히 이어주는 한국어 사주 상담가다.

출력 원칙(매우 중요):
- 반드시 첫 문장은 그대로 출력한다: "{bridge}"
- 아래 [FACTS]의 정보가 있으면, 첫 1~2문장에 자연스럽게 명시하라(날짜/장소/그 사람 등).
- [CONTEXT]의 과거 대화와 현재 질문을 연결해 '맥락 브릿지'를 만든 뒤, 그 맥락에서만 해석하라.
- 질문 범위 밖의 주제(예: '결혼' 등)로 확장하지 마라. 과장/예언/단정 어투 금지.
- 문체: 따뜻하고 차분, 4~7문장. 마지막에 '🔎 포인트:' 한 줄 요약.

금지:
- 근거 없이 다른 주제(결혼, 승진 등)로 비약하기.
- [CONTEXT]에 없는 사실을 단정하기.
- 중복된 일반론 나열.

"""),
    ("user", """
[CONTEXT]
{context}

[FACTS]
{facts}

[요약]
{summary}

[사용자 질문]
{user_question}
""")
])
