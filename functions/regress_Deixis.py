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
    print(f"[JSON_SCAN] sid={session_id} total_turns={total} scored={len(scored)} picked={len(picked)}")
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

    print(f"[DEIXIS] facts={facts}")
    return facts

# ─────────────────────────────────────────────────────────────
# 메인 빌더(V2): 회귀 + 지시어 FACT 통합
# ─────────────────────────────────────────────────────────────
from typing import Dict, Any, Tuple, List

def build_regression_and_deixis_context(
    question: str,
    summary_text: str,
    *,
    session_id: str,          # ★ 반드시 세션 ID를 외부에서 주입
) -> Tuple[str, dict]:
    """
    (프롬프트, 디버그메타) 를 반환.
    동작 순서:
      1) 세션 히스토리 게이트(첫 턴이면 회귀 False)
      2) LLM 회귀 판정(_llm_detect_regression)  ← 키워드 규칙 X, LLM 판단만 사용
      3) 지시어(이때/거기/그 사람 등) 감지 → JSON에서 시간/장소 앵커 복원 후 FACT 주입
      4) 회귀=True면 JSON에서 실제 과거 턴을 스코어링하여 상위 N개 맥락 선택
      5) FACT/컨텍스트를 붙여 LLM 프롬프트 구성 (없으면 원문 그대로)
    로그 프리픽스:
      [REG][IN]    입력/환경
      [REG][HIST]  히스토리 상태
      [REG][META]  메타 추출 결과
      [REG][LLM]   LLM 회귀 판정
      [REG][DEIX]  지시어 FACT 복원
      [REG][SCAN]  JSON 컨텍스트 스캔/선정
      [REG][OUT]   최종 프롬프트 요약
    """

    # ─────────────────────────────────────────────────────────
    # 내부용: 긴 문자열 로그를 줄여 보기 좋게
    # ─────────────────────────────────────────────────────────
    def _brief(s: str, n: int = 140) -> str:
        if not s:
            return ""
        s = str(s).replace("\n", " ").strip()
        return (s[:n] + "…") if len(s) > n else s

    # 0) 입력 확인
    print(f"[REG][IN] session_id={session_id}")
    print(f"[REG][IN] question='{_brief(question)}'")
    print(f"[REG][IN] summary_text='{_brief(summary_text)}'")

    # 1) 히스토리 게이트: 세션에 저장된 과거 턴이 없으면 회귀 불가
    hist = _get_history_stats(session_id=session_id)
    print(f"[REG][HIST] has_history={hist.get('has_history')} turns={hist.get('history_turns')}")
    if not hist.get("has_history"):
        dbg = {"llm": {"is_regression": False, "confidence": 0.0}, "reason": "first_turn_no_history"}
        print("[REG][HIST] first turn → regression=False (hard gate)")
        return question, dbg

    # 2) 현재 발화 메타(키워드/kind/notes 등): JSON 검색 힌트로만 사용
    try:
        meta_now = _extract_meta(question)
    except Exception as e:
        print(f"[REG][META] exception in _extract_meta: {e}")
        meta_now = {"msg_keywords": [], "kind": None, "notes": ""}
    print(f"[REG][META] now={meta_now}")

    # 3) LLM 회귀 판정
    try:
        reg = _llm_detect_regression(question, summary_text, hist)
    except Exception as e:
        print(f"[REG][LLM] exception in _llm_detect_regression: {e}")
        reg = {"is_regression": False, "confidence": 0.0, "topic_keywords": [], "explicit_markers": [], "reasons": "exception"}
    print(f"[REG][LLM] raw={reg}")

    conf_th = float(os.environ.get("REG_CONF_THRESH", "0.65"))
    is_reg = bool(reg.get("is_regression")) and float(reg.get("confidence", 0.0)) >= conf_th
    print(f"[REG][LLM] decision is_reg={is_reg} (conf={reg.get('confidence')}, th={conf_th}) reason='{_brief(reg.get('reasons',''))}'")

    # 공통 디버그 스켈레톤
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
            "now_kind": meta_now.get("kind"), "scored": 0, "filtered_by_min_sim": 0, "picked": 0,
        },
        "facts": {}
    }

    # 4) 지시어 FACT (회귀 여부와 무관하게 항상 시도)
    try:
        facts = _resolve_deixis_and_make_facts(question, session_id=session_id, meta_now=meta_now)
    except Exception as e:
        print(f"[REG][DEIX] exception in _resolve_deixis_and_make_facts: {e}")
        facts = {}
    if facts:
        debug["facts"].update(facts)
    print(f"[REG][DEIX] facts={facts if facts else '{}'}")

    # 5) 회귀=True면 JSON에서 실제 과거 맥락 선택
    rows_fmt: List[dict] = []
    if is_reg:
        merged_kws = list(set((meta_now.get("msg_keywords") or []) + (reg.get("topic_keywords") or [])))
        print(f"[REG][SCAN] merged_kws={merged_kws} kind={meta_now.get('kind')}")
        try:
            rows_fmt, dbg = _select_context_from_json(
                merged_kws=merged_kws,
                target_kind=meta_now.get("kind"),
                limit_pick=8,
                session_id=session_id,
            )
            debug["search"] = {"context_used": len(rows_fmt), **dbg}
            print(f"[REG][SCAN] context_used={len(rows_fmt)} searched_total={dbg.get('searched_total')} "
                  f"scored={dbg.get('scored')} picked={dbg.get('picked')}")
        except Exception as e:
            print(f"[REG][SCAN] exception in _select_context_from_json: {e}")

    # 6) 프롬프트 구성
    header: List[str] = []
    if is_reg:
        header.append(f"사용자가 과거 대화의 연속을 말하고 있습니다. (회귀 감지: True, 신뢰도={reg.get('confidence',0):.2f})")
        header.append("다음 과거 대화 맥락을 참고하여 자연스럽게 이어 답하세요.")

    # 지시어 FACT 명시(있을 때만)
    if "deixis_anchor_date" in debug["facts"]:
        header.append(f"[FACT] '이때'는 {debug['facts']['deixis_anchor_date']['value']} 을(를) 가리킵니다.")
    if "deixis_anchor_place" in debug["facts"]:
        header.append(f"[FACT] '거기/그곳'은 '{debug['facts']['deixis_anchor_place']['value']}' 을(를) 가리킵니다.")
    if "deixis_person" in debug["facts"]:
        header.append(f"[FACT] '그 사람'은 {debug['facts']['deixis_person']['value']} 을 의미합니다.")

    # 과거 대화 라인업(있으면)
    lines = [f"- [{r.get('date','')} {r.get('time','')}] {r.get('role','')}: {r.get('text','')}" for r in rows_fmt]
    body = "\n".join(header)
    if lines:
        body += ("\n과거 대화:\n" + "\n".join(lines))

    # 최종 프롬프트
    if body.strip():
        prompt = f"{body}\n\n현재 발화: {question}"
        print(f"[REG][OUT] prompt_lines={len(prompt.splitlines())} chars={len(prompt)}")
        # 프롬프트 일부 미리보기
        print(f"[REG][OUT] preview:\n{_brief(prompt, 320)}")
        return prompt, debug

    # 컨텍스트/FACT 아무것도 못 붙였으면 원문 그대로
    print("[REG][OUT] no context/fact → return original question")
    return question, debug

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

    
    
    
from langchain_core.prompts import ChatPromptTemplate

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
