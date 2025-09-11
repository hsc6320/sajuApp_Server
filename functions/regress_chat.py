# ================== 회귀(이전 대화 회귀) 감지/맥락 결합 ==================
import json
import os
from typing import Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from chat_json_store import _extract_meta, format_search_results, search_messages
from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

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
{"is_regression": <true|false>, "confidence": 0.0~1.0, "topic_keywords": [], "explicit_markers": [], "reasons": "짧은 이유"}
"""),
    ("user", "has_history={has_history}\nhistory_turns={history_turns}\n\n## summary\n{summary}\n\n## utterance\n{question}\n\nJSON만 출력.")
])

def _get_regression_chain():
    model = os.environ.get("REG_MODEL", "gpt-4o-mini")
    temp = float(os.environ.get("REG_TEMPERATURE", "0.0"))
    print(f"[CHAIN] 회귀판정 체인 초기화: model={model}, temp={temp}")
    
    llm = ChatOpenAI(
        model=model,
        temperature=temp,
        max_tokens=350,
        timeout=45,
        max_retries=2,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    return _REG_PROMPT | llm

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

_REG_CHAIN = _get_regression_chain()

# def _llm_detect_regression(current_text: str, summary_text: str) -> dict:
#     """
#     LLM으로 회귀 여부 감지. JSON만 기대.
#     실패 시 안전 폴백 반환.
#     """
#     try:
#         raw = _REG_CHAIN.run(text=current_text, summary=summary_text).strip()
#         s, e = raw.find("{"), raw.rfind("}")
#         payload = raw[s:e+1] if (s!=-1 and e!=-1) else raw
#         out = json.loads(payload)
#         # 방어적 기본값
#         return {
#             "is_regression": bool(out.get("is_regression", False)),
#             "topic_keywords": out.get("topic_keywords") or []
#         }
#     except Exception as e:
#         print(f"[REG] LLM 회귀 감지 실패 → 폴백 (err={e})")
#         return {"is_regression": False, "topic_keywords": []}

def _llm_detect_regression(question: str, summary_text: str, hist: dict) -> dict:
    try:
        print("_llm_detect_regression() Q :{question} : {summary_text}")
        res = _REG_CHAIN.invoke({
            "has_history": hist.get("has_history", False),
            "history_turns": hist.get("history_turns", 0),
            "summary": summary_text or "",
            "question": question,
        })
        data = json.loads(_to_text(res))
    except Exception as e:
        print(f"[REG] 회귀 LLM 예외 → False 폴백: {e}")
        data = {"is_regression": False, "confidence": 0.0, "topic_keywords": [], "explicit_markers": [], "reasons": "exception"}
        
    # 누락 보정
    data.setdefault("is_regression", False)
    data.setdefault("confidence", 0.0)
    data.setdefault("topic_keywords", [])
    data.setdefault("explicit_markers", [])
    data.setdefault("reasons", "")
    return data


# (B) 히스토리 키워드 겹침으로 가장 관련 세그먼트 선택
def _pick_related_context_by_overlap(meta_now: dict, *, limit_search: int = 50,
                                     min_sim: float = 0.15, max_ctx_chars: int = 2000):
    """
    키워드/종류 기반으로 과거 대화 검색 → 유사도 점수화 → 상위/임계 필터 + 길이 제한.
    return: (rows_formatted, debug_info)
    """
    now_kws = set(meta_now.get("msg_keywords") or [])
    now_kind = meta_now.get("kind") or None

    # 1) 키워드 우선 검색
    rows, total = search_messages(
        keywords=list(now_kws) if now_kws else None,
        keyword_match="any",
        limit=limit_search
    )

    # 2) 폴백: 노트로 자유 검색
    if not rows and meta_now.get("notes"):
        rows, total = search_messages(query=meta_now["notes"], limit=limit_search)

    # 3) 점수화 (자카드 + kind 보너스)
    scored = []
    for r in rows:
        prev_kws = set(r.get("msg_keywords") or [])
        sim = _jaccard(prev_kws, now_kws)  # 안전 구현 필요(둘 다 공집합 → 0.0)
        if now_kind and (r.get("kind") or "").strip().lower() == now_kind:
            sim += 0.15
        scored.append((max(sim, 0.0), r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 4) 임계치 필터 + 상위 N
    filtered = [r for s, r in scored if s >= min_sim][:16]  # 후보 넉넉히
    formatted = format_search_results(filtered)

    # 5) 문자수 제한 (토큰 가드)
    picked, acc = [], 0
    for r in formatted:
        line = f"- [{r.get('date','') or ''} {r.get('time','') or ''}] {r.get('role','') or ''}: {r.get('text','') or ''}"
        if acc + len(line) > max_ctx_chars:
            break
        picked.append(r); acc += len(line)

    return picked, {
        "searched_total": total,
        "now_keywords": list(now_kws),
        "now_kind": now_kind,
        "scored": len(scored),
        "filtered_by_min_sim": len(filtered),
        "picked": len(picked),
    }
    
def _get_history_stats() -> dict:
    # 당신의 저장소 조회 함수에 맞춰 최소 구현 (예: 최근 1개만 조회)
    rows, total = search_messages(limit=1)  # 없으면 ([], 0)을 가정
    return {"has_history": total > 0, "history_turns": total}
    

# (C) 최종: 회귀 판단 → 맥락 프롬프트 생성
def build_question_with_regression_context(question: str, summary_text: str) -> Tuple[str, dict]:
        # 1) 히스토리 게이트
    hist = _get_history_stats()
    print(f"[REG] history: has_history={hist['has_history']}, turns={hist['history_turns']}")
    if not hist["has_history"]:
        print("[REG] no history → regression=False (first turn hard gate)")
        return question, {"regression": False, "reason": "first_turn_no_history"}

    meta_now = _extract_meta(question)
    print(f"[REG] meta_now: {meta_now}")
    print(f"[REG] summary_text: {summary_text}")

    reg = _llm_detect_regression(question, summary_text, hist)
    print(f"[REG] LLM 판정: {reg}")

    conf_th = float(os.environ.get("REG_CONF_THRESH", "0.65"))
    use_regression = bool(reg.get("is_regression")) and float(reg.get("confidence", 0.0)) >= conf_th

    if use_regression:
        # 회귀 True → 과거 문맥을 '선택'만 시도 (실패해도 회귀 True 유지)
        merged_kws = list(set((meta_now.get("msg_keywords") or []) + (reg.get("topic_keywords") or [])))
        context_rows, dbg = _pick_related_context_by_overlap({"msg_keywords": merged_kws, "kind": meta_now.get("kind"), "notes": meta_now.get("notes")})
        lines = [
            f"- [{r.get('date','') or ''} {r.get('time','') or ''}] {r.get('role','') or ''}: {r.get('text','') or ''}"
            for r in context_rows
        ]
        joined = "\n".join(lines)
        kw_str = ", ".join([k for k in merged_kws if k]) or "없음"

        debug = {
            "regression": True,
            "confidence": reg.get("confidence", 0.0),
            "explicit_markers": reg.get("explicit_markers", []),
            "keywords": merged_kws,
            "context_used": len(context_rows),
            **dbg
        }

        if context_rows:
            q = (
                f"사용자가 과거 대화의 연속을 말하고 있습니다. (회귀 감지: True, 신뢰도={reg.get('confidence'):.2f}, 키워드: {kw_str})\n"
                f"다음 과거 대화 맥락을 참고하여 자연스럽게 이어 답하세요.\n"
                f"과거 대화:\n{joined}\n\n"
                f"현재 발화: {question}"
            )
            return q, debug

        # 회귀 True + 컨텍스트 0 → 원문 사용하되 디버그로 회귀 흔적 남김
        return question, debug

    # 회귀 아님
    return question, {"regression": False, "confidence": reg.get("confidence", 0.0)}
# ================== /회귀 감지 ==================
