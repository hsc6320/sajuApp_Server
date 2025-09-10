# ================== 회귀(이전 대화 회귀) 감지/맥락 결합 ==================
import json
import os
from typing import Tuple

from langchain_core.prompts import PromptTemplate
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
_regression_prompt = PromptTemplate.from_template("""
너는 대화 회귀 감지기다.
주어진 "현재 발화"가 과거 대화의 연속인지(회귀) 여부를 판정하고, 회귀라면 관련 주제 키워드를 1~3개 제시하라.
반드시 JSON으로만 답하라. 키는 다음만 포함:
- is_regression: true/false
- topic_keywords: 문자열 배열(회귀일 때 1~3개, 아니면 빈 배열)

[최근 대화 요약(참고용)]
{summary}

[현재 발화]
{text}
""")

def _get_regression_chain() -> LLMChain:
    # 추출용과 같은 저비용 모델 사용
    llm = ChatOpenAI(
        model=os.environ.get("EXTRACT_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        max_tokens=150,
        timeout=15,
    )
    return LLMChain(llm=llm, prompt=_regression_prompt)

_REG_CHAIN = _get_regression_chain()

def _llm_detect_regression(current_text: str, summary_text: str) -> dict:
    """
    LLM으로 회귀 여부 감지. JSON만 기대.
    실패 시 안전 폴백 반환.
    """
    try:
        raw = _REG_CHAIN.run(text=current_text, summary=summary_text).strip()
        s, e = raw.find("{"), raw.rfind("}")
        payload = raw[s:e+1] if (s!=-1 and e!=-1) else raw
        out = json.loads(payload)
        # 방어적 기본값
        return {
            "is_regression": bool(out.get("is_regression", False)),
            "topic_keywords": out.get("topic_keywords") or []
        }
    except Exception as e:
        print(f"[REG] LLM 회귀 감지 실패 → 폴백 (err={e})")
        return {"is_regression": False, "topic_keywords": []}

# (B) 히스토리 키워드 겹침으로 가장 관련 세그먼트 선택
def _pick_related_context_by_overlap(
    meta_now: dict, *, limit_search: int = 50
) -> Tuple[list[dict], dict]:
    """
    현재 발화의 msg_keywords/kind를 기준으로 JSON에서 최근 대화들 중
    겹침율이 높은 것들을 추려 반환.
    return: (rows_formatted, debug_info)
    """
    now_kws = set((meta_now.get("msg_keywords") or []))
    now_kind = meta_now.get("kind") or None

    # 1) 우선 키워드 기반 검색
    rows, total = search_messages(
        keywords=list(now_kws) if now_kws else None,
        keyword_match="any",
        limit=limit_search
    )
    # 2) 키워드 없거나 너무 적으면 자유검색 폴백
    if not rows and meta_now.get("notes"):
        rows, total = search_messages(query=meta_now["notes"], limit=limit_search)

    # 3) 겹침율 + kind-일치 보너스 점수
    scored = []
    for r in rows:
        prev_kws = set((r.get("msg_keywords") or []))
        score = _jaccard(now_kws, prev_kws)
        if now_kind and r.get("kind") == now_kind:
            score += 0.15  # kind 일치 보너스
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 상위 N개만 사용(너무 길면 LLM 프롬프트 과다)
    top = [r for _, r in scored[:8]]
    return format_search_results(top), {
        "searched_total": total,
        "now_keywords": list(now_kws),
        "now_kind": now_kind,
        "picked": len(top),
    }

# (C) 최종: 회귀 판단 → 맥락 프롬프트 생성
def build_question_with_regression_context(
    question: str, summary_text: str
) -> Tuple[str, dict]:
    """
    현재 발화를 회귀 문맥과 결합한 질문 문자열과 디버그 정보를 반환.
    - LLM 회귀 감지 + 메타 추출 + 키워드 겹침 기반 검색
    """
    # 현재 발화 메타
    meta_now = _extract_meta(question)
    print(f"[REG] meta_now: {meta_now}")

    # LLM 회귀 판정
    reg = _llm_detect_regression(question, summary_text)
    print(f"[REG] LLM 판정: {reg}")

    use_regression = reg["is_regression"]

    # 회귀로 판단되면 관련 히스토리 맥락을 붙임
    if use_regression:
        # topic_keywords가 있으면 우선 병합(추출메타와)
        merged_kws = list(set((meta_now.get("msg_keywords") or []) + (reg.get("topic_keywords") or [])))
        meta_now["msg_keywords"] = merged_kws

        context_rows, dbg = _pick_related_context_by_overlap(meta_now)
        #print(f"[REG] 맥락 선택: picked={dbg['picked']}/{dbg['searched_total']} kws={dbg['now_keywords']} kind={dbg['now_kind']}")

        if context_rows:
            lines = []
            for r in context_rows:
                lines.append(f"- [{r.get('date','')} {r.get('time','')}] {r.get('role','')}: {r.get('text','')}")
            joined = "\n".join(lines)
            kw_str = ", ".join(merged_kws) if merged_kws else "없음"

            q = (
                f"사용자가 과거 대화의 연속을 말하고 있습니다. (회귀 감지: True, 키워드: {kw_str})\n"
                f"다음 과거 대화 맥락을 참고하여 자연스럽게 이어 답하세요.\n"
                f"과거 대화:\n{joined}\n\n"
                f"현재 발화: {question}"
            )
            return q, {"regression": True, "context_used": len(context_rows), "keywords": merged_kws, **dbg}

    # 회귀 아니면 원문 그대로
    return question, {"regression": False}
# ================== /회귀 감지 ==================
