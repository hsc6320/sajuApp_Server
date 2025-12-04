# creative_angles.py
from __future__ import annotations
import random

TEN_GOD_CUES = {
    "정재": ["현금흐름", "보수적 운영", "실물/현금 자산"],
    "편재": ["외부기회", "유통/확장", "빠른 자금 회전"],
    "식신": ["실행력", "꾸준함", "성과 축적"],
    "상관": ["표현/혁신", "룰 재정의", "브랜딩"],
    "정관": ["책임/직위", "제도권", "정공법"],
    "편관": ["압박/과제", "도전 환경", "임무 처리"],
    "비견": ["동료/협업", "자율성", "동수 경쟁"],
    "겁재": ["경쟁/분산", "리스크 분기", "빠른 전환"],
    "정인": ["학습/증빙", "레퍼런스", "내실 다지기"],
    "편인": ["아이디어/R&D", "실험", "우회 전략"],
}

TEN_GOD_BRANCH_CUES = {
    "정재": ["재고/현장 통제", "거래 조건 최적화"],
    "편재": ["채널 확장", "네트워킹"],
    "정관": ["평판/지위 강화", "규범 준수"],
    "편관": ["위기관리", "속도전"],
    "식신": ["루틴 최적화", "지속 가능 페이스"],
    "상관": ["과감한 시도", "피봇 신호"],
    "비견": ["파트너십", "내부 조율"],
    "겁재": ["구조조정", "선택과 집중"],
    "정인": ["자격/포트폴리오", "문서화"],
    "편인": ["파일럿/프로토타입", "컨셉 검증"],
}

LIFESTAGE_CUES = {
    "장생": ["새 사이클 시작", "씨앗 뿌리기"],
    "목욕": ["변동/실험", "정비"],
    "관대": ["성장 관성", "가속"],
    "건록": ["실권/핵심역할", "주도권"],
    "제왕": ["피크/집중", "스포트라이트"],
    "쇠": ["둔화", "유지보수"],
    "병": ["부담/오버로드", "정리 필요"],
    "사": ["마무리", "회수/정산"],
    "묘": ["휴지/잠복", "재충전"],
    "절": ["단절/리셋", "재배치"],
    "태": ["준비/학습", "예열"],
    "양": ["발아/초동력", "작은 승리"],
}

def _pick(xs: list[str], k: int=2) -> list[str]:
    xs = list(dict.fromkeys([x for x in xs if x]))  # 중복 제거
    random.shuffle(xs)
    return xs[:k] if xs else []

def build_angle_cues_for_slice(s: dict) -> dict:
    """target_times 항목 1개 → 창의 브리프 단서"""
    tg  = (s.get("sipseong") or "").strip()
    tgb = (s.get("sipseong_branch") or "").strip()
    ls  = (s.get("sibi_unseong") or "").strip()
    cues = []
    cues += _pick(TEN_GOD_CUES.get(tg, []), 2)
    cues += _pick(TEN_GOD_BRANCH_CUES.get(tgb, []), 2)
    cues += _pick(LIFESTAGE_CUES.get(ls, []), 2)
    return {
        "label": s.get("label") or s.get("scope"),
        "ganji": s.get("ganji"),
        "ten_god": tg,
        "ten_god_branch": tgb,
        "life_stage": ls,
        "angles": cues,   # ← “서술 관점 힌트”만 전달 (판단/점수 없음)
    }

def build_creative_brief(payload: dict, question: str) -> dict:
    """전체 비교용 창의 브리프(판단X, 단서O)"""
    slices = payload.get("target_times") or []
    briefs = [build_angle_cues_for_slice(s) for s in slices]
    return {
        "question": question,
        "slices": briefs,
        "style_guidelines": {
            "avoid_repetition": True,
            "max_theory": 1,       # 이론 설명은 짧게
            "one_metaphor_max": 1, # 메타포 1회 이하
            "contrast_first": True # 첫 단락에 차이부터
        }
    }
