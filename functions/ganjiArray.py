# ganjiArray.py — 비교/배열형 타겟 유틸 전부 모음

from datetime import datetime
import re
from typing import Optional, List, Dict, Tuple, Iterable

from Sipsin import (
    _norm_stem,
    branch_from_any,
    get_ji_sipshin_only,
    get_sipshin,
    stem_from_any,
)
from ganji_converter import Scope, get_ilju, get_wolju_from_date, get_year_ganji_from_json
from sip_e_un_sung import unseong_for

# ── 내부에서 호출 시 없으면 무시되는 선택 함수(프로젝트에 있으면 자동 활용) ──
try:
    from ganji_converter import get_siju_from_datetime  # 시주 계산 함수(있으면 사용)
except Exception:  # ImportError 포함
    get_siju_from_datetime = None


# ───────────────────────── 헬퍼들 ─────────────────────────

def _safe_label(dt: datetime, scope: Scope) -> str:
    if scope == "year":
        return f"{dt.year}년"
    if scope == "month":
        return f"{dt.year}년 {dt.month}월"
    if scope == "day":
        return f"{dt.year}년 {dt.month}월 {dt.day}일"
    return f"{dt.year}-{dt.month:02d}-{dt.day:02d} {dt.hour}시"

def _ganji_for(dt: datetime, scope: Scope, JSON_PATH: str) -> Optional[str]:
    if scope == "year":
        return get_year_ganji_from_json(dt, JSON_PATH)
    if scope == "month":
        return get_wolju_from_date(dt, JSON_PATH, month_only=True)  # 월 단위 질문이므로 month_only=True
    if scope == "day":
        return get_ilju(dt)
    if scope == "hour":
        try:
            if get_siju_from_datetime:
                return get_siju_from_datetime(dt)  # 프로젝트에 함수가 있을 때만 사용
        except Exception:
            return None
        return None
    return None

def _sipseong_split(day_stem_hj: str, ganji: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """천간기준 십성, 지지기준 십성을 각각 계산해 반환. (데이터 없으면 None)"""
    if not ganji:
        return None, None
    s = stem_from_any(ganji)
    b = branch_from_any(ganji)
    tg_stem = get_sipshin(day_stem_hj, s) if s else None
    tg_br   = get_ji_sipshin_only(day_stem_hj, b) if b else None
    if tg_stem in ("미정", "없음"): tg_stem = None
    if tg_br   in ("미정", "없음"): tg_br   = None
    return tg_stem, tg_br

# ─────────────────── target_times 엔트리 생성 ───────────────────

def build_target_entry_from_date(
    day_stem_hj: str,
    dt: datetime,
    scope: Scope,
    JSON_PATH: str,
    label: Optional[str] = None
) -> Optional[dict]:
    """단일 날짜/스코프를 배열형 target_times 항목으로 변환."""
    ganji = _ganji_for(dt, scope, JSON_PATH)
    if not ganji:
        return None

    stem  = stem_from_any(ganji)
    branch= branch_from_any(ganji)
    sip_gan, sip_br = _sipseong_split(day_stem_hj, ganji)

    try:
        sibi = unseong_for(day_stem_hj, branch) if branch else None
    except Exception:
        sibi = None

    return {
        "label": label or _safe_label(dt, scope),
        "scope": scope,                 # "year" | "month" | "day" | "hour"
        "ganji": ganji,                 # 예: "乙巳"
        "stem": stem,                   # 예: "乙"
        "branch": branch,               # 예: "巳"
        "sipseong": sip_gan,            # 천간 기준 십성 (예: 상관)
        "sipseong_branch": sip_br,      # 지지 기준 십성 (예: 편재)
        "sibi_unseong": sibi,           # 지지 기반 십이운성 (예: 절)
    }

# ─────────────────── LLM 비교용 포맷터 ───────────────────

def extract_comparison_slices(payload: dict) -> List[Dict[str, str]]:
    """
    LLM에 넘길 '비교 입력' 배열을 만든다. (계산/추정 없음, payload만 사용)
    """
    out: List[Dict[str, str]] = []

    # 우선 target_times 배열을 사용
    for e in (payload.get("target_times") or []):
        out.append({
            "label":            e.get("label") or e.get("ganji") or "",
            "timeframe":        "연운" if e.get("scope") == "year" else (e.get("scope") or ""),
            "ganji":            e.get("ganji") or "",
            "stem":             e.get("stem") or "",
            "branch":           e.get("branch") or "",
            "ten_god_stem":     e.get("sipseong") or "",           # 천간 기준 십성
            "ten_god_branch":   e.get("sipseong_branch") or "",    # 지지 기준 십성
            "life_stage":       e.get("sibi_unseong") or "",       # 십이운성
        })

    # 배열이 비어 있으면 legacy target_time.year만 1개 추출
    if not out:
        legacy_year = (payload.get("target_time") or {}).get("year") or {}
        if legacy_year.get("ganji"):
            out.append({
                "label":            "질문-연운1",
                "timeframe":        "연운",
                "ganji":            legacy_year.get("ganji") or "",
                "stem":             legacy_year.get("stem") or "",
                "branch":           legacy_year.get("branch") or "",
                "ten_god_stem":     legacy_year.get("sipseong") or "",
                "ten_god_branch":   legacy_year.get("sipseong_branch") or "",
                "life_stage":       legacy_year.get("sibi_unseong") or "",
            })

    return out

def format_comparison_block(slices: List[Dict[str, str]]) -> str:
    """
    휴먼 프롬프트에 그대로 붙일 텍스트 블록을 생성.
    """
    if not slices:
        return ""
    lines = ["[COMPARISON_INPUT]"]
    for s in slices:
        line = (
            f"- {s['label']} ({s['timeframe']}): "
            f"간지={s['ganji']}"
        )
        if s.get("ten_god_stem"):
            line += f", 천간 십성={s['ten_god_stem']}"
        if s.get("ten_god_branch"):
            line += f", 지지 십성={s['ten_god_branch']}"
        if s.get("life_stage"):
            line += f", 십이운성={s['life_stage']}"
        lines.append(line)
    return "\n".join(lines)



# ---- (1) 한글 간지 → 한자 간단 맵 (부족하면 계속 보강하면 됨) ----
_HANGUL_TO_HANJA = {
    # 천간
    "갑":"甲","을":"乙","병":"丙","정":"丁","무":"戊","기":"己","경":"庚","신":"辛","임":"壬","계":"癸",
    # 지지
    "자":"子","축":"丑","인":"寅","묘":"卯","진":"辰","사":"巳","오":"午","미":"未","신":"申","유":"酉","술":"戌","해":"亥",
}

def _ko_ganji_to_hanja(token: str) -> str:
    """'갑진' '을사' 같은 2글자 한글 간지를 한자로 변환. 변환 실패 시 원본 반환."""
    if len(token) != 2:
        return token
    a, b = token[0], token[1]
    ha = _HANGUL_TO_HANJA.get(a, a)
    hb = _HANGUL_TO_HANJA.get(b, b)
    # 모두 한자면 OK, 아니면 원본 리턴
    if ha in "甲乙丙丁戊己庚辛壬癸" and hb in "子丑寅卯辰巳午未申酉戌亥":
        return ha + hb
    return token

_HANJA_GAN = "甲乙丙丁戊己庚辛壬癸"
_HANJA_JI  = "子丑寅卯辰巳午未申酉戌亥"

# 한자 간지(예: 甲辰) 또는 한글 간지(예: 갑진) 전부 잡아내기
_RE_HANJA = re.compile(rf"([{_HANJA_GAN}][{_HANJA_JI}])")
_RE_HANGUL = re.compile(r"(갑자|을축|병인|정묘|무진|기사|경오|신미|임신|계유|갑술|을해|병자|정축|무인|계해|임자|신유|경신|무오|정사|병진|을묘|갑인|계유|임신|신미|경오|기사|무진|정묘|병인|을축|갑자)")

def _collect_year_ganji_tokens(text: str) -> list[str]:
    """문장 내에서 연운 비교용 간지 후보(한자/한글)를 모두 한자 간지로 수집."""
    tokens: list[str] = []
    if not text:
        return tokens
    # 1) 한자 간지
    for m in _RE_HANJA.findall(text):
        tokens.append(m)
    # 2) 한글 간지
    for m in _RE_HANGUL.findall(text):
        tokens.append(_ko_ganji_to_hanja(m))
    # 정리: 2글자 한자 간지만 남기고 중복 제거
    uniq = []
    seen = set()
    for t in tokens:
        if isinstance(t, str) and len(t) == 2 and t[0] in _HANJA_GAN and t[1] in _HANJA_JI:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
    
    print(f"_collect_year_ganji_tokens() : uniq : {uniq}")
    return uniq


# 한자 간지 패턴(예: 甲辰, 乙巳)
_GANJI_PAIR_RE = re.compile(r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]")

# YYYY-MM, YYYY-MM-DD, YYYY년, YYYY년 M월 등
# [FIX] 2자리 연도도 인식하도록 수정 (예: "26년 2월")
_YYYY_MM_DD_RE = re.compile(r"(\d{4})[.\-년/\s]?(\d{1,2})[.\-월/\s]?(\d{1,2})[일]?")
_YYYY_MM_RE    = re.compile(r"(\d{4})[.\-년/\s]?(\d{1,2})[.\-월]?")
_YY_MM_RE      = re.compile(r"(?<!\d)(\d{2})\s*년\s*(\d{1,2})\s*월")  # [NEW] 2자리 연도 + 월
_YYYY_RE       = re.compile(r"(\d{4})[.\-]?\s*년?")

def parse_compare_specs(question: str) -> dict:
    """
    질문에서 비교 후보를 추출해 dict로 반환:
    {
      "ganji_years": ["甲辰", "乙巳", ...],         # 간지로만 제시된 경우(디폴트 year로 해석)
      "years":   [2025, 2026, ...],
      "months":  [(2025,5), (2026,3), ...],
      "days":    [(2025,5,28), (2026,7,1), ...],
    }
    """
    q = (question or "").strip()

    ganji_pairs = list(dict.fromkeys(_GANJI_PAIR_RE.findall(q)))

    # 가장 구체적인(일→월→년)부터 추출
    days = []
    for y, m, d in _YYYY_MM_DD_RE.findall(q):
        try:
            days.append((int(y), int(m), int(d)))
        except Exception:
            pass
    days = list(dict.fromkeys(days))

    months = []
    for y, m in _YYYY_MM_RE.findall(q):
        try:
            months.append((int(y), int(m)))
        except Exception:
            pass
    # [NEW] 2자리 연도 + 월 패턴 처리 (예: "26년 2월")
    for y2, m in _YY_MM_RE.findall(q):
        try:
            from ganji_converter import resolve_two_digit_year
            from datetime import datetime
            year_suffix = int(y2)
            full_year = resolve_two_digit_year(year_suffix, today=datetime.now(), prefer_past_on_tie=True)
            months.append((full_year, int(m)))
        except Exception:
            pass
    # 일 패턴에 잡힌 것과 중복 제거
    month_set = set(months)
    months = [x for x in months if all((x[0], x[1], d) not in set(days) for d in range(1,32))]
    months = list(dict.fromkeys(months))

    years = []
    for y in _YYYY_RE.findall(q):
        try:
            years.append(int(y))
        except Exception:
            pass
    # 월/일에 포함된 연도와 중복 제거
    years = [y for y in years if y not in {ymd[0] for ymd in days} and y not in {ym[0] for ym in months}]
    years = list(dict.fromkeys(years))

    return {
        "ganji_years": ganji_pairs,
        "years": years,
        "months": months,
        "days": days,
    }
