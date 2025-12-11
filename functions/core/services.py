
import json
import hashlib
from typing import Optional, List
from datetime import date, datetime
import re

from ganjiArray import extract_comparison_slices, format_comparison_block, parse_compare_specs
from ganji_converter import Scope, get_ilju, get_wolju_from_date, get_year_ganji_from_json, JSON_PATH
from regress_conversation import get_extract_chain, _today, _maybe_override_target_date
from converting_time import extract_target_ganji_v2, convert_relative_time, parse_korean_date_safe, is_month_only_question
from sip_e_un_sung import _branch_of, unseong_for, branch_for, pillars_unseong, seun_unseong
from Sipsin import _norm_stem, branch_from_any, get_sipshin, get_ji_sipshin_only, stem_from_any

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

FORTUNE_KEYS = ["초씨역림", "주역", "점괘", "괘", "육효", "괘상", "점쳐", "점치"]

def is_fortune_query(text: str) -> bool:
    t = (text or "").strip()
    return any(k in t for k in FORTUNE_KEYS)

def _sipseong_split_for_target(day_stem_hj: str, target_ganji: str | None) -> str | None:
    """일간(day_stem_hj) 기준으로 target의
    - 천간 십성(= sipseong)
    - 지지 십성(= sipseong_branch)
    를 함께 반환한다."""
    if not target_ganji:
        return None, None
    t_stem_hj = stem_from_any(target_ganji)
    t_branch_hj = branch_from_any(target_ganji)
    
    ten_god_stem = get_sipshin(day_stem_hj, t_stem_hj) if t_stem_hj else None
    ten_god_branch = get_ji_sipshin_only(day_stem_hj, t_branch_hj) if t_branch_hj else None
    
    if ten_god_stem in ("미정", "없음"): ten_god_stem = None
    if ten_god_branch in ("미정", "없음"): ten_god_branch = None
    
    return ten_god_stem, ten_god_branch

def style_seed_from_payload(payload: dict) -> int:
    key = (payload.get("meta", {}).get("question","") +
           "|" + ",".join([s.get("ganji","") for s in payload.get("target_times", [])]))
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10_000

# ── target_times → legacy(target_time, resolved.flow_now.target) 미러 ──
_TARGET_KEYS = ("ganji", "stem", "branch", "sipseong", "sipseong_branch", "sibi_unseong")
_SCOPES      = ("year", "month", "day", "hour")

def _as_legacy_slot(entry: dict) -> dict:
    return {k: (entry.get(k) if entry.get(k) not in ("", None) else None) for k in _TARGET_KEYS}

def mirror_target_times_to_legacy(payload: dict) -> None:
    """
    target_times에서 scope별(연/월/일/시) '첫 항목'을 뽑아
    - payload.target_time (legacy single 구조)
    - payload.resolved.flow_now.target (해석 경로)
    에 동기화한다.
    기존 single 값이 있으면 빈 슬롯만 채운다(보수적 merge).
    """

    tt = payload.get("target_times") or []
    if not isinstance(tt, list):
        tt = []

    # 각 scope의 첫 항목만 legacy에 반영
    legacy = {s: None for s in _SCOPES}
    seen = set()
    for e in tt:
        scope = e.get("scope")
        if scope in _SCOPES and scope not in seen:
            legacy[scope] = _as_legacy_slot(e)
            seen.add(scope)
            if len(seen) == len(_SCOPES):
                break

    # 기존 single이 있으면 비어 있는 슬롯만 채움
    single = payload.get("target_time") or {s: None for s in _SCOPES}
    for s in _SCOPES:
        if single.get(s) is None and legacy.get(s) is not None:
            single[s] = legacy[s]
    payload["target_time"] = single

    # resolved.flow_now.target 도 동기화
    payload.setdefault("resolved", {})
    payload["resolved"].setdefault("flow_now", {})
    payload["resolved"]["flow_now"].setdefault("target", {})
    for s in _SCOPES:
        slot = payload["target_time"].get(s)
        payload["resolved"]["flow_now"]["target"][s] = (dict(slot) if slot else None)


def _entry_from_known(day_stem_hj, scope: str, g: Optional[str], sip_gan, sip_br, sibi) -> Optional[dict]:
    if not g:
        return None
    return {
        "label": {"year":"연운","month":"월운","day":"일운","hour":"시운"}.get(scope, scope),
        "scope": scope,                # "year" | "month" | "day" | "hour"
        "ganji": g,
        "stem":  stem_from_any(g),
        "branch":branch_from_any(g),
        "sipseong":        sip_gan,    # 천간 기준 십성
        "sipseong_branch": sip_br,     # 지지 기준 십성
        "sibi_unseong":    sibi,       # 지지 기반 십이운성
    }

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

    #print(f"target_ganji_list :{target_ganji_list}")   

    
    # === 1) 타겟 간지 파싱 후, pillars_unseong로 일괄 계산 ===
    pillars_branches = {
        "year":  branch_from_any(t_year_ganji),
        "month": branch_from_any(t_month_ganji),
        "day":   branch_from_any(t_day_ganji),
        "hour":  branch_from_any(t_hour_ganji),
    }

    #  일간(천간) 표준화: 한글/혼합 → 한자(예: '임'→'壬') (★)
    day_stem_hj = _norm_stem(ilGan)  # ilGan 예: '임' 또는 '壬' 한자로 변환

    
    # None이 섞여 있어도 pillars_unseong 내부에서 처리됨
     # 타겟(연/월/일/시) 십이운성 맵
    target_sibi_map = pillars_unseong(day_stem_hj, pillars_branches)
    # 예: {'year': '관대', 'month': '절', 'day': None, 'hour': '장생'}

    # === [B] 현재 대운 십이운성 (★ _branch_of → branch_from_any)
    print(f"current_dw : {current_dw}")
    current_dw_branch = branch_from_any(current_dw)  # 예: '亥' 또는 None
    curr_dw_sibi = unseong_for(day_stem_hj, current_dw_branch) if current_dw_branch else None
    
    print(f"day_stem_hj : {day_stem_hj}, current_dw_branch : {current_dw_branch}, curr_dw_sibi : {curr_dw_sibi}")
    
    # 타겟(연/월/일/시) 십성(천간/지지)
    year_sip_gan,  year_sip_br = _sipseong_split_for_target(day_stem_hj, t_year_ganji)
    month_sip_gan, month_sip_br = _sipseong_split_for_target(day_stem_hj, t_month_ganji)
    day_sip_gan,   day_sip_br = _sipseong_split_for_target(day_stem_hj, t_day_ganji)
    hour_sip_gan,  hour_sip_br = _sipseong_split_for_target(day_stem_hj, t_hour_ganji)
    # 대운 십성 계산으로 교체 ✅
    dw_sip_gan, dw_sip_br = _sipseong_split_for_target(day_stem_hj, current_dw)

    print(f"year_sip_gan : {year_sip_gan}, year_sip_br : {year_sip_br}, month_sip_gan : {month_sip_gan}, month_sip_br : {month_sip_br}")
    # 따옴표 오류 수정(내부 키는 작은따옴표로)
    print(f"target_sibi_map.get(year/month/day) : {target_sibi_map.get('year')}, {target_sibi_map.get('month')}, {target_sibi_map.get('day')}")

    #배열형 target_times 구성(기본 1건 + 비교질문)
    target_times: List[dict] = []

    # 기본 1건(있을 때만)
    if t_year_ganji:
        e = _entry_from_known(day_stem_hj, "year",  t_year_ganji,  year_sip_gan,  year_sip_br,  target_sibi_map.get("year"))
        if e: target_times.append(e)
    if t_month_ganji:
        e = _entry_from_known(day_stem_hj, "month", t_month_ganji, month_sip_gan, month_sip_br, target_sibi_map.get("month"))
        if e: target_times.append(e)
    if t_day_ganji:
        e = _entry_from_known(day_stem_hj, "day",   t_day_ganji,   day_sip_gan,   day_sip_br,   target_sibi_map.get("day"))
        if e: target_times.append(e)
    if t_hour_ganji:
        e = _entry_from_known(day_stem_hj, "hour",  t_hour_ganji,  hour_sip_gan,  hour_sip_br,  target_sibi_map.get("hour"))
        if e: target_times.append(e)

    # 비교 질문 파싱(간지/연/월/일)
    specs = parse_compare_specs(updated_question)

    # (a) 간지(예: 甲辰, 乙巳) → 연운으로 간주
    for gj in (specs.get("ganji_years") or []):
        if any(e.get("scope") == "year" and e.get("ganji") == gj for e in target_times):
            continue
        sip_gan, sip_br = _sipseong_split_for_target(day_stem_hj, gj)
        entry = {
            "label": f"{gj} 연운",
            "scope": "year",
            "ganji": gj,
            "stem":  stem_from_any(gj),
            "branch":branch_from_any(gj),
            "sipseong":        sip_gan,
            "sipseong_branch": sip_br,
            "sibi_unseong":    (unseong_for(day_stem_hj, branch_from_any(gj)) if (day_stem_hj and branch_from_any(gj)) else None),
        }
        target_times.append(entry)

    # (b) 연도 숫자(예: 2025, 2026) → 1월 1일 기준 연운
    for y in (specs.get("years") or []):
        dt = datetime(y, 1, 1)
        gj = get_year_ganji_from_json(dt, JSON_PATH)  # JSON_PATH는 상위 스코프/설정에서 참조
        if not gj: 
            continue
        if any(e.get("scope") == "year" and e.get("ganji") == gj for e in target_times):
            continue
        sip_gan, sip_br = _sipseong_split_for_target(day_stem_hj, gj)
        entry = {
            "label": f"{y}년",
            "scope": "year",
            "ganji": gj,
            "stem":  stem_from_any(gj),
            "branch":branch_from_any(gj),
            "sipseong":        sip_gan,
            "sipseong_branch": sip_br,
            "sibi_unseong":    (unseong_for(day_stem_hj, branch_from_any(gj)) if (day_stem_hj and branch_from_any(gj)) else None),
        }
        target_times.append(entry)

    # (c) 월(YYYY-MM) → 월운
    for (y, m) in (specs.get("months") or []):
        dt = datetime(y, m, 1)
        # [FIX] 월 단위 스펙이므로 month_only=True로 처리하여 정확한 월주 찾기
        gj = get_wolju_from_date(dt, JSON_PATH, month_only=True)
        if not gj:
            continue
        sip_gan, sip_br = _sipseong_split_for_target(day_stem_hj, gj)
        entry = {
            "label": f"{y}년 {m}월",
            "scope": "month",
            "ganji": gj,
            "stem":  stem_from_any(gj),
            "branch":branch_from_any(gj),
            "sipseong":        sip_gan,
            "sipseong_branch": sip_br,
            "sibi_unseong":    (unseong_for(day_stem_hj, branch_from_any(gj)) if (day_stem_hj and branch_from_any(gj)) else None),
        }
        target_times.append(entry)

    # (d) 일(YYYY-MM-DD) → 일운
    for (y, m, d) in (specs.get("days") or []):
        try:
            dt = datetime(y, m, d)
        except Exception:
            continue
        gj = get_ilju(dt)
        if not gj:
            continue
        sip_gan, sip_br = _sipseong_split_for_target(day_stem_hj, gj)
        entry = {
            "label": f"{y}년 {m}월 {d}일",
            "scope": "day",
            "ganji": gj,
            "stem":  stem_from_any(gj),
            "branch":branch_from_any(gj),
            "sipseong":        sip_gan,
            "sipseong_branch": sip_br,
            "sibi_unseong":    (unseong_for(day_stem_hj, branch_from_any(gj)) if (day_stem_hj and branch_from_any(gj)) else None),
        }
        target_times.append(entry)

    # 간단 중복 제거(scope+ganji)
    seen = set(); dedup = []
    for e in target_times:
        key = (e.get("scope"), e.get("ganji"))
        if key in seen: 
            continue
        seen.add(key); dedup.append(e)
    target_times = dedup

    # 최종 스키마 구성
    payload = {
        "saju": {
            "year": year,          # 원국 간지(문자열)
            "month": month,
            "day": day,            # 필요 시 일간/일지 분리 구조로 확장 가능
            "hour": pillar_hour
        },
        "natal": {
            "sipseong_by_pillar": {
                "year": yearGan or None,   # 원국 각 기둥의 '십성' 라벨 (있으면)
                "month": wolGan or None,
                "day": ilGan or None,
                "hour": siGan or None,
            }
        },
        # === 현재 대운 ===
        "current_daewoon": {
            "ganji": current_dw or None,                          # 예: "계해"
            "stem":  stem_from_any(current_dw) if current_dw else None,    # 천간
            "branch":branch_from_any(current_dw) if current_dw else None,  # 지지
            "sipseong":        dw_sip_gan,                        # ✅ 일간 기준 '천간' 십성 (예: 편인)
            "sipseong_branch": dw_sip_br,                         # ✅ 일간 기준 '지지' 십성 (있으면 권장)
            "sibi_unseong":    curr_dw_sibi,                      # ✅ 일간 기준 '지지' 기반 십이운성
        },
        # === 타겟 시점(연/월/일/시) ===
        "target_time": {
            "year":  {
                "ganji": t_year_ganji,                                        # 예: "을사"
                "stem":  stem_from_any(t_year_ganji) if t_year_ganji else None,
                "branch":branch_from_any(t_year_ganji) if t_year_ganji else None,
                "sipseong":        year_sip_gan,                              # ✅ 천간 기준 십성
                "sipseong_branch": year_sip_br,                               # ✅ 지지 기준 십성
                "sibi_unseong":    target_sibi_map.get("year"),               # ✅ 지지 기반 십이운성
            },
            "month": {
                "ganji": t_month_ganji,
                "stem":  stem_from_any(t_month_ganji) if t_month_ganji else None,
                "branch":branch_from_any(t_month_ganji) if t_month_ganji else None,
                "sipseong":        month_sip_gan,
                "sipseong_branch": month_sip_br,
                "sibi_unseong":    target_sibi_map.get("month"),
            },
            "day": {
                "ganji": t_day_ganji,
                "stem":  stem_from_any(t_day_ganji) if t_day_ganji else None,
                "branch":branch_from_any(t_day_ganji) if t_day_ganji else None,
                "sipseong":        day_sip_gan,
                "sipseong_branch": day_sip_br,
                "sibi_unseong":    target_sibi_map.get("day"),
            },
            "hour": {
                "ganji": t_hour_ganji,
                "stem":  stem_from_any(t_hour_ganji) if t_hour_ganji else None,
                "branch":branch_from_any(t_hour_ganji) if t_hour_ganji else None,
                "sipseong":        hour_sip_gan,
                "sipseong_branch": hour_sip_br,
                "sibi_unseong":    target_sibi_map.get("hour"),
            },
        },

        "focus": focus,
        "meta": {
            "user_name": user_name,
            "daewoon": daewoon,
            "yinYang": yinYang,
            "fiveElement": fiveElement,
            "session_id": session_id,      # 필요 시 상위에서 실제 세션 주입
            "question": question,

            # 🔥 요약 엔진에서 바로 읽어갈 수 있는 엔티티 블록
            "entities": {
                "간지": target_ganji_list,   # [연,월,일,시] 중 추출된 간지 목록
                "타겟_연도": t_year_ganji,
                "타겟_월": t_month_ganji,
                "타겟_일": t_day_ganji,
                "타겟_시": t_hour_ganji,
                "키워드": [],                # (옵션) 별도 키워드 추출기로 채움
                "이벤트": []                 # (옵션)
            }
        }
    }
    payload["target_times"] = target_times
    print(f"focus meta 대운 : ${payload["meta"]["daewoon"]}")
    mirror_target_times_to_legacy(payload)
    #print(f"compare_items : {compare_items}, target_times : {target_times}")
    
    
     # === E) 정규화 블록(resolved) 추가: 모델은 여기만 보면 됨 ===
    def _stem(g):   return g[0] if isinstance(g, str) and len(g) >= 1 else None
    def _branch(g): return g[1] if isinstance(g, str) and len(g) >= 2 else None

    resolved_pillars = {
        "year":  {"ganji": year or None,  "stem": _stem(year),  "branch": _branch(year),  "sipseong": None, "sibi_unseong": None},
        "month": {"ganji": month or None, "stem": _stem(month), "branch": _branch(month), "sipseong": None, "sibi_unseong": None},
        "day":   {"ganji": day or None,   "stem": _stem(day),   "branch": _branch(day),   "sipseong": None, "sibi_unseong": None},
        "hour":  {"ganji": pillar_hour or None, "stem": _stem(pillar_hour), "branch": _branch(pillar_hour), "sipseong": None, "sibi_unseong": None},
    }

    # 기존 보유 헬퍼 재사용
    # stem_from_any("乙巳") -> "乙", branch_from_any("乙巳") -> "巳"
    payload["resolved"] = {
        "pillars": resolved_pillars,
        "flow_now": {
            "daewoon": {
                "ganji": current_dw or None,
                "stem":  stem_from_any(current_dw)   if current_dw else None,
                "branch":branch_from_any(current_dw) if current_dw else None,
                "sipseong":        dw_sip_gan,   # ✅ 대운 '천간' 기준 십성
                "sipseong_branch": dw_sip_br,    # ✅ 대운 '지지' 기준 십성 (신규)
                "sibi_unseong":    curr_dw_sibi, # 대운 십이운성 (지지 기반)
            },
            "target": {
                "year":  {
                    "ganji":   t_year_ganji,
                    "stem":    stem_from_any(t_year_ganji)   if t_year_ganji else None,
                    "branch":  branch_from_any(t_year_ganji) if t_year_ganji else None,
                    "sipseong":        year_sip_gan,    # ✅ 연운 '천간' 기준 십성
                    "sipseong_branch": year_sip_br,     # ✅ 연운 '지지' 기준 십성 (신규)
                    "sibi_unseong":    target_sibi_map.get("year"),
                },
                "month": {
                    "ganji":   t_month_ganji,
                    "stem":    stem_from_any(t_month_ganji)   if t_month_ganji else None,
                    "branch":  branch_from_any(t_month_ganji) if t_month_ganji else None,
                    "sipseong":        month_sip_gan,   # ✅ 월운 '천간' 기준 십성
                    "sipseong_branch": month_sip_br,    # ✅ 월운 '지지' 기준 십성 (신규)
                    "sibi_unseong":    target_sibi_map.get("month"),
                },
                "day":   {
                    "ganji":   t_day_ganji,
                    "stem":    stem_from_any(t_day_ganji)   if t_day_ganji else None,
                    "branch":  branch_from_any(t_day_ganji) if t_day_ganji else None,
                    "sipseong":        day_sip_gan,     # ✅ 일운 '천간' 기준 십성
                    "sipseong_branch": day_sip_br,      # ✅ 일운 '지지' 기준 십성 (신규)
                    "sibi_unseong":    target_sibi_map.get("day"),
                },
                "hour":  {
                    "ganji":   t_hour_ganji,
                    "stem":    stem_from_any(t_hour_ganji)   if t_hour_ganji else None,
                    "branch":  branch_from_any(t_hour_ganji) if t_hour_ganji else None,
                    "sipseong":        hour_sip_gan,    # ✅ 시운 '천간' 기준 십성
                    "sipseong_branch": hour_sip_br,     # ✅ 시운 '지지' 기준 십성 (신규)
                    "sibi_unseong":    target_sibi_map.get("hour"),
                },
            }
        },
        "canon": {
            "sipseong_vocab": ["비견","겁재","식신","상관","편재","정재","편관","정관","편인","정인"],
            "sibi_vocab":     ["장생","목욕","관대","건록","제왕","쇠","병","사","묘","절","태","양"]
        }
    }

    return payload


def extract_meta_and_convert(question: str) -> tuple[dict, str]:
    """메타 추출 + 상대시간 → 절대/간지 치환까지 한 번에.
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

    # 2) 기본 필드 보정
    parsed.setdefault("msg_keywords", [])
    parsed.setdefault("target_date", None)
    parsed.setdefault("time", None)
    parsed.setdefault("kind", None)
    parsed.setdefault("notes", "")
    parsed.setdefault("_facts", {})

    # [NEW] Month Granularity Fix: 
    # If target_date is 1st of month (e.g. 2025-11-01) but user didn't ask for 1st, 
    # move to 15th to capture the main Saju month (Solar term).
    if parsed.get("target_date") and parsed["target_date"].endswith("-01"):
        # Check if user explicitly asked for 1st
        # Regex: (?<!\d)1일 looks for '1일' not preceded by a digit (so '11일' is ignored).
        # Also check '첫날', '1st'.
        is_specifically_first = re.search(r"(?<!\d)1일|첫날|1st", question)
        if not is_specifically_first:
            try:
                # Move to 15th
                y_str, m_str, _ = parsed["target_date"].split("-")
                new_date = f"{y_str}-{m_str}-15"
                parsed["target_date"] = new_date
                print(f"[DEIXIS] Month-only query detected. Shifted {y_str}-{m_str}-01 to {new_date} for better Saju Month match.")
            except Exception:
                pass

    # 3) target_date 채우기 (절대 안전)
    #    - LLM이 채워줬다면 그대로 둠
    #    - 없으면 한국어/ISO 패턴을 안전 파서로만 처리 (절대 int(None) 금지)
    if not parsed["target_date"]:
        today = _today()
        # 3-1) 한국어/일반 패턴 파싱
        y, m, d = parse_korean_date_safe(question)

        iso_str = None
        if y is not None and m is not None and d is not None:
            # 연/월/일 모두 있으면 그대로
            try:
                iso_str = date(y, m, d).isoformat()
            except Exception as e:
                print(f"[DEIXIS] 년월일 조합 실패: {e}")

        elif m is not None and d is not None:
            # 연도가 없으면 올해로 보정(정책적으로 today.year 사용)
            try:
                iso_str = date(today.year, m, d).isoformat()
            except Exception as e:
                print(f"[DEIXIS] 월일→올해 보정 실패: {e}")

        if iso_str:
            parsed["target_date"] = iso_str
            parsed["_facts"]["deixis_anchor_date"] = {
                "value": iso_str,
                "source": "korean_abs_or_mmdd"
            }
            print(f"[DEIXIS] target_date 확정 → {iso_str}")

    # 4) 상대시간 치환: expressions에 **질문 원문을 반드시 포함**
    today = _today()
    print(f"[REAL TODAY] {today}") # [NEW] 실제 오늘 날짜 기록

    cy, cm, cd = today.year, today.month, today.day
    expressions = list(dict.fromkeys((parsed.get("msg_keywords") or []) + [question]))
    
    # [FIX] target_date가 있다면, 그 날짜를 '현재 기준일'로 삼아 상대시간 계산
    # 예: "2025-11-15 3일 뒤" -> 11/15 기준 +3일
    if parsed.get("target_date"):
        try:
            td = datetime.fromisoformat(parsed["target_date"])
            cy, cm, cd = td.year, td.month, td.day
            print(f"[DEIXIS] Anchor Override: {cy}-{cm}-{cd} (by target_date)")
        except Exception:
            pass

    # [CHANGE] [TODAY DATE] -> [ANCHOR DATE] 의미 명확화
    print(f"[ANCHOR DATE] {cy}-{cm}-{cd}")
    
    # target_date를 내부 정책(예: '이번주', '다음달')로 덮어쓸 수 있는 후처리
    _maybe_override_target_date(question, parsed, today)

    try:
        abs_kws, updated_q = convert_relative_time(
            question=question,
            expressions=expressions,
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
