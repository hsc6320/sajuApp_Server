
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
from datetime import datetime

def _extract_birth_year(birth_str: str) -> Optional[int]:
    """
    생년월일 문자열에서 출생 연도를 추출합니다.
    
    Args:
        birth_str: 생년월일 문자열 (예: "1988-07-16", "19880716", "1988/07/16")
    
    Returns:
        출생 연도 (int) 또는 None
    """
    if not birth_str or not isinstance(birth_str, str):
        return None
    
    birth_str = birth_str.strip()
    if not birth_str:
        return None
    
    try:
        # YYYY-MM-DD 형식
        if "-" in birth_str:
            parts = birth_str.split("-")
            if len(parts) >= 1:
                return int(parts[0])
        # YYYYMMDD 형식
        elif len(birth_str) >= 4:
            return int(birth_str[:4])
        # YYYY/MM/DD 형식
        elif "/" in birth_str:
            parts = birth_str.split("/")
            if len(parts) >= 1:
                return int(parts[0])
    except (ValueError, TypeError):
        pass
    
    return None

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
    # ✅ 일간이 없거나 타겟이 없으면 None 반환
    if not day_stem_hj or not target_ganji:
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


def calculate_daewoon_by_age(daewoon_list: List[str], first_luck_age: Optional[int], birth_year: Optional[int] = None, day_stem_hj: Optional[str] = None) -> List[dict]:
    """
    대운 배열과 대운 시작 나이를 받아서 나이대별 대운을 계산합니다.
    생년월일이 있으면 년도 정보도 함께 계산하고, 일간이 있으면 십성과 십이운성도 계산합니다.
    
    Args:
        daewoon_list: 대운 배열 (예: ["壬戌", "辛酉", "庚申", ...])
        first_luck_age: 대운 시작 나이 (예: 4)
        birth_year: 출생 연도 (예: 1988, 선택사항)
        day_stem_hj: 일간 한자 (예: "壬", 십성 계산용, 선택사항)
    
    Returns:
        나이대별 대운 정보 리스트
        예: [
            {
                "year_range": "1992-2001", 
                "age_range": "4-13", 
                "daewoon": "壬戌",
                "stem": "壬",
                "branch": "戌",
                "sipseong": "비견",
                "sipseong_branch": "편인",
                "sibi_unseong": "건록",
                "start_year": 1992, 
                "end_year": 2001, 
                "start_age": 4, 
                "end_age": 13
            },
            ...
        ]
    """
    if not daewoon_list or not isinstance(daewoon_list, list) or len(daewoon_list) == 0:
        return []
    
    if first_luck_age is None or first_luck_age < 0:
        return []
    
    result = []
    daewoon_duration = 10  # 각 대운은 10년씩 지속
    
    for idx, daewoon in enumerate(daewoon_list):
        start_age = first_luck_age + (idx * daewoon_duration)
        end_age = start_age + daewoon_duration - 1
        
        # 간지에서 천간과 지지 추출
        stem = stem_from_any(daewoon) if daewoon else None
        branch = branch_from_any(daewoon) if daewoon else None
        
        item = {
            "age_range": f"{start_age}-{end_age}",
            "start_age": start_age,
            "end_age": end_age,
            "daewoon": daewoon,
            "stem": stem,
            "branch": branch
        }
        
        # 생년월일이 있으면 년도 정보도 계산
        if birth_year is not None:
            start_year = birth_year + start_age
            end_year = birth_year + end_age
            item["year_range"] = f"{start_year}-{end_year}"
            item["start_year"] = start_year
            item["end_year"] = end_year
        
        # 일간이 있으면 십성과 십이운성 계산
        if day_stem_hj and daewoon:
            try:
                # 천간/지지 십성 계산
                sipseong, sipseong_branch = _sipseong_split_for_target(day_stem_hj, daewoon)
                item["sipseong"] = sipseong
                item["sipseong_branch"] = sipseong_branch
                
                # 십이운성 계산 (지지 기반)
                if branch:
                    sibi_unseong = unseong_for(day_stem_hj, branch)
                    item["sibi_unseong"] = sibi_unseong
            except Exception as e:
                print(f"[calculate_daewoon_by_age] ⚠️ 십성/십이운성 계산 실패: {e}")
                # 계산 실패해도 기본 정보는 유지
        
        result.append(item)
    
    return result

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
    session_id = data.get("session_id") or "single_global_session"  # 필요 시 요청에서 받기
    
    # ✅ [NEW] 모드 구분 (saju / fortune)
    mode = (data.get("mode") or "saju").strip().lower()

    # ✅ [NEW] 대운 정보 (배열 형태 지원)
    daewoon_raw = data.get("daewoon")
    if isinstance(daewoon_raw, list):
        daewoon = daewoon_raw  # 배열 그대로 사용
        daewoon_str = ", ".join(daewoon_raw)  # 로그/표시용 문자열
    else:
        daewoon = daewoon_raw or ""  # 기존 문자열 형태
        daewoon_str = daewoon_raw or ""
    
    current_dw = data.get("currentDaewoon", "") or "" # 문자열/간지표현일 수 있음
    
    # ✅ [NEW] 대운 시작 나이
    first_luck_age = data.get("firstLuckAge")
    if first_luck_age is not None:
        try:
            first_luck_age = int(first_luck_age)
        except (ValueError, TypeError):
            first_luck_age = None

    # 사주 원국 기둥 (키가 없을 수 있으니 dict.get 사용)
    year        = sajuganji.get("년주", "") or ""
    month       = sajuganji.get("월주", "") or ""
    day         = sajuganji.get("일주", "") or ""
    pillar_hour = sajuganji.get("시주", "") or ""      # ❗ time 변수명 피함

    # ✅ 십성 정보 (sipseong_info 객체 또는 개별 필드 지원)
    sipseong_info = data.get("sipseong_info") or {}
    
    # sipseong_info 객체가 있으면 우선 사용, 없으면 기존 개별 필드 사용
    yinYang = sipseong_info.get("yinYang") or data.get("yinYang", "") or ""
    fiveElement = sipseong_info.get("fiveElement") or data.get("fiveElement", "") or ""
    
    # 년간/년지
    yearGan = sipseong_info.get("yearGan") or sipseong_info.get("년간") or data.get("yearGan") or ""
    yearJi  = sipseong_info.get("yearJi") or sipseong_info.get("년지") or data.get("yearJi") or ""
    
    # 월간/월지
    wolGan  = sipseong_info.get("wolGan") or sipseong_info.get("월간") or data.get("wolGan") or ""
    wolJi   = sipseong_info.get("wolJi") or sipseong_info.get("월지") or data.get("wolJi") or ""
    
    # 일간/일지
    ilGan   = sipseong_info.get("ilGan") or sipseong_info.get("일간") or data.get("ilGan") or ""
    ilJi    = sipseong_info.get("ilJi") or sipseong_info.get("일지") or data.get("ilJi") or ""
    
    # ✅ 일간이 없거나 "일간"이라는 라벨이면 일주에서 추출 (fallback)
    # Flutter에서 올바른 일간을 전송하므로, 이 로직은 fallback으로만 사용
    if (not ilGan or ilGan == "일간") and day:
        try:
            extracted_ilGan = stem_from_any(day)  # 일주에서 천간 추출
            if extracted_ilGan:
                ilGan = extracted_ilGan
                print(f"[make_saju_payload] ⚠️ 일간이 비어있어 일주({day})에서 추출: {ilGan}")
        except Exception as e:
            print(f"[make_saju_payload] ⚠️ 일주에서 일간 추출 실패: {e}")
    
    # 시간/시지
    siGan   = sipseong_info.get("siGan") or sipseong_info.get("시간") or data.get("siGan") or ""
    siJi    = sipseong_info.get("siJi") or sipseong_info.get("시지") or data.get("siJi") or ""
    
    # 대운간/대운지
    currDwGan = sipseong_info.get("currDaewoonGan") or sipseong_info.get("대운간") or data.get("currDaewoonGan", "") or ""
    currDwJi  = sipseong_info.get("currDaewoonJi") or sipseong_info.get("대운지") or data.get("currDaewoonJi", "") or ""
    print(f"make_saju_payload] 간지정보 확인 : siGan={siGan} currDwGan={currDwGan} wolGan = {wolGan}")
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
    # ✅ ilGan이 비어있으면 None 반환 (에러 방지)
    try:
        day_stem_hj = _norm_stem(ilGan) if ilGan else None
    except ValueError as e:
        print(f"[make_saju_payload] ⚠️ 일간 정규화 실패: {e}, ilGan={ilGan}")
        day_stem_hj = None

    
    # None이 섞여 있어도 pillars_unseong 내부에서 처리됨
    # 타겟(연/월/일/시) 십이운성 맵
    target_sibi_map = pillars_unseong(day_stem_hj, pillars_branches) if day_stem_hj else {k: None for k in pillars_branches}
    # 예: {'year': '관대', 'month': '절', 'day': None, 'hour': '장생'}

    # === [B] 현재 대운 십이운성 (★ _branch_of → branch_from_any)
    print(f"current_dw : {current_dw}")
    current_dw_branch = branch_from_any(current_dw)  # 예: '亥' 또는 None
    curr_dw_sibi = unseong_for(day_stem_hj, current_dw_branch) if (day_stem_hj and current_dw_branch) else None
    
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

    # === [NEW] daewoon_by_age 계산 (payload 생성 전에 미리 계산) ===
    import re
    birth_year = _extract_birth_year(data.get("birth") or data.get("birthday") or "")
    daewoon_by_age = calculate_daewoon_by_age(
        daewoon if isinstance(daewoon, list) else [],
        first_luck_age,
        birth_year,
        day_stem_hj
    )
    
    # === [NEW] 질문에서 년도를 추출하여 daewoon_by_age에서 대운 찾기 ===
    # 중요: 특정 년도를 언급한 질문이면, 그 년도에 해당하는 대운을 반드시 사용해야 함
    # 예: "2007년 사주" → 2007년에 해당하는 대운을 찾아서 사용 (나이와 무관하게 해당 년도의 대운 사용)
    # 예: "2008년에 나 무슨 대운 이였지?" → 2008년에 해당하는 대운을 찾아서 사용
    # 예: "88년", "21년" 같은 2자리 년도도 인식 (1988년, 2021년으로 해석)
    # 50 이상: 1900년대 (예: 88 → 1988), 50 미만: 2000년대 (예: 21 → 2021)
    # ⚠️ 주의: updated_question은 "2005년"이 "乙酉년"으로 변환될 수 있으므로, 원본 question에서 먼저 추출
    
    # === [NEW] 질문에서 년도를 추출하여 daewoon_by_age에서 대운 찾기 ===
    # 원본 question에서 먼저 년도 추출 (변환 전 원문이므로 년도 숫자가 그대로 있음)
    # 또한 msg_keywords에서도 년도 추출 시도 (LLM이 이미 추출한 경우)
    # print(f"[make_saju_payload] 🔍 년도 추출 시도: question='{question}', updated_question='{updated_question}'")

    # 1) 원본 question에서 4자리
    year_numbers_4digit = re.findall(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)', question)
    print(f"[make_saju_payload] 🔍 4자리 년도 추출 (원본): {year_numbers_4digit}")
    if not year_numbers_4digit:
        year_numbers_4digit = re.findall(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)', updated_question)
        print(f"[make_saju_payload] 🔍 4자리 년도 추출 (변환): {year_numbers_4digit}")

    # 2) 원본 question에서 2자리 (년/년도 붙은 것만)
    year_numbers_2digit = re.findall(r'(?<!\d)(\d{2})\s*(?:년|년도)(?!\d)', question)
    print(f"[make_saju_payload] 🔍 2자리 년도 추출 (원본): {year_numbers_2digit}")
    if not year_numbers_2digit:
        year_numbers_2digit = re.findall(r'(?<!\d)(\d{2})\s*(?:년|년도)(?!\d)', updated_question)
        print(f"[make_saju_payload] 🔍 2자리 년도 추출 (변환): {year_numbers_2digit}")

    # 2) msg_keywords에서도 년도 추출 시도 (LLM이 이미 추출한 경우)
    msg_keywords = data.get("msg_keywords") or []
    if msg_keywords:
        for kw in msg_keywords:
            # 4자리 년도 패턴
            kw_years_4 = re.findall(r'(?<!\d)(19\d{2}|20\d{2})(?!\d)', str(kw))
            if kw_years_4:
                year_numbers_4digit.extend(kw_years_4)
                print(f"[make_saju_payload] 🔍 msg_keywords에서 4자리 년도 추출: {kw_years_4}")
            # 2자리 년도 패턴 (숫자만 있는 경우, 2000년대로 가정)
            kw_years_2 = re.findall(r'^(\d{2})$', str(kw))
            if kw_years_2:
                # 2자리 년도를 4자리로 변환 (2000년대로 가정)
                for yy_str in kw_years_2:
                    try:
                        yy = int(yy_str)
                        full_year = 2000 + yy  # 2000년대로 가정
                        year_numbers_4digit.append(str(full_year))
                        print(f"[make_saju_payload] 🔍 msg_keywords에서 2자리 년도 변환: {yy_str} → {full_year}")
                    except (ValueError, TypeError):
                        continue
    
    # 2자리 년도를 4자리로 변환 (2000년대로 가정)
    year_numbers_2digit_converted = []
    for yy_str in year_numbers_2digit:
        try:
            yy = int(yy_str)
            full_year = 2000 + yy  # 2000년대로 가정
            year_numbers_2digit_converted.append(str(full_year))
            print(f"[make_saju_payload] 🔍 2자리 년도 변환: {yy_str} → {full_year}")
        except (ValueError, TypeError):
            continue
    
    # 최종 년도 리스트 (중복 제거)
    year_numbers = list(dict.fromkeys(year_numbers_4digit + year_numbers_2digit_converted))
    print(f"[make_saju_payload] 🔍 최종 추출된 년도: {year_numbers}, daewoon_by_age 개수: {len(daewoon_by_age) if daewoon_by_age else 0}")
    
    matched_daewoon = None
    if year_numbers and daewoon_by_age:
        for year_str in year_numbers:
            try:
                target_year = int(year_str)
                for item in daewoon_by_age:
                    start_year = item.get("start_year")
                    end_year = item.get("end_year")
                    if start_year is not None and end_year is not None:
                        if start_year <= target_year <= end_year:
                            matched_daewoon = item
                            print(f"[make_saju_payload] ✅ {target_year}년 대운 매칭: {item.get('daewoon')} ({item.get('year_range')})")
                            break
                if matched_daewoon:
                    break
            except (ValueError, TypeError):
                continue
    
    # 대운이 매칭되었으면 current_dw와 십성 정보를 확실하게 업데이트
    if matched_daewoon:
        matched_dw_ganji = matched_daewoon.get("daewoon")
        if matched_dw_ganji:
            current_dw = matched_dw_ganji
            # 대운 십성/십이운성을 무조건 재계산하여 확실하게 설정 (일간 기준)
            if day_stem_hj:
                dw_sip_gan_new, dw_sip_br_new = _sipseong_split_for_target(day_stem_hj, matched_dw_ganji)
                # 재계산된 값으로 확실하게 업데이트
                dw_sip_gan = dw_sip_gan_new
                dw_sip_br = dw_sip_br_new
                # 십이운성 재계산
                matched_dw_branch = matched_daewoon.get("branch")
                if matched_dw_branch:
                    curr_dw_sibi = unseong_for(day_stem_hj, matched_dw_branch) if day_stem_hj else None
                else:
                    # branch가 없으면 간지에서 추출
                    matched_dw_branch = branch_from_any(matched_dw_ganji)
                    if matched_dw_branch:
                        curr_dw_sibi = unseong_for(day_stem_hj, matched_dw_branch) if day_stem_hj else None
        print(f"[make_saju_payload] ✅ 대운 정보 업데이트: {current_dw} (십성: {dw_sip_gan}/{dw_sip_br}, 십이운성: {curr_dw_sibi})")

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
        "mode": mode,  # ✅ [NEW] saju / fortune 모드 구분
        "meta": {
            "user_name": user_name,
            "daewoon": daewoon,  # 배열 또는 문자열
            "daewoon_list": daewoon if isinstance(daewoon, list) else None,  # ✅ [NEW] 대운 배열 (배열인 경우만)
            "first_luck_age": first_luck_age,  # ✅ [NEW] 대운 시작 나이
            "daewoon_by_age": daewoon_by_age,  # ✅ [NEW] 나이대별 대운 정보 (년도, 십성, 십이운성 포함) - 위에서 계산된 값 재사용
            "yinYang": yinYang,
            "fiveElement": fiveElement,
            "session_id": session_id,      # 필요 시 상위에서 실제 세션 주입
            "question": question,
            
            # ✅ [NEW] 십성 정보 (상세)
            "sipseong_detail": {
                "yearGan": yearGan,
                "yearJi": yearJi,
                "wolGan": wolGan,
                "wolJi": wolJi,
                "ilGan": ilGan,
                "ilJi": ilJi,
                "siGan": siGan,
                "siJi": siJi,
                "currDaewoonGan": currDwGan,
                "currDaewoonJi": currDwJi,
            },

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
    print(f"focus meta 대운 : ${daewoon_str}, 대운시작나이: {first_luck_age}, 모드: {mode}")
    
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
    
    # 대운이 매칭되었으면 resolved.flow_now.daewoon도 업데이트
    resolved_daewoon_ganji = current_dw
    resolved_daewoon_stem = stem_from_any(current_dw) if current_dw else None
    resolved_daewoon_branch = branch_from_any(current_dw) if current_dw else None
    resolved_daewoon_sipseong = dw_sip_gan
    resolved_daewoon_sipseong_branch = dw_sip_br
    resolved_daewoon_sibi_unseong = curr_dw_sibi
    
    if matched_daewoon:
        resolved_daewoon_ganji = matched_daewoon.get("daewoon")
        resolved_daewoon_stem = stem_from_any(resolved_daewoon_ganji) if resolved_daewoon_ganji else None
        resolved_daewoon_branch = branch_from_any(resolved_daewoon_ganji) if resolved_daewoon_ganji else None
        
        # 대운 십성/십이운성을 무조건 재계산하여 확실하게 설정 (일간 기준)
        if resolved_daewoon_ganji and day_stem_hj:
            dw_sip_gan_new, dw_sip_br_new = _sipseong_split_for_target(day_stem_hj, resolved_daewoon_ganji)
            # 재계산된 값으로 확실하게 업데이트
            resolved_daewoon_sipseong = dw_sip_gan_new
            resolved_daewoon_sipseong_branch = dw_sip_br_new
            # 십이운성 재계산
            if resolved_daewoon_branch:
                resolved_daewoon_sibi_unseong = unseong_for(day_stem_hj, resolved_daewoon_branch) if day_stem_hj else None
            else:
                resolved_daewoon_sibi_unseong = None
        else:
            # 일간이 없으면 matched_daewoon의 기존 값 사용
            resolved_daewoon_sipseong = matched_daewoon.get("sipseong")
            resolved_daewoon_sipseong_branch = matched_daewoon.get("sipseong_branch")
            resolved_daewoon_sibi_unseong = matched_daewoon.get("sibi_unseong")
    
    payload["resolved"] = {
        "pillars": resolved_pillars,
        "flow_now": {
            "daewoon": {
                "ganji": resolved_daewoon_ganji or None,
                "stem":  resolved_daewoon_stem or None,
                "branch": resolved_daewoon_branch or None,
                "sipseong":        resolved_daewoon_sipseong or None,   # ✅ 대운 '천간' 기준 십성
                "sipseong_branch": resolved_daewoon_sipseong_branch or None,    # ✅ 대운 '지지' 기준 십성 (신규)
                "sibi_unseong":    resolved_daewoon_sibi_unseong or None, # 대운 십이운성 (지지 기반)
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
