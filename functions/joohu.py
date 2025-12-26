# -*- coding: utf-8 -*-
"""
조후(調候) 계산 모듈

조후(調候)란?
- 사주명리학에서 월령(月令, 월주 지지)과 전체 지지 분포를 기반으로
  한열조습(寒熱燥濕)의 균형을 판단하는 전통 이론
- 일간 중심의 십이운성과 달리, 사주 전체의 환경적·구조적 균형을 평가

역할 및 설계 원칙:
- 십성/십이운성의 보조 해석 레이어 (주해석이 아님)
- "왜 체감이 다른지"에 대한 구조적 설명 제공
- 결론 생성이 아닌 해석 보정용으로만 사용
- 십성/십이운성의 결론을 뒤집거나 부정하지 않음

출처:
- 전통 명리학 조후 이론 (《연해자평》, 《적천수》 등)
- 월령 기반 계절 특성과 오행 분포를 종합한 판단 기준
"""

from typing import Optional, Dict, List
from Sipsin import ji_to_element, five_element_map, branch_from_any, stem_from_any


# ============================================================================
# 월령(월주 지지)의 계절 특성 정의
# ============================================================================
# 조후 판단의 핵심: 월령이 어떤 계절에 해당하는지에 따라 필요한 기운이 다름
# 
# 겨울(한/寒): 추운 계절 → 따뜻한 화(火) 기운 필요
# 여름(열/熱): 더운 계절 → 시원한 수(水) 기운 필요  
# 가을(조/燥): 건조한 계절 → 습(濕) 기운 필요
# 봄(습/濕): 습한 계절 → 건조한 조(燥) 기운 필요

WINTER_BRANCHES = ['子', '丑', '亥']  # 겨울 지지: 한(寒) → 화(火) 필요
SUMMER_BRANCHES = ['午', '未', '巳']  # 여름 지지: 열(熱) → 수(水) 필요
AUTUMN_BRANCHES = ['申', '酉', '戌']  # 가을 지지: 조(燥) → 습(濕) 필요
SPRING_BRANCHES = ['寅', '卯', '辰']  # 봄 지지: 습(濕) → 조(燥) 필요

# ============================================================================
# 오행별 한열조습 특성 (참고용 - 현재 로직에서는 직접 사용하지 않음)
# ============================================================================
# 화(火) = 열(熱): 따뜻함, 활력
# 수(水) = 한(寒): 차가움, 냉정
# 목(木) = 습(濕): 습기, 생장
# 금(金) = 조(燥): 건조, 수렴
# 토(土) = 중성: 다른 오행과 조합에 따라 조/습 결정

FIRE_ELEMENTS = ['丙', '丁', '午', '巳']  # 화 = 열 (참고용)
WATER_ELEMENTS = ['壬', '癸', '子', '亥']  # 수 = 한 (참고용)
WOOD_ELEMENTS = ['甲', '乙', '寅', '卯']  # 목 = 습 (참고용)
METAL_ELEMENTS = ['庚', '辛', '申', '酉']  # 금 = 조 (참고용)


def _normalize_branch(branch: Optional[str]) -> Optional[str]:
    """
    지지를 한자로 표준화하는 내부 함수
    
    입력된 지지(한글/한자/간지 문자열)를 표준 한자 지지로 변환
    예: '자' → '子', '丙寅' → '寅', '子' → '子'
    
    Args:
        branch: 지지 문자열 (한글, 한자, 또는 간지 전체)
                예: '자', '子', '丙寅' 등
    
    Returns:
        표준화된 한자 지지 (예: '子', '寅') 또는 None (변환 실패 시)
    """
    if not branch:
        return None
    
    # 한글 지지 → 한자 지지 변환 테이블
    ko_to_hj = {
        '자': '子', '축': '丑', '인': '寅', '묘': '卯',
        '진': '辰', '사': '巳', '오': '午', '미': '未',
        '신': '申', '유': '酉', '술': '戌', '해': '亥'
    }
    
    branch = branch.strip()
    
    # 1) 한글 지지인 경우: 직접 변환
    if branch in ko_to_hj:
        return ko_to_hj[branch]
    
    # 2) 이미 한자 지지인 경우: 그대로 반환
    valid_hj_branches = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    if branch in valid_hj_branches:
        return branch
    
    # 3) 간지 문자열(예: '丙寅')인 경우: 마지막 글자(지지) 추출
    if len(branch) >= 2:
        last_char = branch[-1]  # 마지막 글자가 지지
        # 한자 지지인 경우
        if last_char in valid_hj_branches:
            return last_char
        # 한글 지지인 경우 변환
        if last_char in ko_to_hj:
            return ko_to_hj[last_char]
    
    return None


def _count_elements_in_pillars(pillars: Dict[str, Optional[str]]) -> Dict[str, int]:
    """
    사주 기둥(년/월/일/시)에서 오행 개수를 집계하는 내부 함수
    
    각 기둥의 천간과 지지에서 오행을 추출하여 전체 사주에서
    각 오행(화/수/목/금/토)이 몇 개씩 있는지 카운트
    
    Args:
        pillars: 사주 기둥 딕셔너리
                예: {"year": "甲子", "month": "丙寅", "day": "戊辰", "hour": "庚午"}
    
    Returns:
        오행별 개수 딕셔너리
        예: {'火': 1, '水': 1, '木': 2, '金': 1, '土': 1}
        - '火': 화(火) 오행 개수 (열/熱 기운)
        - '水': 수(水) 오행 개수 (한/寒 기운)
        - '木': 목(木) 오행 개수 (습/濕 기운)
        - '金': 금(金) 오행 개수 (조/燥 기운)
        - '土': 토(土) 오행 개수 (중성)
    """
    # 오행별 카운터 초기화
    element_count = {
        '火': 0,  # 화(火) = 열(熱) 기운
        '水': 0,  # 수(水) = 한(寒) 기운
        '木': 0,  # 목(木) = 습(濕) 기운
        '金': 0,  # 금(金) = 조(燥) 기운
        '土': 0,  # 토(土) = 중성 (조후 판단에서는 직접 사용하지 않음)
    }
    
    # 각 기둥(년/월/일/시)을 순회하며 오행 집계
    for pillar_name, ganji in pillars.items():
        if not ganji:
            continue  # 간지가 없으면 스킵
        
        # 1) 천간(天干)의 오행 추출 및 카운트
        # 예: "甲子"에서 "甲" → 목(木)
        stem = stem_from_any(ganji)
        if stem and stem in five_element_map:
            element = five_element_map[stem]
            element_count[element] = element_count.get(element, 0) + 1
        
        # 2) 지지(地支)의 오행 추출 및 카운트
        # 예: "甲子"에서 "子" → 수(水)
        branch = branch_from_any(ganji)
        if branch and branch in ji_to_element:
            element = ji_to_element[branch]
            element_count[element] = element_count.get(element, 0) + 1
    
    return element_count


def calculate_joohu(
    month_branch: Optional[str],
    pillars: Dict[str, Optional[str]]
) -> Dict[str, bool]:
    """
    조후(調候) 계산 핵심 함수
    
    월령(월주 지지)의 계절 특성과 전체 사주의 오행 분포를 종합하여
    한열조습(寒熱燥濕)의 균형 상태를 판단하고 필요한 기운을 제시
    
    판단 로직:
    1. 월령이 어떤 계절(겨울/여름/가을/봄)인지 확인
    2. 전체 사주(년/월/일/시)에서 각 오행 개수 집계
    3. 계절에 필요한 기운이 부족하거나 반대 기운이 과다하면 해당 flag를 True로 설정
    
    Args:
        month_branch: 월령(월주 지지) - 한자 또는 한글
                     예: '子', '자', '丙寅' (마지막 글자 '寅' 사용)
        pillars: 사주 기둥 딕셔너리
                예: {"year": "甲子", "month": "丙寅", "day": "戊辰", "hour": "庚午"}
    
    Returns:
        조후 flags 딕셔너리:
        {
            "need_warm": bool,   # 한(寒) → 화(火) 기운이 필요한 구조
                                 # (월령이 겨울이고 화 기운 부족 시 True)
            "need_cool": bool,   # 열(熱) → 수(水) 기운이 필요한 구조
                                # (월령이 여름이고 수 기운 부족 시 True)
            "need_dry": bool,    # 습(濕) → 조(燥) 기운이 필요한 구조
                                # (월령이 봄이고 금 기운 부족 시 True)
            "need_moist": bool,  # 조(燥) → 습(濕) 기운이 필요한 구조
                                # (월령이 가을이고 목 기운 부족 시 True)
            "is_balanced": bool, # 조후가 균형 잡혀 있는 구조
                                # (계절에 필요한 기운이 충분하여 조후가 만족되는 경우 True)
        }
    """
    # 기본값: 모든 flags를 False로 초기화
    # 조후가 균형 잡혀 있거나 판단 불가능한 경우 모두 False
    result = {
        "need_warm": False,   # 한(寒) → 화(火) 필요 여부
        "need_cool": False,   # 열(熱) → 수(水) 필요 여부
        "need_dry": False,    # 습(濕) → 조(燥) 필요 여부
        "need_moist": False,  # 조(燥) → 습(濕) 필요 여부
        "is_balanced": False, # 조후가 균형 잡혀 있는지 여부
    }
    
    # 월령(월주 지지) 정규화: 한글/한자/간지 문자열 → 표준 한자 지지
    normalized_month = _normalize_branch(month_branch)
    if not normalized_month:
        # 월령이 없거나 변환 실패 시 조후 판단 불가 → 모든 flags False 반환
        return result
    
    # 전체 사주(년/월/일/시)에서 오행 개수 집계
    # 각 기둥의 천간과 지지에서 오행을 추출하여 카운트
    element_count = _count_elements_in_pillars(pillars)
    fire_count = element_count.get('火', 0)    # 화(火) = 열(熱) 기운 개수
    water_count = element_count.get('水', 0)   # 수(水) = 한(寒) 기운 개수
    wood_count = element_count.get('木', 0)    # 목(木) = 습(濕) 기운 개수
    metal_count = element_count.get('金', 0)   # 금(金) = 조(燥) 기운 개수
    
    # ========================================================================
    # 1. 겨울(한/寒) 판단: 월령이 겨울이고 화(火) 기운이 부족하면 need_warm = True
    #                     충분하면 is_balanced = True
    # ========================================================================
    # 겨울(子, 丑, 亥)은 한기가 강하므로 따뜻한 화(火) 기운이 필요
    if normalized_month in WINTER_BRANCHES:
        # 조건 1: 화 기운이 전혀 없으면 → 한기 과다, 화 필요
        if fire_count == 0:
            result["need_warm"] = True
        # 조건 2: 화 1개, 수 2개 이상 → 한기가 화를 압도, 화 필요
        elif fire_count == 1 and (water_count >= 2):
            result["need_warm"] = True
        # 조건 3: 화 기운이 2개 이상이고 수 기운이 화보다 적거나 같으면 → 조후 만족
        elif fire_count >= 2 and (water_count <= fire_count):
            result["is_balanced"] = True
    
    # ========================================================================
    # 2. 여름(열/熱) 판단: 월령이 여름이고 수(水) 기운이 부족하면 need_cool = True
    #                     충분하면 is_balanced = True
    # ========================================================================
    # 여름(午, 未, 巳)은 열기가 강하므로 시원한 수(水) 기운이 필요
    if normalized_month in SUMMER_BRANCHES:
        # 조건 1: 수 기운이 전혀 없으면 → 열기 과다, 수 필요
        if water_count == 0:
            result["need_cool"] = True
        # 조건 2: 수 1개, 화 2개 이상 → 열기가 수를 압도, 수 필요
        elif water_count == 1 and (fire_count >= 2):
            result["need_cool"] = True
        # 조건 3: 수 기운이 2개 이상이고 화 기운이 수보다 적거나 같으면 → 조후 만족
        elif water_count >= 2 and (fire_count <= water_count):
            result["is_balanced"] = True
    
    # ========================================================================
    # 3. 가을(조/燥) 판단: 월령이 가을이고 습(濕) 기운이 부족하면 need_moist = True
    #                     충분하면 is_balanced = True
    # ========================================================================
    # 가을(申, 酉, 戌)은 조기가 강하므로 습한 목(木) 기운이 필요
    if normalized_month in AUTUMN_BRANCHES:
        # 조건 1: 목(습) 기운이 전혀 없으면 → 조기 과다, 목 필요
        if wood_count == 0:
            result["need_moist"] = True
        # 조건 2: 목 1개 이하, 금 2개 이상 → 조기가 습을 압도, 목 필요
        elif wood_count <= 1 and (metal_count >= 2):
            result["need_moist"] = True
        # 조건 3: 목 기운이 2개 이상이고 금 기운이 목보다 적거나 같으면 → 조후 만족
        elif wood_count >= 2 and (metal_count <= wood_count):
            result["is_balanced"] = True
    
    # ========================================================================
    # 4. 봄(습/濕) 판단: 월령이 봄이고 조(燥) 기운이 부족하면 need_dry = True
    #                   충분하면 is_balanced = True
    # ========================================================================
    # 봄(寅, 卯, 辰)은 습기가 강하므로 건조한 금(金) 기운이 필요
    if normalized_month in SPRING_BRANCHES:
        # 조건 1: 금(조) 기운이 전혀 없으면 → 습기 과다, 금 필요
        if metal_count == 0:
            result["need_dry"] = True
        # 조건 2: 금 1개 이하, 목 2개 이상 → 습기가 조를 압도, 금 필요
        elif metal_count <= 1 and (wood_count >= 2):
            result["need_dry"] = True
        # 조건 3: 금 기운이 2개 이상이고 목 기운이 금보다 적거나 같으면 → 조후 만족
        elif metal_count >= 2 and (wood_count <= metal_count):
            result["is_balanced"] = True
    
    return result


def get_joohu_flags(
    year: Optional[str] = None,
    month: Optional[str] = None,
    day: Optional[str] = None,
    hour: Optional[str] = None
) -> Dict[str, bool]:
    """
    사주 원국 기둥으로부터 조후 flags 계산 (편의 함수)
    
    이 함수는 make_saju_payload 등에서 직접 호출하기 쉽도록
    사주 기둥을 개별 파라미터로 받아 내부적으로 딕셔너리로 변환한 뒤
    calculate_joohu()를 호출하는 래퍼 함수
    
    Args:
        year: 년주 간지 (예: "甲子" 또는 "갑자")
        month: 월주 간지 (예: "丙寅" 또는 "병인")
               - 이 값에서 지지를 추출하여 월령으로 사용
        day: 일주 간지 (예: "戊辰" 또는 "무진")
        hour: 시주 간지 (예: "庚午" 또는 "경오")
    
    Returns:
        조후 flags 딕셔너리 (calculate_joohu()와 동일한 형식):
        {
            "need_warm": bool,   # 한(寒) → 화(火) 필요
            "need_cool": bool,   # 열(熱) → 수(水) 필요
            "need_dry": bool,    # 습(濕) → 조(燥) 필요
            "need_moist": bool,  # 조(燥) → 습(濕) 필요
            "is_balanced": bool, # 조후가 균형 잡혀 있는지
        }
    
    Example:
        >>> flags = get_joohu_flags("甲子", "丙寅", "戊辰", "庚午")
        >>> # 겨울(子) + 화 기운 부족 → need_warm = True 가능
    """
    # 월주 간지에서 지지(월령) 추출
    # 예: "丙寅" → "寅" (봄, 습)
    month_branch = branch_from_any(month) if month else None
    
    # 사주 기둥을 딕셔너리 형태로 구성
    pillars = {
        "year": year,   # 년주
        "month": month, # 월주 (월령 판단용)
        "day": day,     # 일주
        "hour": hour,   # 시주
    }
    
    # 핵심 계산 함수 호출
    return calculate_joohu(month_branch, pillars)

