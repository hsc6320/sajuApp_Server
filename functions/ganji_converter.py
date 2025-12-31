import json
from datetime import datetime
import os

# 십간/십이지 및 한자 매핑
gan_list = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
ji_list = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]

gan_list_hanja = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
ji_list_hanja = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

# 60갑자 리스트
ganji60 = [
    '갑자', '을축', '병인', '정묘', '무진', '기사', '경오', '신미', '임신', '계유',
    '갑술', '을해', '병자', '정축', '무인', '기묘', '경진', '신사', '임오', '계미',
    '갑신', '을유', '병술', '정해', '무자', '기축', '경인', '신묘', '임진', '계사',
    '갑오', '을미', '병신', '정유', '무술', '기해', '경자', '신축', '임인', '계묘',
    '갑진', '을사', '병오', '정미', '무신', '기유', '경술', '신해', '임자', '계축',
    '갑인', '을묘', '병진', '정사', '무오', '기미', '경신', '신유', '임술', '계해'
]

# 한글 간지 → 한자 간지
def convert_ganji_to_hanja(ganji):
    ganji = ganji.strip()
    if len(ganji) != 2:
        return ganji

    gan = ganji[0]
    ji = ganji[1]

    try:
        gan_index = gan_list.index(gan)
        ji_index = ji_list.index(ji)
        return gan_list_hanja[gan_index] + ji_list_hanja[ji_index]
    except ValueError:
        return ganji


# 현재 파일 위치 기준으로 JSON 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CURRENT_DIR, "converted.json")

# 전역 캐시
_JSON_CACHE = {}

def _get_cached_json(json_path: str):
    if json_path not in _JSON_CACHE:
        with open(json_path, "r", encoding="utf-8") as f:
            _JSON_CACHE[json_path] = json.load(f)
    return _JSON_CACHE[json_path]

def get_year_ganji_from_json(date: datetime, json_path: str = JSON_PATH) -> str:
    json_data = _get_cached_json(json_path)

     # ✅ 범위 가드 (JSON 커버리지 밖이면 명확히 예외)
    min_base = min(datetime.strptime(e["양력기준일"], "%Y-%m-%d") for e in json_data)
    max_base = max(datetime.strptime(e["양력기준일"], "%Y-%m-%d") for e in json_data)
    assert min_base <= date <= max_base, (
        f"간지 조회 범위 초과: {date.date()} (지원: {min_base.date()}~{max_base.date()})"
    )
    
    for entry in reversed(json_data):
        base_date = datetime.strptime(entry["양력기준일"], "%Y-%m-%d")
        if date >= base_date:
            ganji = entry["년주"]
            return convert_ganji_to_hanja(ganji)



import json
from datetime import datetime

solar_terms = [
    {"name": "입춘", "month": 2, "day": 4},
    {"name": "경칩", "month": 3, "day": 5},
    {"name": "청명", "month": 4, "day": 5},
    {"name": "입하", "month": 5, "day": 5},
    {"name": "망종", "month": 6, "day": 6},
    {"name": "소서", "month": 7, "day": 7},
    {"name": "입추", "month": 8, "day": 8},
    {"name": "백로", "month": 9, "day": 8},
    {"name": "한로", "month": 10, "day": 8},
    {"name": "입동", "month": 11, "day": 8},
    {"name": "대설", "month": 12, "day": 7},
    {"name": "소한", "month": 1, "day": 5},  # 마지막 (월 인덱스 = 11)
]

# 월간+월지 (월주) 테이블: 12절기 기준 12개월 × 5 그룹
month_stem_table = [
    ['丙寅', '丁卯', '戊辰', '己巳', '庚午', '辛未', '壬申', '癸酉', '甲戌', '乙亥', '丙子', '丁丑'],  # 甲, 己
    ['戊寅', '己卯', '庚辰', '辛巳', '壬午', '癸未', '甲申', '乙酉', '丙戌', '丁亥', '戊子', '己丑'],  # 乙, 庚
    ['庚寅', '辛卯', '壬辰', '癸巳', '甲午', '乙未', '丙申', '丁酉', '戊戌', '己亥', '庚子', '辛丑'],  # 丙, 辛
    ['壬寅', '癸卯', '甲辰', '乙巳', '丙午', '丁未', '戊申', '己酉', '庚戌', '辛亥', '壬子', '癸丑'],  # 丁, 壬
    ['甲寅', '乙卯', '丙辰', '丁巳', '戊午', '己未', '庚申', '辛酉', '壬戌', '癸亥', '甲子', '乙丑'],  # 戊, 癸
]



def get_year_group_index(year_stem: str) -> int:
    match year_stem:
        case '갑' | '기':
            return 0
        case '을' | '경':
            return 1
        case '병' | '신':
            return 2
        case '정' | '임':
            return 3
        case '무' | '계':
            return 4
        case _:
            return -1

def get_wolju_from_date(
    solar_date: datetime,
    json_path: str = JSON_PATH,
    month_only: bool = False,   # ✅ 월-only 질문용 플래그 추가
) -> str | None:
    """
    solar_date: 기준 양력 날짜
    month_only: True면 '25년 12월 직업운' 같이 '월 전체' 질문으로 보고,
                해당 연/월에 시작하는 양력기준일 레코드를 우선 사용한다.
                False면 기존처럼 '해당 일자 기준'으로 가장 가까운 과거 기준일을 사용.
    """
    json_data = _get_cached_json(json_path)

    selected_item = None
    closest_solar_date = None

    # -----------------------------
    # 1) 월-only 모드: 같은 연도+월의 양력기준일 레코드 우선 찾기
    # -----------------------------
    if month_only:
        for item in json_data:
            try:
                current_solar = datetime.strptime(item["양력기준일"], "%Y-%m-%d")
            except Exception:
                continue

            if (
                current_solar.year == solar_date.year
                and current_solar.month == solar_date.month
            ):
                selected_item = item
                closest_solar_date = current_solar
                break

    # -----------------------------
    # 2) fallback 또는 일반 모드: 기존 로직
    #    (양력기준일 <= solar_date 중 가장 최근)
    # -----------------------------
    if selected_item is None:
        for item in json_data:
            try:
                current_solar = datetime.strptime(item["양력기준일"], "%Y-%m-%d")
                if current_solar > solar_date:
                    continue
                if closest_solar_date is None or current_solar > closest_solar_date:
                    closest_solar_date = current_solar
                    selected_item = item
            except Exception:
                continue

    if selected_item is None or closest_solar_date is None:
        return None

    # 1. 년간 추출
    year_stem = selected_item["년주"].strip()[0]
    group_index = get_year_group_index(year_stem)
    if group_index == -1:
        return None

    # 2. 절기 기준으로 월 인덱스 계산
    # 소한(1월 5일) 이전의 날짜는 전년도 12월(month_index=11)로 처리
    sohan_date = datetime(solar_date.year, 1, 5)
    if solar_date < sohan_date:
        month_index = 11  # 전년도 12월
    else:
        # 소한 이후부터 각 절기를 순회하며 해당하는 월 인덱스 찾기
        month_index = -1
        # 입력 날짜 이하의 절기 중 가장 큰 절기를 찾아야 함
        # 소한은 전년도 12월을 의미하므로, 소한 이후 날짜에서는 소한을 제외하고 찾아야 함
        # 역순으로 순회하되 소한(인덱스 11)은 제외
        for i in range(len(solar_terms) - 2, -1, -1):  # 역순 순회 (소한 제외: len-2부터 0까지)
            term = solar_terms[i]
            term_year = solar_date.year
            term_date = datetime(term_year, term["month"], term["day"])
            if solar_date >= term_date:
                month_index = i
                break  # 역순이므로 첫 번째로 만족하는 것이 가장 큰 절기
        
        # 만약 입춘~대설 중 어떤 절기도 만족하지 않으면 소한을 선택 (1월 5일 ~ 입춘 전)
        if month_index == -1:
            month_index = 11  # 소한 (전년도 12월)
    
    # 4) 테이블에서 '월주(천간+지지)'를 그대로 반환
    wolju = month_stem_table[group_index][month_index]
    print(f"[월주 계산] 최종 결과: {wolju}")
    return wolju





# 기준 JSON 데이터 불러오기
def get_base_json_item(solar_date: datetime, json_path= JSON_PATH) -> dict:
    json_data = _get_cached_json(json_path)

    closest_data = None
    closest_date = None

    for item in json_data:
        item_date = datetime.strptime(item["양력기준일"], "%Y-%m-%d")
        if item_date <= solar_date:
            if closest_date is None or item_date > closest_date:
                closest_date = item_date
                closest_data = item

    if closest_data is None:
        raise Exception("기준일을 찾을 수 없습니다.")

    return closest_data


# 일주 계산 함수
def get_ilju(solar_date: datetime, json_path="converted.json") -> str:
    item = get_base_json_item(solar_date, json_path)
    base_ilju = item["일주"].strip()
    base_date = datetime.strptime(item["양력기준일"], "%Y-%m-%d")
    base_index = ganji60.index(base_ilju)

    diff_days = (solar_date - base_date).days
    ilju_index = (base_index + diff_days) % 60
    ilju = ganji60[ilju_index].strip()
    print(f"ganji60[{ilju_index}] : {ilju}")
    return convert_ganji_to_hanja(ilju)


def resolve_two_digit_year(year_suffix: int,
                           today: datetime | None = None,
                           prefer_past_on_tie: bool = True) -> int:
    """
    두 자리 연도(0..99)를 현재 날짜 기준 '가장 가까운 4자리 연도'로 해석.
    예) 2025년 현재:
        24 -> 2024, 26 -> 2026, 99 -> 1999, 00 -> 2000
    """
    if not (0 <= year_suffix <= 99):
        raise ValueError("year_suffix must be in [0, 99]")

    today = today or datetime.now()
    current_year = today.year
    century = (current_year // 100) * 100

    candidates = [
        century - 100 + year_suffix,  # 이전 세기
        century + year_suffix,        # 현재 세기
        century + 100 + year_suffix,  # 다음 세기
    ]

    # 가장 가까운 연도 선택 (절대차 최소)
    diffs = [abs(y - current_year) for y in candidates]
    min_diff = min(diffs)
    closest = [y for y, d in zip(candidates, diffs) if d == min_diff]

    if len(closest) == 1:
        return closest[0]
    # 동률이면 과거/미래 선호 규칙
    return max(closest) if not prefer_past_on_tie else min(closest)

# 모듈 로드 시 한 번만 계산 (캐시 활용)
def _json_year_bounds(json_path: str) -> tuple[int, int]:
    # 직접 import json 등 제거, 상단 import 활용
    data = _get_cached_json(json_path)

    # 양력기준일이 오름차순/내림차순 어떤 상태든 안전하게 min/max 계산
    years = [datetime.strptime(e["양력기준일"], "%Y-%m-%d").year for e in data]
    return min(years), max(years)


from datetime import datetime
from typing import Optional, Literal

Scope = Literal["year", "month", "day", "hour"]
