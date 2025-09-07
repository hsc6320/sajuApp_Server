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

def get_year_ganji_from_json(date: datetime, json_path: str = JSON_PATH) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

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

def get_wolju_from_date(solar_date: datetime, json_path: str = JSON_PATH) -> str | None:
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    closest_solar_date = None
    selected_item = None

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

    print(f"음력 변환 일 : {selected_item}")

    # 1. 년간 추출
    year_stem = selected_item["년주"].strip()[0]
    group_index = get_year_group_index(year_stem)
    if group_index == -1:
        return None

    # 2. 절기 기준으로 월 인덱스 계산
    month_index = -1
    for i, term in enumerate(solar_terms):
        term_year = solar_date.year
        if i == 11:  # 소한 이전일 경우는 전년도 12월 처리
            term_year += 1
        term_date = datetime(term_year, term["month"], term["day"])
        if solar_date >= term_date:
            month_index = i

    if month_index == -1:
        month_index = 11
    
    # 4) 테이블에서 '월주(천간+지지)'를 그대로 반환  ✅ 핵심
    wolju = month_stem_table[group_index][month_index]
    return wolju




# 기준 JSON 데이터 불러오기
def get_base_json_item(solar_date: datetime, json_path= JSON_PATH) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

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