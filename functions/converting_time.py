from datetime import datetime
import os
import re
from ganji_converter import get_wolju_from_date, get_year_ganji_from_json, get_ilju

# from ganji_converter import get_year_ganji_from_json

# 현재 파일 위치 기준으로 JSON 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CURRENT_DIR, "converted.json")

import re
from datetime import datetime, timedelta

K_NUM = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
}

def parse_korean_int(token: str) -> int | None:
    token = token.strip()
    if token.isdigit():
        return int(token)
    return K_NUM.get(token)



def handle_relative_day_keyword_with_ilju(
    question: str,
    token: str,                 # '오늘' | '내일' | '모레' | '글피'
    base_date: datetime,
    json_path: str,
    relative_to_ganji_map: dict,
    absolute_expressions: list
) -> None:
    delta_days_map = {"오늘":0, "내일":1, "모레":2, "글피":3}
    if token not in delta_days_map:
        return

    target_date = base_date + timedelta(days=delta_days_map[token])
    ty, tm = target_date.year, target_date.month

    # 절대 키워드(연/월) 누적
    if str(ty) not in absolute_expressions:
        absolute_expressions.append(str(ty))
    abs_month = f"{tm}월"
    if abs_month not in absolute_expressions:
        absolute_expressions.append(abs_month)

    # 간지 계산
    year_ganji = get_year_ganji_from_json(datetime(ty, 5, 1), json_path)  # 년주(간지만 표기)
    wolju      = get_wolju_from_date(datetime(ty, tm, 1), json_path)      # 월주(한자 2글자)
    ilju       = get_ilju(target_date, json_path)                          # 일주(한자 2글자)
    print(f"일주 계산 : {year_ganji}.{wolju}.{ilju}")

    # 질문 치환: 년 + 월 + 일 (월/일은 “~월/~일”처럼 표기)
    if wolju and ilju:
        relative_to_ganji_map[token] = f"{year_ganji}년 {wolju}월 {ilju}일"
    elif wolju:
        relative_to_ganji_map[token] = f"{year_ganji}년 {wolju}월"
    else:
        relative_to_ganji_map[token] = f"{year_ganji}년"


def handle_korean_month_offset(
    question: str,
    item: str,
    current_year: int,
    current_month: int,
    json_path: str,
    relative_to_ganji_map: dict,
    absolute_expressions: list
) -> bool:
    """
    '한달 후', '세 달 전', '3개월 후' 등 → 연간+월주 치환, 절대 연/월 추가
    반환: 처리했으면 True
    """
    # (한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|숫자) 달/개월 (뒤|후|전)
    m = re.search(r"(?<!\d)\s*(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|\d+)\s*(?:달|개월)\s*(뒤|후|전)\b", item)
    if not m:
        return False

    raw_n, direction = m.group(1), m.group(2)
    n = parse_korean_int(raw_n)
    if n is None:
        return False

    new_month = current_month
    new_year = current_year
    if direction in ("뒤", "후"):
        new_month += n
    else:
        new_month -= n

    while new_month > 12:
        new_month -= 12
        new_year += 1
    while new_month < 1:
        new_month += 12
        new_year -= 1

    # 절대 키워드
    if str(new_year) not in absolute_expressions:
        absolute_expressions.append(str(new_year))
    abs_month = f"{new_month}월"
    if abs_month not in absolute_expressions:
        absolute_expressions.append(abs_month)

    # 질문에서 실제 토큰을 키로 (공백/표기 차이 대응)
    tok_q = re.search(rf"\b{raw_n}\s*(?:달|개월)\s*(?:뒤|후|전)\b", question)
    token = tok_q.group(0) if tok_q else m.group(0)

    # 연간 + 월주로 치환
    ganji_year = get_year_ganji_from_json(datetime(new_year, 5, 1), json_path)
    wolju = get_wolju_from_date(datetime(new_year, new_month, 1), json_path)
    if wolju:
        relative_to_ganji_map[token] = f"{ganji_year}년 {wolju}월"
    else:
        relative_to_ganji_map[token] = f"{ganji_year}년"
    return True


def handle_month_in_item(
    item: str,
    target_year: int,
    json_path: str,
    relative_to_ganji_map: dict,
) -> str | None:
    """
    item에서 '7월', '12월' 등 월 표현을 찾아,
    - 해당 연도의 월주(干支)를 구해 치환 맵에 등록하고
    - absolute_expressions에 넣을 '7월' 형태 문자열을 리턴
    없으면 None 리턴
    """
    m = re.search(r"\b(\d{1,2})\s*월\b", item)
    if not m:
        return False

    month_num = int(m.group(1))
    if not (1 <= month_num <= 12):
        return False

    wolju = get_wolju_from_date(datetime(target_year, month_num, 1), json_path)
    if not wolju:
        return None

    original_token = m.group(0)              # '7월' 또는 '7 월'
    relative_to_ganji_map[original_token] = f"{wolju}월"  # 예: '癸未월'
    # relative_to_ganji_map[original_token] = replaced_value
    print(f"월주 키워드 변환 {wolju}월")

    return f"{wolju}월"  # absolute_expressions에 넣을 값


def convert_relative_time(question: str, expressions: list[str], current_year: int = None, current_month: int = None, current_day: int = None) -> list[str]:
    now = datetime.now()
    if current_year is None:
        current_year = now.year
    if current_month is None:
        current_month = now.month
    if current_day is None:
        current_day = now.day

    absolute_expressions = []
    relative_to_ganji_map = {}  # 👈 상대 표현 → 간지
    context_year = None


    #for item in expressions:
    for item in sorted((str(x).strip() for x in expressions), key=len, reverse=True):
        item =  str(item).strip()
        context_year = current_year

         # === 일 단위 키워드: 오늘/내일/모레/글피 ===
        if any(k in item for k in ("오늘", "내일", "모레", "글피")):
            for k in ("오늘", "내일", "모레", "글피"):
                if k in item:
                    handle_relative_day_keyword_with_ilju(
                        question=question,
                        token=k,
                        base_date=datetime(current_year, current_month, current_day),
                        json_path=JSON_PATH,
                        relative_to_ganji_map=relative_to_ganji_map,
                        absolute_expressions=absolute_expressions,
                    )

        # === 한/두/세/… 달 전·후 / 숫자 달·개월 전·후 ===
        if handle_korean_month_offset(
            question=question,
            item=item,
            current_year=current_year,
            current_month=current_month,
            json_path=JSON_PATH,
            relative_to_ganji_map=relative_to_ganji_map,
            absolute_expressions=absolute_expressions,
        ):
            continue  # 처리 끝나면 다음 item으로 넘어감

        # === 년 단위 상대 표현 ===
        if "내후년" in item:
            absolute_expressions.append(str(current_year + 2))
            context_year = current_year + 2
            ganji = get_year_ganji_from_json(datetime(current_year+2, 5, 1), JSON_PATH)
            relative_to_ganji_map["내후년"] = f"{ganji}년"

            print(f" '내후년' 간지변환 : {ganji}")
            if (abs_month := handle_month_in_item(item, current_year+2, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        elif "내년" in item:
            absolute_expressions.append(str(current_year + 1))
            context_year = current_year + 1
            ganji = get_year_ganji_from_json(datetime(current_year+1, 5, 1), JSON_PATH)
            relative_to_ganji_map["내년"] = f"{ganji}년"

            print(f" '내년' 간지변환 : {ganji}")
            if (abs_month := handle_month_in_item(item, current_year+1, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        elif "올해" in item:
            absolute_expressions.append(str(current_year))
            context_year = current_year
            ganji = get_year_ganji_from_json(datetime(current_year, 5, 1), JSON_PATH)
            relative_to_ganji_map["올해"] = f"{ganji}년"
            print(f" '올해' 간지변환 : {ganji}")
            # 월주는 get_wolju_from_date()로 바로 처리
            if (abs_month := handle_month_in_item(item, current_year, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)
            # if month_match := re.search(r"\d{1,2}월", item):
            #     absolute_expressions.append(month_match.group())

        elif "재작년" in item:
            absolute_expressions.append(str(current_year - 2))
            context_year = current_year - 2
            ganji = get_year_ganji_from_json(datetime(current_year-2, 5, 1), JSON_PATH)
            relative_to_ganji_map["재작년"] = f"{ganji}년"
            print(f"'재작년' 간지변환 : {ganji}")

            if (abs_month := handle_month_in_item(item, current_year-2, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        elif "작년" in item:
            absolute_expressions.append(str(current_year - 1))
            context_year = current_year - 1
            ganji = get_year_ganji_from_json(datetime(current_year-1, 5, 1), JSON_PATH)
            relative_to_ganji_map["작년"] = f"{ganji}년"
            print(f" '작년' 간지변환 : {ganji}")

            if (abs_month := handle_month_in_item(item, current_year-1, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        elif re.search(r"\d+\s*년\s*[뒤후전]", item):
            matches = re.findall(r"\d+", item)
            print(f"matches : {matches}")
            if not matches:
                print(f"⚠️ 숫자 추출 실패: '{item}' → 스킵됨")
                continue  # 또는 pass

            offset = int(matches[0])

            # 질문문에서 실제로 등장한 토큰을 잡아 키로 사용 (공백/표기 차이 대응)
            # ex) '3년후', '3년 후' 모두 포착 
            token_in_question = re.search(rf"\b{offset}\s*년\s*(뒤|후|전)\b", question)
            token = token_in_question.group(0) if token_in_question else item.strip()

            if "뒤" in token or "후" in token:
                ganji = get_year_ganji_from_json(datetime(current_year+offset, 5, 1), JSON_PATH)
                print(f"'뒤' '후' 간지변환 : {ganji}")
                relative_to_ganji_map[token] = f"{ganji}년"
                if (abs_month := handle_month_in_item(item, current_year+offset, JSON_PATH, relative_to_ganji_map)):
                    absolute_expressions.append(abs_month)
            elif "전" in token:
                ganji = get_year_ganji_from_json(datetime(current_year-offset, 5, 1), JSON_PATH)
                relative_to_ganji_map[token] = f"{ganji}년전"
                print(f"'전' 간지변환 : {ganji}")
                #relative_to_ganji_map["전"] = f"{ganji}년"
            if month_match := re.search(r"\d{1,2}월", item):
                absolute_expressions.append(month_match.group())
        
        elif (m2 := re.search(r"(?<!\d)(\d{2})\s*년\s*년도\b", item)):
            token_2digit = m2.group(0)                  # 실제 매칭된 원문: '24년' or '24 년'
            year_suffix = int(m2.group(1))

            # 현재 세기 계산
            century = (current_year // 100) * 100
            full_year = (
                century + year_suffix
                if year_suffix >= (current_year % 100)
                else century + 100 + year_suffix
            )
            absolute_expressions.append(str(full_year))

            # 연간 간지 치환
            ganji = get_year_ganji_from_json(datetime(full_year, 5, 1), JSON_PATH)
            relative_to_ganji_map[token_2digit] = f"{ganji}년"
            print(f"두자리 년도 치환 간지 정보 : {ganji}년")

            # 월까지 함께 있으면 월주 치환
            if (abs_month := handle_month_in_item(item, full_year, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        # === 단독 월 표현 처리 ===
        elif re.search(r"\b(\d{1,2})\s*월\b", item):
            print(f"단독 월 표현 처리 {item}")
            ty = context_year if context_year is not None else current_year
            if (abs_month := handle_month_in_item(item, current_year, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        # === 월 단위 상대 표현 === (예: 3개월 후 / 2달 전 / 3 달 후)
        elif (m := re.search(r"(?<!\d)(\d+)\s*(?:개월|달)\s*(뒤|후|전)\b", item)):
            offset = int(m.group(1))
            direction = m.group(2)  # '뒤' | '후' | '전'

            new_month = current_month
            new_year = current_year

            if direction in ("뒤", "후"):
                new_month += offset
            else:  # '전'
                new_month -= offset

            # 월 단위 오버플로우 조정
            while new_month > 12:
                new_month -= 12
                new_year += 1
            while new_month < 1:
                new_month += 12
                new_year -= 1

            # 절대 키워드 목록 (원하면 유지)
            absolute_expressions.append(str(new_year))
            absolute_expressions.append(f"{new_month}월")
            
            # 질문 치환용 토큰은 실제 질문에서 잡아서 쓰기 (공백/표기 차이 방지)
            token_in_question = re.search(rf"\b{offset}\s*(?:개월|달)\s*(?:뒤|후|전)\b", question)
            token = token_in_question.group(0) if token_in_question else m.group(0)

            # 연간/월주로 치환 (원하면 빼도 됨)
            ganji_year = get_year_ganji_from_json(datetime(new_year, 5, 1), JSON_PATH)
            wolju = get_wolju_from_date(datetime(new_year, new_month, 1), JSON_PATH)  # 예: '丙戌'
            
            # 🔹 핵심: 토큰을 "연간 + 월주"로 한 번에 치환
            if wolju:
                relative_to_ganji_map[token] = f"{ganji_year}년 {wolju}월"
            else:
                # 월주 계산 실패 시 최소 연간만이라도
                relative_to_ganji_map[token] = f"{ganji_year}년"

            # 월주 치환까지 하고 싶으면:
            if (abs_month := handle_month_in_item(f"{new_month}월", new_year, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        # ===== 명시적 연도(yyyy년) (+ 월) =====
        elif re.match(r"\d{4}년", item):
            year = re.match(r"\d{4}", item).group()
            absolute_expressions.append(year)
            if (abs_month := handle_month_in_item(item, year, JSON_PATH, relative_to_ganji_map)):
                absolute_expressions.append(abs_month)

        # === 그 외 ===
        else:
            absolute_expressions.append(item)

    # ===== 루프 종료 후: 월 누락 보정 =====
    # expressions 추출 단계에서 월을 못 잡은 경우를 대비해 question 전체에서 한 번 더 시도
    ty = context_year if context_year is not None else current_year
    if (abs_month := handle_month_in_item(question, ty, JSON_PATH, relative_to_ganji_map)):
        absolute_expressions.append(abs_month)

     # === 질문 텍스트에서 상대 표현을 간지로 치환 ===
    question_updated = question
    for old, new in relative_to_ganji_map.items():
        question_updated = question_updated.replace(old, new)


    return absolute_expressions, question_updated


import re
from typing import Optional, Tuple, List

# 천간/지지
STEMS_KO = ["갑","을","병","정","무","기","경","신","임","계"]
STEMS_HZ = list("甲乙丙丁戊己庚辛壬癸")
BRANCH_KO= ["자","축","인","묘","진","사","오","미","신","유","술","해"]
BRANCH_HZ= list("子丑寅卯辰巳午未申酉戌亥")

def to_hanzi_gan(gan_ko: str) -> str:
    if gan_ko in STEMS_KO:
        return STEMS_HZ[STEMS_KO.index(gan_ko)]
    return gan_ko

def to_hanzi_ji(ji_ko: str) -> str:
    if ji_ko in BRANCH_KO:
        return BRANCH_HZ[BRANCH_KO.index(ji_ko)]
    return ji_ko

def normalize_ganji(token: str) -> str:
    """공백/접미사 제거 + 한글 간지면 한자로 변환 → '乙巳' 형태로 통일"""
    t = re.sub(r"\s+", "", token)
    t = re.sub(r"[년월일시年月日時]", "", t)
    # 한글 간지 2글자 -> 한자로 교체
    if len(t) == 2 and t[0] in "".join(STEMS_KO) and t[1] in "".join(BRANCH_KO):
        return to_hanzi_gan(t[0]) + to_hanzi_ji(t[1])
    # 이미 한자라면 그대로
    return t

# 접미사 "필수"로 강제 (이전 버전과의 차이!)
YEAR_RX  = re.compile(r"(?:[갑을병정무기경신임계甲乙丙丁戊己庚辛壬癸]\s*[자축인묘진사오미신유술해子丑寅卯辰巳午未申酉戌亥])\s*(?:년|年)")
MONTH_RX = re.compile(r"(?:[갑을병정무기경신임계甲乙丙丁戊己庚辛壬癸]\s*[자축인묘진사오미신유술해子丑寅卯辰巳午未申酉戌亥])\s*(?:월|月)")
DAY_RX   = re.compile(r"(?:[갑을병정무기경신임계甲乙丙丁戊己庚辛壬癸]\s*[자축인묘진사오미신유술해子丑寅卯辰巳午未申酉戌亥])\s*(?:일|日)")
HOUR_RX  = re.compile(r"(?:[갑을병정무기경신임계甲乙丙丁戊己庚辛壬癸]\s*[자축인묘진사오미신유술해子丑寅卯辰巳午未申酉戌亥])\s*(?:시|時)")

def sexagenary_of_gregorian_year(year: int, prefer_hanzi: bool = True) -> str:
    """
    서기 연도 → 간지. 1984=甲子 기준.
    index = (year - 4) % 10/12
    """
    stem = (year - 4) % 10
    branch = (year - 4) % 12
    gan_ko = STEMS_KO[stem]
    ji_ko  = BRANCH_KO[branch]
    if prefer_hanzi:
        return STEMS_HZ[stem] + BRANCH_HZ[branch]
    return gan_ko + ji_ko

# def extract_target_ganji_v2(absolute_keywords: List[str], updated_question: str
#                              ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
# ── 간지 패턴: '甲申', '乙巳' 등
GANJI_RX = r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]"
YEAR_RX  = re.compile(rf"{GANJI_RX}\s*년")
MONTH_RX = re.compile(rf"{GANJI_RX}\s*월")
DAY_RX   = re.compile(rf"{GANJI_RX}\s*일")
HOUR_RX  = re.compile(rf"{GANJI_RX}\s*시")

def normalize_ganji(s: str) -> str:
    return re.search(GANJI_RX, s).group(0) if re.search(GANJI_RX, s) else None

def sexagenary_of_gregorian_year(y: int, prefer_hanzi=True) -> str:
    # 연간지 변환기(간단 스텁). 실제 로직/테이블과 연결되어 있다면 그걸 호출하세요.
    # 여기선 안전하게 None 반환 방지용으로 둡니다.
    try:
        stems = "甲乙丙丁戊己庚辛壬癸"
        branches = "子丑寅卯辰巳午未申酉戌亥"
        idx = (y - 4) % 60
        return stems[idx % 10] + branches[idx % 12]
    except:
        return None

def extract_target_ganji_v2(updated_question: str
                             ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    updated_question(문자열)에서 타겟 연/월/일/시 간지 추출.
    우선순위: 접미사 명시(년/월/일/시) → yyyy년 숫자→연간지.
    """
    src = updated_question or ""
    year = month = day = hour = None

    m = YEAR_RX.search(src)
    if m: year = normalize_ganji(m.group(0))
    m = MONTH_RX.search(src)
    if m: month = normalize_ganji(m.group(0))
    m = DAY_RX.search(src)
    if m: day = normalize_ganji(m.group(0))
    m = HOUR_RX.search(src)
    if m: hour = normalize_ganji(m.group(0))

    if year is None:
        ymatch = re.search(r"(\d{4})\s*(?:년|年)", src)
        if ymatch:
            y = int(ymatch.group(1))
            yg = sexagenary_of_gregorian_year(y, prefer_hanzi=True)
            if yg: year = yg

    return year, month, day, hour

