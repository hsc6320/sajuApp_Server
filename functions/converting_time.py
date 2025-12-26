from datetime import datetime
import os
import re
from ganji_converter import _json_year_bounds, get_wolju_from_date, get_year_ganji_from_json, get_ilju, resolve_two_digit_year

# from ganji_converter import get_year_ganji_from_json

# 현재 파일 위치 기준으로 JSON 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CURRENT_DIR, "converted.json")

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

# 예: "25년 12월 직업운 어때?" 처럼 '월만' 있는 질문인가?
def is_month_only_question(q: str) -> bool:
    has_month = "월" in q
    has_day = re.search(r"\d{1,2}\s*일", q)
    return has_month and not has_day

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
        
    month_only = is_month_only_question(question)
    # 간지 계산
    year_ganji = get_year_ganji_from_json(datetime(ty, 5, 1), json_path)  # 년주(간지만 표기)
    wolju      = get_wolju_from_date(target_date, json_path, month_only)      # [FIX] target_date 사용 (1일 고정 X)
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
    wolju = get_wolju_from_date(datetime(new_year, new_month, 15), json_path) # [FIX] 1일->15일 (절기 반영)
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

    wolju = get_wolju_from_date(datetime(target_year, month_num, 15), json_path) # [FIX] 1일->15일 (절기 반영)
    if not wolju:
        return None

    original_token = m.group(0)              # '7월' 또는 '7 월'
    relative_to_ganji_map[original_token] = f"{wolju}월"  # 예: '癸未월'
    # relative_to_ganji_map[original_token] = replaced_value
    print(f"월주 키워드 변환(Fixed15) {wolju}월")

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
            print(f"n년뒤/후/전")
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
        
        elif (m2 := re.search(r"(?<!\d)(\d{2})\s*년\b", item)):
            # 기본 매칭: '24년' 또는 '24 년'
            token_2digit = m2.group(0)
            year_suffix = int(m2.group(1))

            # 현재 세기 계산
            # 기존:
            # century = (current_year // 100) * 100
            # full_year = (
            #     century + year_suffix
            #     if year_suffix >= (current_year % 100)
            #     else century + 100 + year_suffix
            # )
            MIN_YEAR, MAX_YEAR = _json_year_bounds(JSON_PATH)

            # 교체:
            full_year = resolve_two_digit_year(year_suffix, today=datetime.now(), prefer_past_on_tie=True)

            # ✅ 여기서 범위 가드
            assert MIN_YEAR <= full_year <= MAX_YEAR, f"비정상 연도 해석: {full_year} (지원 범위: {MIN_YEAR}~{MAX_YEAR})"


            absolute_expressions.append(str(full_year))

            # 연간 간지 치환
            ganji = get_year_ganji_from_json(datetime(full_year, 5, 1), JSON_PATH)
            # 1) expressions 아이템 안에서의 기본 토큰 치환
            relative_to_ganji_map[token_2digit] = f"{ganji}년"

            # 2) 실제 질문에 '26년에', '26 년에' 같이 조사가 붙어 있는 경우까지 치환
            #    - 예: "26년에 주식하면…" → "丙午년에 주식하면…"
            post_pattern = rf"{year_suffix}\s*년에"
            for tok in re.findall(post_pattern, question):
                replaced = tok.replace(str(year_suffix), ganji)
                relative_to_ganji_map[tok] = replaced

            print(f"두자리 년도 치환 간지 정보 : {ganji}년 (full_year={full_year})")

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
            print(f"=== 월 단위 상대 표현 === (예: 3개월 후 / 2달 전 / 3 달 후)")
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
            wolju = get_wolju_from_date(datetime(new_year, new_month, 15), JSON_PATH)  # [FIX] 15일
            
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
            print(f"명시적 연도(yyyy년) (+ 월)")
            m_year = re.match(r"(?P<y>\d{4})", item)
            if m_year:
                MIN_YEAR, MAX_YEAR = _json_year_bounds(JSON_PATH)
                year = int(m_year.group("y"))

                # ✅ JSON 커버 범위 가드
                if not (MIN_YEAR <= year <= MAX_YEAR):
                    raise ValueError(f"지원하지 않는 연도: {year} (지원 범위: {MIN_YEAR}~{MAX_YEAR})")

                # 절대표현(연도)은 그대로 넣되,
                absolute_expressions.append(str(year))

                # 선택: 'yyyy년' → '간지년' 치환 맵도 등록해주면 후속 처리에 유리
                y_ganji = get_year_ganji_from_json(datetime(year, 5, 1), JSON_PATH)
                # 원문 토큰(공백/‘년’ 유무 등)을 최대한 보존
                # 예: "2024년", "2024 년" 모두 매칭
                for tok in re.findall(rf"{year}\s*년", item):
                    relative_to_ganji_map[tok] = f"{y_ganji}년"

                # 월이 있다면 월주까지
                abs_month = handle_month_in_item(item, year, JSON_PATH, relative_to_ganji_map)
                if abs_month is not None:
                    absolute_expressions.append(abs_month)

        # === 그 외 ===
        else:
            if item != question and " " not in item:
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
    print("sexagenary_of_gregorian_year222")
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

    # 1차: 간지 '○○년' 이 이미 있는 경우는 위에서 처리 완료
    # 2차: 숫자 연도(yyyy년) → 연간지
    if year is None:
        ymatch = re.search(r"(\d{4})\s*(?:년|年)", src)
        if ymatch:
            y = int(ymatch.group(1))
            yg = sexagenary_of_gregorian_year(y, prefer_hanzi=True)
            if yg:
                year = yg

    # 3차: 두 자리 연도(예: '26년')도 간지로 해석
    #     - 상대시간 변환이 제대로 안 돼서 '2026년'으로 확장되지 못한 경우를 보완
    if year is None:
        y2 = re.search(r"(?<!\d)(\d{2})\s*(?:년|年)", src)
        if y2:
            suffix = int(y2.group(1))
            try:
                full_year = resolve_two_digit_year(
                    suffix,
                    today=datetime.now(),
                    prefer_past_on_tie=True,
                )
                yg = sexagenary_of_gregorian_year(full_year, prefer_hanzi=True)
                if yg:
                    year = yg
            except Exception:
                # 어떤 이유로든 해석 실패 시 조용히 통과 (year=None 유지)
                pass

    return year, month, day, hour


# utils/date_parse_ko.py (새 파일로 두거나 main.py 상단에 넣어도 됨)
def _to_int(s: Optional[str]) -> Optional[int]:
    try:
        return int(s) if s is not None else None
    except Exception:
        return None

def parse_korean_date_safe(text: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    '1988년 11월 22일', '1988년11월22일', '11월22일', '11월 22일' 등 다양한 표기를 파싱.
    없으면 None 반환. int(None) 호출을 절대 하지 않도록 보장.
    """
    t = text or ""

    # 1) 년-월-일 완전표기 (공백 유무 허용)
    m = re.search(r'(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일', t)
    if m:
        return (_to_int(m.group(1)), _to_int(m.group(2)), _to_int(m.group(3)))

    # 1-2) 년-월만 있는 경우 (일=1로 간주) -> [NEW] 15일로 변경(절기 고려)
    m = re.search(r'(\d{4})\s*년\s*(\d{1,2})\s*월', t)
    if m:
        return (_to_int(m.group(1)), _to_int(m.group(2)), 15)

    # 2) yyyy.mm.dd / yyyy-mm-dd / yyyy/mm/dd
    m = re.search(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', t)
    if m:
        return (_to_int(m.group(1)), _to_int(m.group(2)), _to_int(m.group(3)))

    # 3) 월-일만 (공백 유무 허용)
    m = re.search(r'(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일(?!\d)', t)
    if m:
        return (None, _to_int(m.group(1)), _to_int(m.group(2)))

    # 4) mm.dd / mm-dd / mm/dd (연도 없음)
    m = re.search(r'(?<!\d)(\d{1,2})[./-](\d{1,2})(?!\d)', t)
    if m:
        return (None, _to_int(m.group(1)), _to_int(m.group(2)))

    return (None, None, None)
