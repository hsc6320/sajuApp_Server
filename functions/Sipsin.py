# 오행 매핑
import re
from converting_time import normalize_ganji


five_element_map = {
    '甲': '木', '乙': '木',
    '丙': '火', '丁': '火',
    '戊': '土', '己': '土',
    '庚': '金', '辛': '金',
    '壬': '水', '癸': '水',
}

# 십이지 → 오행
ji_to_element = {
    '子': '水',
    '丑': '土',
    '寅': '木',
    '卯': '木',
    '辰': '土',
    '巳': '火',
    '午': '火',
    '未': '土',
    '申': '金',
    '酉': '金',
    '戌': '土',
    '亥': '水',
}

# 오행 상생, 상극 관계
element_produces = {
    '木': '火',
    '火': '土',
    '土': '金',
    '金': '水',
    '水': '木',
}

element_overcomes = {
    '木': '土',
    '火': '金',
    '土': '水',
    '金': '木',
    '水': '火',
}


def get_sipshin(il_gan: str, target_gan: str) -> str:
    def is_yang(gan):
        return gan in ['甲', '丙', '戊', '庚', '壬']

    il_gan = il_gan.strip()
    target_gan = target_gan.strip()

    il_element = five_element_map.get(il_gan)
    target_element = five_element_map.get(target_gan)

    if il_element is None or target_element is None:
        return "미정"

    same_yang = is_yang(il_gan)
    same_yang2 = is_yang(target_gan)

    if il_element == target_element:
        return '비견' if same_yang == same_yang2 else '겁재'

    if element_produces.get(il_element) == target_element:
        return '식신' if same_yang == same_yang2 else '상관'

    if element_produces.get(target_element) == il_element:
        return '편인' if same_yang == same_yang2 else '정인'

    if element_overcomes.get(il_element) == target_element:
        return '편재' if same_yang == same_yang2 else '정재'

    if element_overcomes.get(target_element) == il_element:
        return '편관' if same_yang == same_yang2 else '정관'

    return '미정'


# 지지 → 지장간 (藏干)
ji_to_hidden_stems = {
    '子': ['壬', '癸'],
    '丑': ['癸', '辛', '己'],
    '寅': ['戊', '丙', '甲'],
    '卯': ['甲', '乙'],
    '辰': ['乙', '癸', '戊'],
    '巳': ['戊', '庚', '丙'],
    '午': ['丙', '己', '丁'],
    '未': ['丁', '乙', '己'],
    '申': ['戊', '壬', '庚'],
    '酉': ['庚'],
    '戌': ['辛', '丁', '戊'],
    '亥': ['戊', '甲', '壬'],
}

# 십신 계산 함수는 기존 get_sipshin(일간, 타간)을 사용

def get_ji_sipshin_only(ilgan: str, ji: str) -> str:
    il_gan = ilgan.strip().strip('"')
    target_ji = ji.strip().strip('"')

    hidden_stems = ji_to_hidden_stems.get(target_ji, [])
    
    if hidden_stems:
        last_stem = hidden_stems[-1]
        return get_sipshin(il_gan, last_stem)
    
    return '없음'


# ── 표준 테이블
BRANCHES_HJ = ["子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥"]
BRANCHES_KO = ["자","축","인","묘","진","사","오","미","신","유","술","해"]

STEMS_HJ = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸"]
STEMS_KO = ["갑","을","병","정","무","기","경","신","임","계"]

KO2HJ_BRANCH = {ko: hj for ko, hj in zip(BRANCHES_KO, BRANCHES_HJ)}
HJ2KO_BRANCH = {hj: ko for ko, hj in zip(BRANCHES_HJ, BRANCHES_KO)}

KO2HJ_STEM   = {ko: hj for ko, hj in zip(STEMS_KO, STEMS_HJ)}
HJ2KO_STEM   = {hj: ko for ko, hj in zip(STEMS_HJ, STEMS_KO)}

def _norm_branch(x: str | None) -> str | None:
    """
    지지 입력을 한자(子..亥)로 표준화.
    허용: '戌', '술', '庚戌' 같은 혼합 문자열(마지막 글자 사용)
    """
    if not x or not isinstance(x, str):
        return None
    c = x.strip()[-1]            # 마지막 글자(지지 후보)
    if c in BRANCHES_HJ:         # 이미 한자 지지
        return c
    if c in KO2HJ_BRANCH:        # 한글 지지 → 한자
        return KO2HJ_BRANCH[c]
    return None

def _norm_stem(x: str | None) -> str | None:
    """
    천간 입력을 한자(甲..癸)로 표준화.
    허용: '庚', '경', '경오'(앞 글자 사용) 등
    """
    if not x or not isinstance(x, str):
        return None
    c = x.strip()[0]             # 첫 글자(천간 후보)
    if c in STEMS_HJ:
        return c
    if c in KO2HJ_STEM:
        return KO2HJ_STEM[c]
    return None


def split_ganji_parts(s: str | None) -> tuple[str | None, str | None]:
    """
    기존 normalize_ganji(s)로 간지 토큰을 먼저 추출(정규식 매치)한 뒤,
    천간/지지를 각각 한자 표준으로 반환.
    예: '2018년 무술' → ('戊','戌'), '경오' → ('庚','午')
    """
    if not s: 
        return None, None
    #token = normalize_ganji(s)  # ← 네가 이미 가진 함수 (없으면 None)
    token = norm_ganji_to_hanzi(s)   # ← 네가 이미 가진 함수 (없으면 None)
    if not token:
        return None, None
    return _norm_stem(token), _norm_branch(token)

def stem_from_any(s: str | None) -> str | None:
    st, _ = split_ganji_parts(s); return st

def branch_from_any(s: str | None) -> str | None:
    _, br = split_ganji_parts(s); return br


# ── 간지 패턴: '甲申', '乙巳' 등
GANJI_RX = r"[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]"
YEAR_RX  = re.compile(rf"{GANJI_RX}\s*년")
MONTH_RX = re.compile(rf"{GANJI_RX}\s*월")
DAY_RX   = re.compile(rf"{GANJI_RX}\s*일")
HOUR_RX  = re.compile(rf"{GANJI_RX}\s*시")
def norm_ganji_to_hanzi(s: str | None) -> str | None:
    """
    입력 문자열에서 간지 토큰(한자/한글/혼합)을 표준 한자 2글자(甲..癸)(子..亥)로 추출.
    예) '계해' → '癸亥', '경오년' → '庚午', '2018년 무술' → '戊戌'
    """
    if not s:
        return None

    t = str(s).strip()
    # 흔한 구분자/공백 제거
    t = t.replace(" ", "").replace("-", "").replace("/", "")

    # 1) 한글 2글자 간지 (예: 계해, 경오)
    if len(t) >= 2 and t[0] in KO2HJ_STEM and t[-1] in KO2HJ_BRANCH:
        return KO2HJ_STEM[t[0]] + KO2HJ_BRANCH[t[-1]]

    # 2) 혼합형: (한자 천간 + 한글 지지) 또는 (한글 천간 + 한자 지지)
    if len(t) >= 2:
        a, b = t[0], t[-1]
        if a in STEMS_HJ and b in KO2HJ_BRANCH:
            return a + KO2HJ_BRANCH[b]
        if a in KO2HJ_STEM and b in BRANCHES_HJ:
            return KO2HJ_STEM[a] + b

    # 3) 한자 간지 패턴 내장 검색 (문장 속에 섞여 있을 때)
    m = re.search(GANJI_RX, t)
    if m:
        return m.group(0)

    return None


    