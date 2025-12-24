# sip_e_un_sung_lite.py
# -*- coding: utf-8 -*-
"""
십이운성 계산 전용 초경량 유틸
- unseong_for(stem, branch):  천간/지지 → 십이운성
- branch_for(stem, unseong):  천간/운성 → 지지
- pillars_unseong(day_stem, pillars): 일간 기준 (년/월/일/시 지지) 일괄 계산
- seun_unseong(day_stem, year_branch): 일간 기준 '세운 지지'의 운성
한글/한자 모두 입력 가능: ('임'=='壬', '사'=='巳' 등)
"""

# 10천간/12지지/운성
STEMS_KO = ['갑','을','병','정','무','기','경','신','임','계']
STEMS_HJ = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
BRANCHES_KO = ['자','축','인','묘','진','사','오','미','신','유','술','해']
BRANCHES_HJ = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
UNSEONG = ['장생','목욕','관대','건록','제왕','쇠','병','사','묘','절','태','양']

# 표준화 매핑
STEM_ALIASES = {**{k:k for k in STEMS_KO}, **dict(zip(STEMS_HJ, STEMS_KO))}
BRANCH_ALIASES = {**{k:k for k in BRANCHES_KO}, **dict(zip(BRANCHES_HJ, BRANCHES_KO))}

# 양간/음간 진행 방향
YANG_STEMS = {'갑','병','무','경','임'}  # +1 (순행)
YIN_STEMS  = {'을','정','기','신','계'}  # -1 (역행)

# 각 천간의 '장생 시작 지지' (이 규칙만 알면 전체 운성이 결정됨)
START_BRANCH_FOR_JANGSAENG = {
    '갑':'해', '을':'오', '병':'인', '정':'유', '무':'인',
    '기':'유', '경':'사', '신':'자', '임':'신', '계':'묘',
}

def _idx(lst, item):
    try:
        return lst.index(item)
    except ValueError:
        raise ValueError(f"'{item}' not in {lst}")

def _norm_stem(s):
    if not s:
        return None               # ✅ 널 세이프
    try:
        return STEM_ALIASES[s]
    except KeyError:
        raise ValueError(f"Unknown stem: {s}")

def _norm_branch(b):
    if not b:
        return None               # ✅ 널 세이프
    
    try:
        return BRANCH_ALIASES[b]
    except KeyError:
        raise ValueError(f"Unknown branch: {b}")

def unseong_for(stem, branch):
    """
    입력: stem(천간, '임' 또는 '壬'), branch(지지, '사' 또는 '巳')
    출력: '관대' 같은 십이운성 문자열
    """
    s = _norm_stem(stem)
    b = _norm_branch(branch)
    
    if s is None or b is None:    # ✅ 정규화 실패는 계산 생략
        return None

    dirn = +1 if s in YANG_STEMS else -1
    start = _idx(BRANCHES_KO, START_BRANCH_FOR_JANGSAENG[s])  # 장생
    target = _idx(BRANCHES_KO, b)

    steps = (target - start) % 12 if dirn == +1 else (start - target) % 12
    return UNSEONG[steps]

def branch_for(stem, unseong_name, return_hanja=False):
    """
    입력: stem(천간), unseong_name('제왕' 등), return_hanja=True면 한자 지지 반환
    출력: 지지 ('묘' 또는 '卯')
    """
    s = _norm_stem(stem)
    if unseong_name not in UNSEONG:
        raise ValueError(f"Unknown unseong: {unseong_name}")

    dirn = +1 if s in YANG_STEMS else -1
    start = _idx(BRANCHES_KO, START_BRANCH_FOR_JANGSAENG[s])
    steps = UNSEONG.index(unseong_name)
    idx = (start + steps) % 12 if dirn == +1 else (start - steps) % 12

    ko = BRANCHES_KO[idx]
    return BRANCHES_HJ[idx] if return_hanja else ko

def pillars_unseong(day_stem, pillars):
    """
    일간 기준으로 년/월/일/시 지지들의 운성을 dict로 반환.
    pillars 예시: {'year':'진','month':'사','day':'신','hour':'유'}
                  한자 가능: {'year':'辰', ...}
    """
    s = _norm_stem(day_stem)
    
    # ✅ day_stem이 None이면 모든 운성을 None으로 반환
    if not s:
        return {k: None for k in pillars.keys()}
    
    out = {}
    for k, v in pillars.items():
        if not v:                 # ✅ None/빈값이면 그대로 None 저장
            out[k] = None
            continue
        try:
            out[k] = unseong_for(s, v)
        except Exception as e:
            print(f"[pillars_unseong] skip {k}={v!r}: {e}")
            out[k] = None
    return out

def seun_unseong(day_stem, year_branch):
    """일간 기준 '해(세운) 지지'의 운성만 간단히 구함"""
    return unseong_for(day_stem, year_branch)


# === 십이신살 계산 ===
# 십이신살: 겁살, 재살, 천살, 지살, 연살, 월살, 망신살, 장성살, 반안살, 역마살, 육해살, 화개살
SINSAL = ['겁살','재살','천살','지살','연살','월살','망신살','장성살','반안살','역마살','육해살','화개살']

# 삼합(三合) 그룹 정의
SAMHAP_GROUPS = {
    '인오술': ['인', '오', '술'],  # 寅午戌
    '신자진': ['신', '자', '진'],  # 申子辰
    '사유축': ['사', '유', '축'],  # 巳酉丑
    '해묘미': ['해', '묘', '미'],  # 亥卯未
}

# 각 삼합 그룹별 신살 매핑표 (이미지 표 기준)
# 형식: {삼합그룹: {대상지지: 신살인덱스}}
SINSAL_TABLE = {
    '인오술': {  # 寅午戌 그룹
        '해': 0, '자': 1, '축': 2, '인': 3, '묘': 4, '진': 5, '사': 6, '오': 7, '미': 8, '신': 9, '유': 10, '술': 11
    },
    '신자진': {  # 申子辰 그룹
        '사': 0, '오': 1, '미': 2, '신': 3, '유': 4, '술': 5, '해': 6, '자': 7, '축': 8, '인': 9, '묘': 10, '진': 11
    },
    '사유축': {  # 巳酉丑 그룹
        '인': 0, '묘': 1, '진': 2, '사': 3, '오': 4, '미': 5, '신': 6, '유': 7, '술': 8, '해': 9, '자': 10, '축': 11
    },
    '해묘미': {  # 亥卯未 그룹
        '신': 0, '유': 1, '술': 2, '해': 3, '자': 4, '축': 5, '인': 6, '묘': 7, '진': 8, '사': 9, '오': 10, '미': 11
    }
}

def _find_samhap_group(branch_ko: str) -> str | None:
    """지지가 속한 삼합 그룹 찾기"""
    for group_name, branches in SAMHAP_GROUPS.items():
        if branch_ko in branches:
            return group_name
    return None

def sinsal_for(day_branch, target_branch):
    """
    입력: day_branch(일지, '사' 또는 '巳'), target_branch(대상 지지, '해' 또는 '亥')
    출력: '연살' 같은 십이신살 문자열
    
    계산 방법:
    - 일지가 속한 삼합 그룹 찾기
    - 해당 그룹의 신살 표에서 대상 지지의 신살 찾기
    """
    day_b = _norm_branch(day_branch)
    target_b = _norm_branch(target_branch)
    
    if day_b is None or target_b is None:
        return None
    
    # 일지가 속한 삼합 그룹 찾기
    samhap_group = _find_samhap_group(day_b)
    if not samhap_group:
        return None
    
    # 해당 그룹의 신살 표에서 대상 지지 찾기
    sinsal_table = SINSAL_TABLE.get(samhap_group)
    if not sinsal_table:
        return None
    
    sinsal_idx = sinsal_table.get(target_b)
    if sinsal_idx is None:
        return None
    
    return SINSAL[sinsal_idx]

def pillars_sinsal(day_branch, pillars):
    """
    일지 기준으로 년/월/일/시 지지들의 십이신살을 dict로 반환.
    pillars 예시: {'year':'진','month':'사','day':'신','hour':'유'}
                  한자 가능: {'year':'辰', ...}
    """
    day_b = _norm_branch(day_branch)
    
    # ✅ day_branch가 None이면 모든 신살을 None으로 반환
    if not day_b:
        return {k: None for k in pillars.keys()}
    
    out = {}
    for k, v in pillars.items():
        if not v:                 # ✅ None/빈값이면 그대로 None 저장
            out[k] = None
            continue
        try:
            out[k] = sinsal_for(day_branch, v)
        except Exception as e:
            print(f"[pillars_sinsal] skip {k}={v!r}: {e}")
            out[k] = None
    return out

# === 4대 흉살 계산 ===
# 4대 흉살: 백호살, 괴강살, 양인살, 귀문관살

# 백호살: 특정 간지 조합
BAEKHOSAL_GANJI = ['甲辰', '乙未', '丙戌', '丁丑', '戊辰', '壬戌', '癸丑']
BAEKHOSAL_GANJI_KO = ['갑진', '을미', '병술', '정축', '무진', '임술', '계축']

# 괴강살: 특정 간지 조합
GOEGANGSAL_GANJI = ['庚辰', '壬戌', '戊辰', '庚申', '壬辰', '戊戌']
GOEGANGSAL_GANJI_KO = ['경진', '임술', '무진', '경신', '임진', '무술']

# 양인살: 특정 지지
YANGINSAL_BRANCHES = ['卯', '酉', '午']
YANGINSAL_BRANCHES_KO = ['묘', '유', '오']

# 귀문관살: 지지 조합 쌍
GUIMUNGWANSAL_PAIRS = [
    ('子', '酉'), ('丑', '戌'), ('寅', '亥'), ('卯', '申'), ('辰', '未'), ('巳', '午')
]
GUIMUNGWANSAL_PAIRS_KO = [
    ('자', '유'), ('축', '술'), ('인', '해'), ('묘', '신'), ('진', '미'), ('사', '오')
]

def _normalize_ganji(ganji: str | None) -> str | None:
    """간지를 한자로 정규화"""
    if not ganji:
        return None
    # 한글 간지 → 한자 변환
    from Sipsin import KO2HJ_STEM, KO2HJ_BRANCH
    if len(ganji) >= 2:
        stem_char = ganji[0]
        branch_char = ganji[-1]
        stem_hj = KO2HJ_STEM.get(stem_char, stem_char)
        branch_hj = KO2HJ_BRANCH.get(branch_char, branch_char)
        if stem_hj in STEMS_HJ and branch_hj in BRANCHES_HJ:
            return stem_hj + branch_hj
    # 이미 한자면 그대로 반환
    if len(ganji) >= 2 and ganji[0] in STEMS_HJ and ganji[-1] in BRANCHES_HJ:
        return ganji
    return None

def _normalize_branch(branch: str | None) -> str | None:
    """지지를 한자로 정규화"""
    return _norm_branch(branch)

def check_4dae_hyungsal(year: str | None, month: str | None, day: str | None, hour: str | None) -> dict:
    """
    사주 원국에서 4대 흉살 확인
    
    Args:
        year, month, day, hour: 년주, 월주, 일주, 시주 (간지 문자열)
    
    Returns:
        dict: {
            "baekhosal": ["갑진", ...] 또는 [],
            "goegangsal": ["경진", ...] 또는 [],
            "yanginsal": ["묘", ...] 또는 [],
            "guimungwansal": ["자-유", ...] 또는 []
        }
    """
    result = {
        "baekhosal": [],
        "goegangsal": [],
        "yanginsal": [],
        "guimungwansal": []
    }
    
    # 모든 간지 정규화
    pillars = {
        "year": _normalize_ganji(year),
        "month": _normalize_ganji(month),
        "day": _normalize_ganji(day),
        "hour": _normalize_ganji(hour)
    }
    
    # 1. 백호살 확인 (간지 전체 확인)
    for pillar_name, ganji in pillars.items():
        if not ganji:
            continue
        if ganji in BAEKHOSAL_GANJI:
            result["baekhosal"].append(f"{pillar_name}:{ganji}")
    
    # 2. 괴강살 확인 (간지 전체 확인)
    for pillar_name, ganji in pillars.items():
        if not ganji:
            continue
        if ganji in GOEGANGSAL_GANJI:
            result["goegangsal"].append(f"{pillar_name}:{ganji}")
    
    # 3. 양인살 확인 (지지만 확인)
    all_branches = []
    for pillar_name, ganji in pillars.items():
        if ganji:
            branch = _normalize_branch(ganji[-1])
            if branch:
                all_branches.append((pillar_name, branch))
    
    for pillar_name, branch in all_branches:
        if branch in YANGINSAL_BRANCHES:
            result["yanginsal"].append(f"{pillar_name}:{branch}")
    
    # 4. 귀문관살 확인 (지지 쌍 확인)
    branch_list = [b for _, b in all_branches if b]
    for pair_hj, pair_ko in zip(GUIMUNGWANSAL_PAIRS, GUIMUNGWANSAL_PAIRS_KO):
        b1, b2 = pair_hj
        if b1 in branch_list and b2 in branch_list:
            # 어떤 기둥에 있는지 찾기
            found_pillars = []
            for pillar_name, branch in all_branches:
                if branch == b1 or branch == b2:
                    found_pillars.append(f"{pillar_name}:{branch}")
            if found_pillars:
                result["guimungwansal"].append(f"{pair_ko[0]}-{pair_ko[1]}({', '.join(found_pillars)})")
    
    return result

# === 0) 간지에서 지지(마지막 글자)만 뽑아오는 작은 유틸 ===
def _branch_of(ganji: str | None) -> str | None:
    """
    '戊戌' → '戌', '경오' → '오', None → None
    한자/한글 모두 허용. 주어진 문자열의 '마지막 글자'를 지지로 간주.
    """
    if not ganji or not isinstance(ganji, str) or len(ganji) == 0:
        return None
    return ganji[-1]


