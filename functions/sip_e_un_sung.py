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


# === 0) 간지에서 지지(마지막 글자)만 뽑아오는 작은 유틸 ===
def _branch_of(ganji: str | None) -> str | None:
    """
    '戊戌' → '戌', '경오' → '오', None → None
    한자/한글 모두 허용. 주어진 문자열의 '마지막 글자'를 지지로 간주.
    """
    if not ganji or not isinstance(ganji, str) or len(ganji) == 0:
        return None
    return ganji[-1]


