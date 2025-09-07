# 오행 매핑
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
