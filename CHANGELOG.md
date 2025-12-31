## Unreleased

# 월주 계산 로직 개선 및 버그 수정

## 📋 개요
월주(月柱) 계산 로직의 버그를 수정하고, 월 단위 질문 처리 방식을 개선하여 정확한 월주 계산이 이루어지도록 했습니다.

---

## 1. 월주 계산 절기 선택 버그 수정

### 1.1 문제점
- 절기 배열에서 소한(1월 5일)이 마지막 인덱스(11)에 위치
- 순회 시 소한이 마지막에 선택되어 잘못된 월 인덱스 반환
- 예: 2026년 3월 15일 → 경칩(인덱스 1)이어야 하는데 소한(인덱스 11)이 선택됨

### 1.2 수정 내용 (ganji_converter.py)
- 절기 순회 로직을 역순으로 변경
- 소한을 제외하고 역순 순회하여 입력 날짜 이하의 가장 큰 절기 선택
- 올바른 절기 매칭 보장

### 1.3 수정 전/후
- **수정 전**: `month_stem_table[2][11]` = 辛丑 (잘못된 결과)
- **수정 후**: `month_stem_table[2][1]` = 辛卯 (올바른 결과)

---

## 2. 월 단위 질문 처리 개선

### 2.1 문제점
- "26년 3월" 같은 월 단위 질문에서 `month_only=False`로 처리
- 일반 모드로 처리되어 잘못된 기준일 선택
- 월 단위 질문의 의도와 맞지 않는 처리

### 2.2 수정 내용
다음 함수들에서 `month_only=True` 파라미터 추가:

#### 2.2.1 handle_month_in_item (converting_time.py)
- "3월", "12월" 같은 단독 월 표현 처리
- `get_wolju_from_date(..., month_only=True)` 추가

#### 2.2.2 handle_korean_month_offset (converting_time.py)
- "한달 후", "세 달 전" 같은 상대적 월 표현 처리
- `get_wolju_from_date(..., month_only=True)` 추가

#### 2.2.3 converting_time.py 368번 줄
- 숫자로 된 개월 오프셋 처리
- `get_wolju_from_date(..., month_only=True)` 추가

#### 2.2.4 _ganji_for (ganjiArray.py)
- scope가 "month"일 때 월주 계산
- `get_wolju_from_date(..., month_only=True)` 추가

### 2.3 동작 방식
- **월 단위 질문** ("26년 3월"): `month_only=True`
  - 같은 연도+월의 양력기준일 레코드 우선 검색
  - 해당 월의 정확한 월주 계산
- **일 단위 질문** ("26년 3월 17일"): `month_only=False`
  - 입력 날짜 이하의 가장 가까운 과거 기준일 검색
  - 일반 모드로 처리

### 2.4 is_month_only_question 함수
```python
def is_month_only_question(q: str) -> bool:
    has_month = "월" in q
    has_day = re.search(r"\d{1,2}\s*일", q)
    return has_month and not has_day
```
- "월"이 있고 "일"이 없으면 `True`
- "일"이 있으면 `False`

---

## 3. 로그 개선

### 3.1 월주 계산 로그
- **제거**: 상세한 계산 과정 로그 (기준일 선택, 년간 추출, 절기 계산 등)
- **유지**: 최종 결과만 출력
  - `[월주 계산] 최종 결과: {wolju}`

### 3.2 간지 변환 로그 추가
- **main.py**: 원본 질문과 변환된 질문 비교
  - `[간지 변환] 원본 질문: '{question}' → 변환된 질문: '{updated_question}'`
- **services.py**: updated_question에 포함된 간지 정보 출력
  - `[간지 변환] updated_question에 포함된 간지 정보: '{updated_question}'`

---

## 4. 수정된 파일 목록

### 4.1 ganji_converter.py
- 절기 선택 로직 수정 (역순 순회, 소한 제외)
- 상세 로그 제거, 최종 결과만 출력

### 4.2 converting_time.py
- `handle_month_in_item`: `month_only=True` 추가
- `handle_korean_month_offset`: `month_only=True` 추가
- 368번 줄: `month_only=True` 추가

### 4.3 ganjiArray.py
- `_ganji_for`: scope가 "month"일 때 `month_only=True` 추가

### 4.4 main.py
- 간지 변환 로그 추가

### 4.5 core/services.py
- `make_saju_payload`: updated_question 간지 정보 로그 추가

---

## 5. 검증 결과

### 5.1 절기 선택 버그 수정
- ✅ 2026년 3월 15일 → 경칩(인덱스 1) 올바르게 선택
- ✅ 월주 계산 결과 정확성 향상

### 5.2 월 단위 질문 처리
- ✅ "26년 3월" → `month_only=True`로 처리
- ✅ "26년 3월 17일" → `month_only=False`로 처리
- ✅ 각 질문 유형에 맞는 정확한 월주 계산

### 5.3 로그 개선
- ✅ 불필요한 상세 로그 제거로 로그 가독성 향상
- ✅ 간지 변환 과정 추적 가능

---

## 6. 주요 개선 사항 요약

1. **절기 선택 버그 수정**: 소한이 배열 마지막에 있어 발생한 버그 수정
2. **월 단위 질문 처리 개선**: `month_only=True` 파라미터를 적절한 위치에 추가
3. **로그 최적화**: 상세 로그 제거, 핵심 정보만 출력
4. **간지 변환 추적**: updated_question의 간지 정보 확인 가능

---
