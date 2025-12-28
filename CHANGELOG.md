## Unreleased

# 대화 내용 삭제 기능 추가

## 📋 개요
Flutter 앱에서 대화 내용 삭제 요청 시 서버에서 해당 사용자의 JSON 파일을 삭제하는 기능을 추가했습니다.

---

## 1. delete_history 플래그 처리

### 1.1 기능 설명
- Flutter에서 `delete_history: "true"` 플래그를 전송하면 서버에서 해당 사용자의 대화 내용 JSON 파일을 삭제합니다.
- `reset` 플래그와 유사하지만 별도로 처리되며, 응답 필드가 다릅니다.

### 1.2 구현 내용 (main.py)
- `delete_history` 플래그 파싱 로직 추가
  - 문자열/불리언/숫자 형태 모두 지원 (`"true"`, `"1"`, `"yes"`, `"y"` 등)
  - `reset` 플래그와 동일한 유연한 파싱 방식 적용
- 사용자 컨텍스트 재설정
  - `name`과 `birth`를 사용하여 `make_user_key()`로 정확한 user_id 생성
  - `app_uid`를 포함하여 올바른 파일 경로 지정
- 파일 삭제 처리
  - `delete_current_user_store()` 함수 호출
  - GCS 또는 로컬 파일 삭제
  - 삭제 결과 로그 출력
- 응답 반환
  - `{"delete_history": bool, "user_id": str, "path": str}` 형식으로 응답

### 1.3 파일 경로 구조
- Cloud: `gs://<GCS_BUCKET>/users/<app_uid>/profiles/<name>__<birth>.json`
- 예시: `gs://chatsaju-5cd67-convos/users/hsc6320/profiles/홍승창__19880716.json`

### 1.4 수정된 파일
- `functions/main.py`
  - `make_user_key` import 추가
  - `delete_history` 플래그 처리 로직 추가 (약 30줄)
  - `reset` 플래그 처리 바로 다음에 배치

### 1.5 요청 예시
```json
{
  "delete_history": "true",
  "name": "홍승창",
  "birth": "1988-07-16",
  "app_uid": "hsc6320",
  "session_id": "single_global_session"
}
```

### 1.6 응답 예시
```json
{
  "delete_history": true,
  "user_id": "홍승창__19880716",
  "path": "gs://chatsaju-5cd67-convos/users/hsc6320/profiles/홍승창__19880716.json"
}
```

---

# 개인맞춤입력 정보 처리 및 조후(調候) 기능 추가

## 📋 개요
사주 해석의 현실 적합도를 높이기 위해 개인맞춤입력 정보 처리 기능을 추가하고, 조후(調候) 계산 기능을 구현하여 해석의 구조적 보정 레이어를 제공합니다.

---

## 1. 개인맞춤입력 정보 수신 및 처리

### 1.1 수신 필드 정의
- **필수 필드**: `jobStatus`, `jobName`, `maritalStatus`, `concerns`
- **권장 필드**: `lifeStage`, `moneyActivity`, `relationshipStatus`
- **보조 필드**: `hobbies`, `traits`
- **민감 필드**: `hasHealthConcern`
- **기타 필드**: `note`

### 1.2 데이터 수신 로직 (main.py)
- 루트 레벨과 중첩 객체(`personalInfo`, `personal_info`) 모두 지원
- 디버깅 로그로 수신 여부 확인
- 서버 로그에 개인맞춤입력 정보 섹션 출력
- 각 필드별 상세 정보 및 전체 객체 JSON 출력

### 1.3 Context 생성 및 전달
- `personal_info_context` 생성
- `enhanced_context`에 포함하여 LLM에 전달
- 디버깅 로그로 포함 여부 확인
- LLM 응답 반영 확인 (키워드 매칭: 정확 매칭, 공백 제거 후 매칭, 부분 매칭)

### 1.4 Payload 구성 (core/services.py)
- `make_saju_payload` 함수에서 `payload.meta.personal_info`에 개인맞춤입력 정보 포함
- 사주 구조 계산과 완전 분리하여 저장

### 1.5 프롬프트 개선 (prompts/saju_prompts.py)
- **개인맞춤입력 정보 사용 규칙 추가**
  - 절대 금지: 사주 구조(간지·십성·운) 계산에 사용 금지
  - 사용 목적: 해석·조언의 현실 적합도 보정용(context)으로만 사용
- **필수 사용 규칙 강화**
  - 직업명(`jobName`)이 있으면 반드시 해당 직업을 언급
  - 직업 상태(`jobStatus`)에 맞는 해석
  - 혼인 상태(`maritalStatus`) 고려
  - 현재 고민 영역(`concerns`)에 집중
  - 재물 활동(`moneyActivity`)에 맞는 조언
  - 연애 상태(`relationshipStatus`) 고려
  - 취미 성향(`hobbies`) 관련 맥락 고려
- **출력 형식 규칙 추가**
  - `SAJU_EXPLAIN` 형식에서도 개인맞춤입력 정보를 자연스럽게 언급하도록 지시

---

## 2. 조후(調候) 기능 추가

### 2.1 새 파일 생성
- **functions/joohu.py** (330줄)
  - 조후 계산 모듈 생성
  - 월령 기반 한열조습(寒熱燥濕) 판단 로직
  - `calculate_joohu()`: 핵심 계산 함수
  - `get_joohu_flags()`: 편의 함수
  - `is_balanced` 플래그 추가 (조후 균형 상태)

### 2.2 Payload 구성 (core/services.py)
- 조후 계산 로직 추가:
  - 원국 조후: `joohu_natal`
  - 대운 조후: `joohu_daewoon` (원국 + 대운 합산)
  - 세운 조후: `joohu_seun` (원국 + 세운 합산)
  - 월운 조후: `joohu_wolun` (원국 + 월운 합산)
- Payload에 조후 필드 추가:
  - `payload["natal"]["joohu"]`
  - `payload["current_daewoon"]["joohu"]`
  - `payload["target_time"]["year"]["joohu"]`
  - `payload["target_time"]["month"]["joohu"]`
  - `payload["resolved"]["flow_now"]["daewoon"]["joohu"]`
  - `payload["resolved"]["flow_now"]["target"]["year"]["joohu"]`
  - `payload["resolved"]["flow_now"]["target"]["month"]["joohu"]`
- 디버깅 로그 추가:
  - 조후 계산 결과 로그
  - Payload 포함 여부 확인 로그

### 2.3 프롬프트 개선 (prompts/saju_prompts.py)
- **조후 필드 정의 추가** (간결화됨):
  - `natal.joohu` 필드 설명
  - `current_daewoon.joohu` 필드 설명
  - `target_time.{year|month}.joohu` 필드 설명
- **조후 반영 규칙 추가** (간결화됨):
  - 필수 포함 요구사항
  - 데이터 경로 우선순위
  - 표현 예시 간결화
- **형식별 출력 규칙에 조후 포함 명시**:
  - `SAJU_EXPLAIN`: 조후 1회 이상 포함
  - `SAJU_STRUCTURED`: 조후 1회 이상 포함

### 2.4 프롬프트 간결화
- 조후 관련 프롬프트 간결화
  - 기존: 약 40줄의 상세 설명
  - 수정: 약 16줄로 간결화 (약 60% 감소)
  - 핵심 내용 유지:
    - 조후는 보조 해석 레이어
    - "조후" 용어 사용 금지
    - 표현 예시 간결화

---

## 3. 주요 설계 원칙

### 3.1 개인맞춤입력 정보
- 사주 구조 계산에는 사용하지 않음
- 해석·조언의 현실 적합도 보정용으로만 사용
- 키워드를 자연스럽게 문장에 포함

### 3.2 조후 기능
- 십성/십이운성의 보조 해석 레이어
- 결론 생성이 아닌 해석 보정용
- "왜 체감이 다른지"에 대한 구조적 설명
- 십성/십이운성의 결론을 뒤집지 않음

---

## 4. 데이터 흐름

```
프론트엔드 
  → main.py (개인맞춤입력 정보 수신)
  → core/services.py (payload 구성: personal_info + joohu)
  → main.py (context 생성)
  → prompts/saju_prompts.py (LLM 프롬프트)
  → LLM 응답
  → main.py (반영 확인)
```

---

## 5. 수정된 파일 목록

### 5.1 신규 파일
- `functions/joohu.py` (330줄)

### 5.2 수정된 파일
- `functions/main.py`
  - 개인맞춤입력 정보 수신 로직
  - 서버 로그 출력
  - Context 생성 및 전달
  - LLM 응답 반영 확인
- `functions/core/services.py`
  - `make_saju_payload`에 `personal_info` 포함
  - 조후 계산 로직 추가 (52곳 수정)
  - Payload에 조후 필드 추가
- `functions/prompts/saju_prompts.py`
  - 개인맞춤입력 정보 사용 규칙 추가
  - 필수 사용 규칙 강화
  - 출력 형식 규칙 추가
  - 조후 규칙 추가 및 간결화

---

## 6. 검증 결과

### 6.1 개인맞춤입력 정보
- ✅ 개인맞춤입력 정보 정상 수신
- ✅ 프롬프트에 정상 포함 (191자)
- ✅ LLM 응답에 일부 반영 ("직장인", "재물/투자")
- ✅ 맥락적 활용 확인 ("직장인으로서 월급 기반의 안정성")

### 6.2 조후 기능
- ✅ 조후 계산 정상 작동 (로그 확인)
- ✅ Payload에 조후 데이터 포함 확인
- ✅ LLM 응답에 조후 관련 내용 반영 확인
- ✅ 프롬프트 간결화 완료

---

## 7. 향후 개선 사항

- 직업명 등 일부 키워드의 직접 언급 강화
- 프롬프트 지시사항 강화 완료
- 다음 요청부터 더 적극적으로 활용 예상
