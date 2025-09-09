from datetime import datetime
import re
from typing import List, Dict, Optional, Tuple
import json

FACT_KEYS = ["종목명","인물","타겟_연도","타겟_월","타겟_일","타겟_시","간지","키워드"]
EVENT_SYNONYMS: dict[str, list[str]] = {
    "면접": ["면접", "인터뷰"],
    "결혼": ["결혼식", "웨딩", "결혼"],
    "여행": ["해외여행", "국내여행", "여행운", "출장", "여행", "휴가", "트립", "trip"],
    "시험": ["시험", "수능", "자격증", "고시"],
    "생일": ["생일", "생신", "birthday", "돌잔치", "돌"],
    "기념일": ["기념일", "anniversary"],
}

# ──────────────────────────────
# 1) 유틸: 중복 제거
def _dedup_list(items):
    seen, out = set(), []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x.lower() in seen:
            continue
        seen.add(x.lower())
        out.append(x)
    return out

# ──────────────────────────────
# 2) 이벤트 정규화
def _normalize_event_kind(kind: str) -> str:
    k = (kind or "").strip().lower()
    if not k:
        return k
    mapping = {
        "면접": "면접", "인터뷰": "면접",
        "결혼": "결혼", "결혼식": "결혼", "웨딩": "결혼",
        "여행": "여행", "출장": "여행",
        "시험": "시험", "수능": "시험", "자격증": "시험",
        "생일": "생일", "돌": "생일", "생신": "생일", "birthday": "생일",
        "기념일": "기념일", "anniversary": "기념일"
    }
    for key in sorted(mapping.keys(), key=len, reverse=True):
        if key in k:
            return mapping[key]
    return kind.strip()

def _normalize_date(s: str) -> str:
    if not s:
        return s
    raw = s.strip()
    try:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
            try:
                return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
            except:
                pass
        m = re.match(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", raw)
        if m:
            y, mo, d = map(int, m.groups())
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return raw
    except:
        return raw
def _ensure_dict(obj, default=None):
    """obj가 dict가 아니어도 최대한 dict로 변환 (str JSON 지원). 실패 시 default 반환."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {} if default is None else default

def _normalize_events(evts) -> list[dict]:
    """evts가 list/dict/str/JSON 등 어떤 형식이 와도
    [{'종류':..., '날짜':..., '설명':...}] 리스트로 정규화."""
    out: list[dict] = []

    if not evts:
        return out

    # JSON 문자열이면 파싱
    if isinstance(evts, str):
        try:
            maybe = json.loads(evts)
            return _normalize_events(maybe)
        except Exception:
            # 그냥 '여행' 같은 단일 문자열로 간주
            evts = [evts]

    # 단일 dict면 리스트화
    if isinstance(evts, dict):
        evts = [evts]

    if not isinstance(evts, list):
        return out

    for e in evts:
        if isinstance(e, str):
            e = {"종류": e}
        if not isinstance(e, dict):
            continue

        kind = _normalize_event_kind(e.get("종류", ""))
        date = _normalize_date(e.get("날짜", ""))
        desc = (e.get("설명") or "").strip()

        if not kind:
            continue

        item = {"종류": kind}
        if date:
            item["날짜"] = date
        if desc:
            item["설명"] = desc
        out.append(item)

    return out


# ──────────────────────────────
# 3) FACTS 병합
def _merge_entities(base: dict, new: dict) -> dict:
    print("\n[MERGE] start")

    # ⚠️ 방어: 문자열/엉뚱한 타입이 들어와도 dict로 강제
    base = _ensure_dict(base, {})
    new  = _ensure_dict(new,  {})

    merged = {
        k: list(base.get(k, []))
        for k in ["종목명","인물","타겟_연도","타겟_월","타겟_일","타겟_시","간지","키워드"]
    }

    for k, vals in (new or {}).items():
        if k in merged:
            merged[k].extend(vals or [])
            merged[k] = _dedup_list(merged[k])[:8]

    # 이벤트도 어떤 형식이 와도 정규화해서 병합
    base_events = _normalize_events(base.get("이벤트", []))
    new_events  = _normalize_events(new.get("이벤트", []))

    try:
        print(f"base_events : {base_events} , new_events : {new_events}")
        print(f"[MERGE] events base={len(base_events)} new={len(new_events)}")
    except Exception:
        pass

    seen = set()
    merged_events = []
    for e in base_events + new_events:
        key = (e.get("종류",""), e.get("날짜",""), e.get("설명",""))
        if key in seen:
            continue
        seen.add(key)
        merged_events.append(e)

    merged["이벤트"] = merged_events[:20]
    print(f"[MERGE] events merged={len(merged['이벤트'])}")
    return merged


# ──────────────────────────────
# 4) FACTS 포맷팅
FACTS_HEADER = "📌 FACTS(엔티티):"

def _format_facts_block(facts: dict) -> str:
    print(f"[FACTS] formatting keys: {list(facts.keys())}")
    facts = facts or {}
    lines = [
        FACTS_HEADER,
        f"- 인물: {', '.join(facts.get('인물', [])) or '없음'}",
        f"- 종목명: {', '.join(facts.get('종목명', [])) or '없음'}",
        f"- 타겟_연도: {', '.join(facts.get('타겟_연도', [])) or '없음'}",
        f"- 타겟_월: {', '.join(facts.get('타겟_월', [])) or '없음'}",
        f"- 타겟_일: {', '.join(facts.get('타겟_일', [])) or '없음'}",
        f"- 타겟_시: {', '.join(facts.get('타겟_시', [])) or '없음'}",
        f"- 간지: {', '.join(facts.get('간지', [])) or '없음'}",
        f"- 키워드: {', '.join(facts.get('키워드', [])) or '없음'}",
        f"- 이벤트:"
    ]
    events = facts.get("이벤트", []) or []
    if not events:
        lines.append("  (없음)")
    else:
        for e in events:
            lines.append(f"  - 종류: {e.get('종류','')}")
            if e.get("날짜"):  lines.append(f"    날짜: {e['날짜']}")
            if e.get("설명"):  lines.append(f"    설명: {e['설명']}")
    return "\n".join(lines)

# ──────────────────────────────
# 5) FACTS 파싱
def _dedup_events(evts: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for e in evts or []:
        key = (e.get("종류",""), e.get("날짜",""), e.get("설명",""))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out

def _parse_facts_from_summary(summary: str) -> dict:
    print("[PARSE] start")
    print(f"[PARSE] summary : {summary}")
    base = {k: [] for k in ["종목명","인물","타겟_연도","타겟_월","타겟_일","타겟_시","간지","키워드"]}
    base["이벤트"] = []
    if not summary or FACTS_HEADER not in summary:
        print("[PARSE] header not found → return empty")
        return base

    # ✅ 요약에 FACTS 블록이 여러 번 들어있어도 "가장 마지막" 것만 파싱
    block = summary.rsplit(FACTS_HEADER, 1)[1]
    lines = [l.rstrip() for l in block.splitlines()]

    simple_keys = {
        "인물": "인물",
        "종목명": "종목명",
        "타겟_연도": "타겟_연도",
        "타겟_월": "타겟_월",
        "타겟_일": "타겟_일",
        "타겟_시": "타겟_시",
        "간지": "간지",
        "키워드": "키워드",
    }

    i = 0
    print(f"len {lines}")
    while i < len(lines):
        ln = lines[i].strip()

        # 🚫 방어: 혹시 또 다른 FACTS 헤더가 나오면 그 뒤는 무시
        if ln.startswith(FACTS_HEADER):
            break

        for label, key in simple_keys.items():
            if ln.startswith(f"- {label}:"):
                vals = ln.split(":", 1)[1].strip()
                items = [v.strip() for v in vals.split(",") if v.strip() and v.strip() != "없음"]
                base[key] = items
                print(f"[PARSE] {label} ← {items}")

        if ln.startswith("- 이벤트:"):
            i += 1
            # 들여쓰기 2칸(스페이스) 기반 하위 라인 파싱 유지
            while i < len(lines) and lines[i].startswith("  "):
                if lines[i].strip().startswith("- 종류:"):
                    kind = lines[i].split(":", 1)[1].strip()
                    evt = {"종류": _normalize_event_kind(kind)}
                    j = i + 1
                    while j < len(lines) and lines[j].startswith("    "):
                        sub = lines[j].strip()
                        if sub.startswith("날짜:"):
                            evt["날짜"] = _normalize_date(sub.split(":", 1)[1].strip())
                        elif sub.startswith("설명:"):
                            evt["설명"] = sub.split(":", 1)[1].strip()
                        j += 1
                    base["이벤트"].append(evt)
                  #  print(f"[PARSE] event +1 {evt}")
                    i = j - 1
        i += 1

    # ✅ 파싱 결과 정규화 + 중복 제거(이전 요약에 중복 FACTS가 있었어도 여기서 정리)
    base["이벤트"] = _dedup_events(_normalize_events(base.get("이벤트", [])))

    print("[PARSE] end", {k: len(v) for k,v in base.items()})
    return base

# ──────────────────────────────
# 6) summary에 FACTS 병합
def enrich_summary_with_entities(prev_summary: str, new_entities: dict, keep_tail_chars: int = 1200) -> str:
    print("[FACTS] enrich start")
    print("[FACTS] --- enrich_summary_with_entities 시작 ---")
    print(f"[FACTS] 이전 summary 길이: {len(prev_summary) if prev_summary else 0}")
    print(f"[FACTS] 신규 엔티티 입력: {json.dumps(new_entities, ensure_ascii=False)}")
    prev_summary = prev_summary or ""
    if FACTS_HEADER in prev_summary:
        prev_body = prev_summary.split(FACTS_HEADER, 1)[0].rstrip()
        print("[FACTS] header found → split body")
        print(f"prev_body : {prev_body}")
        print(f"prev_summary : {prev_summary}")
    else:
        prev_body = prev_summary.strip()
        print("[FACTS] no header → body only")
        print(f"prev_body : {prev_body}")        
    if len(prev_body) > keep_tail_chars:
        prev_body = prev_body[-keep_tail_chars:]
        print(f"[FACTS] body trimmed to {keep_tail_chars} chars")
        
    old_facts = _parse_facts_from_summary(prev_summary)
    merged = _merge_entities(old_facts, new_entities)
       # 🟡 디버깅 로그
    print("===== FACTS Debug Log =====")
    print(f"[FACTS] 신규 엔티티 입력: {new_entities}")
    print(f"[FACTS] 기존 FACTS 파싱 결과: {old_facts}")
    print(f"[FACTS] 병합된 FACTS 결과: {merged}")
    print("============================")

    parts = [prev_body.strip(), "", _format_facts_block(merged)]
    print(f"parts : {parts}")
    result = "\n".join([p for p in parts if p is not None]).strip()
    print(f"result : {result}")
    print("[FACTS] enrich end")
    return result

# 2) 이벤트 추출기 (모드와 무관하게 항상 실행)
def _extract_events_from_text(text: str,  payload: dict | None) -> list[str]:
    t = (text or "").strip().lower()
    if not t:
        return []
    found = set()
    # 한글 보호: 별도 토큰화 없이 부분 포함도 허용
    for canon, words in EVENT_SYNONYMS.items():
        if any(w in t for w in words):
            found.add(canon)
    return list(found)

def _build_event_desc_from_payload(payload: dict | None) -> str:
    if not payload:
        return ""
    tt = (payload.get("target_time") or {})
    y = (tt.get("year")  or {}).get("ganji")
    m = (tt.get("month") or {}).get("ganji")
    d = (tt.get("day")   or {}).get("ganji")
    h = (tt.get("hour")  or {}).get("ganji")
    parts = [p for p in [y,m,d,h] if p]
    return f"타겟 간지: {' '.join(parts)}" if parts else ""

def _scan_event_kinds(text: str) -> list[str]:
    """사용자 문장 안에서 이벤트 의도(여러 개 가능)를 추출해 표준 이벤트명으로 반환."""
    if not text:
        return []
    t = text.strip().lower()
    found = []
    for canon, words in EVENT_SYNONYMS.items():
        for w in sorted(words, key=len, reverse=True):
            if w in t:
                found.append(canon)
                break  # 같은 canon 중복 방지
    return _dedup_list(found)

# ===== [D] extract_entities_for_summary 인자화, 안전화 =====
def extract_entities_for_summary(user_text: str, assistant_text: str, payload: dict | None = None) -> dict:
    print("\n[ENTITIES] --- extract_entities_for_summary 시작 ---")
    print(f"[ENTITIES] 사용자 입력: {user_text}")
    print(f"[ENTITIES] AI 응답: {assistant_text}")    
    
    ents = {k: [] for k in FACT_KEYS + ["이벤트"]}
    #print(f"ents_2 {ents}") //출력 : ents_2 {'종목명': [], '인물': [], '타겟_연도': [], '타겟_월': [], '타겟_일': [], '타겟_시': [], '간지': [], '키워드': [], '이벤트': []}

    # ✅ payload.target_time에서 간지 보강 (없으면 skip)
    if payload:
        tt = (payload.get("target_time") or {})        
        y = (tt.get("year")  or {}).get("ganji")
        m = (tt.get("month") or {}).get("ganji")
        d = (tt.get("day")   or {}).get("ganji")
        h = (tt.get("hour")  or {}).get("ganji")
        #print(f"tt : {tt} y:{y}, m:{m}, d:{d}") 
        # 출력 :tt : {'year': {'ganji': '乙巳', 'sipseong': None, 'sibi_unseong': None}, 'month': {'ganji': '甲申', 'sipseong': None, 'sibi_unseong': None}, 'day': {'ganji': '壬午', 'sipseong': None, 'sibi_unseong': None}, 'hour': {'ganji': None, 'sipseong': None, 'sibi_unseong': None}} 
                                                #y:乙巳, m:甲申, d:壬午        
        if y: ents["타겟_연도"].append(y); ents["간지"].append(y)
        if m: ents["타겟_월"].append(m);   ents["간지"].append(m)
        if d: ents["타겟_일"].append(d);   ents["간지"].append(d)
        if h: ents["타겟_시"].append(h);   ents["간지"].append(h)
    

    # 중복 제거 & 상한
    for k in FACT_KEYS:
        ents[k] = _dedup_list(ents.get(k, []))[:8]

    # 이벤트 추출 → dict로 저장 + 간지 설명 부여
    kinds = _scan_event_kinds(user_text)  # ex) ["여행"]
    if kinds:
        desc = _build_event_desc_from_payload(payload)
        ents["이벤트"] = [{"종류": k, **({"설명": desc} if desc else {})} for k in kinds]
        
    # [FIX] 이벤트는 분류/모드와 무관하게 항상 추출
    extracted = _extract_events_from_text(user_text, payload)
    if extracted:
        ents["이벤트"] = extracted


    #print(f"[ENTITIES] --- ents: {ents} ---\n") 출력 : [ENTITIES] --- ents: {'종목명': [], '인물': [], '타겟_연도': ['乙巳'], '타겟_월': ['甲申'], '타겟_일': ['壬午'], '타겟_시': [], '간지': ['乙巳', '甲申', '壬午'], '키워드': [], '이벤트': []} ---
    print("[ENTITIES] --- extract_entities_for_summary 끝 ---\n")
    return ents


# ===== [C] wanted 이벤트 종류 간단 추출기 =====
# 질문에서 특정 이벤트 의도를 추출 (룰 기반 키워드 매칭)
def _wanted_event_kind(text: str) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    print(f"_wanted_event_kind {t}")
    # 우선순위가 겹칠 수 있으므로, 긴 키워드 먼저 체크
    rules = [
        # (키워드들, 정규화된 이벤트명)
        (["면접", "인터뷰"], "면접"),
        (["결혼식", "웨딩", "결혼"], "결혼"),
        (["출장", "여행"], "여행"),
        (["시험", "수능", "자격증", "고시"], "시험"),
        (["생일", "생신", "birthday", "돌잔치", "돌"], "생일"),
        (["기념일", "anniversary"], "기념일"),
    ]
    for keywords, kind in rules:
        for kw in keywords:
            if kw in t:
                return kind
    return None

def _fallback_desc_from_facts(facts: dict) -> str:
    def last(key):
        v = facts.get(key, [])
        return v[-1] if v else None
    y, m, d, h = last("타겟_연도"), last("타겟_월"), last("타겟_일"), last("타겟_시")
    parts = [p for p in [y,m,d,h] if p]
    return f"타겟 간지(추정): {' '.join(parts)}" if parts else ""

def quick_lookup_from_facts(question: str, summary_text: str) -> str | None:
    kind = _wanted_event_kind(question)
    if not kind:
        return None
    print(f"kind : {kind}")
    facts = _parse_facts_from_summary(summary_text)

    # 가장 최근 해당 이벤트
    target = None
    for e in reversed(facts.get("이벤트", [])):
        if _normalize_event_kind(e.get("종류","")) == _normalize_event_kind(kind):
            target = e
            break
    if not target:
        return None

    date = target.get("날짜")
    desc = target.get("설명")
    if not date and not desc:
        # 🔹 과거 이벤트가 비어있으면 FACTS의 최신 간지로 설명 폴백
        desc = _fallback_desc_from_facts(facts)

    lines = ["[MODE: LOOKUP]"]
    lines.append(f"- {kind} 날짜: {date if date else '날짜 정보 없음'}")
    if desc:
        lines.append(f"- 메모: {desc}")
    return "\n".join(lines)


# # ──────────────────────────────
# # 8) record_turn
# global_summary = ""

# def record_turn(user_text: str, assistant_text: str):
#     global global_summary
#     print("\n================= record_turn start =================")
#     print(f"[TURN] user: {user_text}")
#     print(f"[TURN] assistant: {assistant_text[:100]}{'...' if len(assistant_text)>100 else ''}")
#     prev_summary = global_summary
#     ents = extract_entities_for_summary(user_text, assistant_text)
#     print(f"[TURN] ents={ents}")
#     new_summary = enrich_summary_with_entities(prev_summary, ents, keep_tail_chars=1200)
#     global_summary = new_summary
#     print("[TURN] roundtrip summary head:")
#     print(global_summary[:300])
#     print("================== record_turn end ==================\n")
