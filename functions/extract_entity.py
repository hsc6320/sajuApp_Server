from datetime import datetime
import re
from typing import List, Dict, Optional, Tuple

from datetime import datetime
import re


FACT_KEYS = ["종목명","인물","타겟_연도","타겟_월","타겟_일","타겟_시","간지","키워드"]

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

def _normalize_events(evts: list[dict]) -> list[dict]:
    out = []
    for e in evts or []:
        kind = _normalize_event_kind(e.get("종류", ""))
        date = _normalize_date(e.get("날짜", ""))
        desc = (e.get("설명") or "").strip()
        if kind:
            out.append({"종류": kind, "날짜": date, "설명": desc})
    return out

# ──────────────────────────────
# 3) FACTS 병합
def _merge_entities(base: dict, new: dict) -> dict:
    print("[MERGE] start")
    merged = {
        k: list(base.get(k, []))
        for k in ["종목명","인물","타겟_연도","타겟_월","타겟_일","타겟_시","간지","키워드"]
    }
    for k, vals in (new or {}).items():
        if k in merged:
            before = len(merged[k])
            merged[k].extend(vals or [])
            merged[k] = _dedup_list(merged[k])[:8]
            print(f"[MERGE] {k}: before={before}, add={len(vals or [])}, after={len(merged[k])}")

    base_events = base.get("이벤트", []) or []
    new_events = _normalize_events((new or {}).get("이벤트", []))
    print(f"[MERGE] events base={len(base_events)} new={len(new_events)}")
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
def _parse_facts_from_summary(summary: str) -> dict:
    print("[PARSE] start")
    base = {k: [] for k in ["종목명","인물","타겟_연도","타겟_월","타겟_일","타겟_시","간지","키워드"]}
    base["이벤트"] = []
    if not summary or FACTS_HEADER not in summary:
        print("[PARSE] header not found → return empty")
        return base
    block = summary.split(FACTS_HEADER, 1)[1]
    lines = [l.rstrip() for l in block.splitlines()]
    simple_keys = {
        "인물": "인물", 
        "종목명": "종목명",
        "타겟_연도": "타겟_연도", 
        "타겟_월": "타겟_월",
        "타겟_일": "타겟_일", 
        "타겟_시": "타겟_시",
        "간지": "간지", 
        "키워드": "키워드"
    }
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        for label, key in simple_keys.items():
            if ln.startswith(f"- {label}:"):
                vals = ln.split(":", 1)[1].strip()
                items = [v.strip() for v in vals.split(",") if v.strip() and v.strip() != "없음"]
                base[key] = items
                print(f"[PARSE] {label} ← {items}")
        if ln.startswith("- 이벤트:"):
            i += 1
            while i < len(lines) and lines[i].startswith("  "):
                if lines[i].strip().startswith("- 종류:"):
                    kind = lines[i].split(":",1)[1].strip()
                    evt = {"종류": _normalize_event_kind(kind)}
                    j = i + 1
                    while j < len(lines) and lines[j].startswith("    "):
                        sub = lines[j].strip()
                        if sub.startswith("날짜:"):
                            evt["날짜"] = _normalize_date(sub.split(":",1)[1].strip())
                        elif sub.startswith("설명:"):
                            evt["설명"] = sub.split(":",1)[1].strip()
                        j += 1
                    base["이벤트"].append(evt)
                    print(f"[PARSE] event +1 {evt}")
                    i = j - 1
        i += 1
    print("[PARSE] end", {k: len(v) for k,v in base.items()})
    return base

# ──────────────────────────────
# 6) summary에 FACTS 병합
def enrich_summary_with_entities(prev_summary: str, new_entities: dict, keep_tail_chars: int = 1200) -> str:
    print("[FACTS] enrich start")
    print("\n[FACTS] --- enrich_summary_with_entities 시작 ---")
    print(f"[FACTS] 이전 summary 길이: {len(prev_summary) if prev_summary else 0}")
    print(f"[FACTS] 신규 엔티티 입력: {json.dumps(new_entities, ensure_ascii=False)}")
    prev_summary = prev_summary or ""
    if FACTS_HEADER in prev_summary:
        prev_body = prev_summary.split(FACTS_HEADER, 1)[0].rstrip()
        print("[FACTS] header found → split body")
    else:
        prev_body = prev_summary.strip()
        print("[FACTS] no header → body only")
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
    result = "\n".join([p for p in parts if p is not None]).strip()
    print("[FACTS] enrich end")
    return result

# ===== [D] extract_entities_for_summary 인자화, 안전화 =====
def extract_entities_for_summary(user_text: str, assistant_text: str, payload: dict | None = None) -> dict:
    print("\n[ENTITIES] --- extract_entities_for_summary 시작 ---")
    print(f"[ENTITIES] 사용자 입력: {user_text}")
    print(f"[ENTITIES] AI 응답: {assistant_text}")

    ents = {k: [] for k in FACT_KEYS + ["이벤트"]}

    # ✅ payload.target_time에서 간지 보강 (없으면 skip)
    if payload:
        tt = (payload.get("target_time") or {})
        y = (tt.get("year")  or {}).get("ganji")
        m = (tt.get("month") or {}).get("ganji")
        d = (tt.get("day")   or {}).get("ganji")
        h = (tt.get("hour")  or {}).get("ganji")

        if y: ents["타겟_연도"].append(y); ents["간지"].append(y)
        if m: ents["타겟_월"].append(m);   ents["간지"].append(m)
        if d: ents["타겟_일"].append(d);   ents["간지"].append(d)
        if h: ents["타겟_시"].append(h);   ents["간지"].append(h)

    # TODO: 필요하다면 여기서 user_text/assistant_text에서
    # 종목명/인물/키워드 간단 정규식 추출을 추가할 수 있음.

    # 중복 제거 & 상한
    for k in FACT_KEYS:
        ents[k] = _dedup_list(ents.get(k, []))[:8]


    print("[ENTITIES] --- extract_entities_for_summary 끝 ---\n")
    return ents


# ===== [C] wanted 이벤트 종류 간단 추출기 =====
# 질문에서 특정 이벤트 의도를 추출 (룰 기반 키워드 매칭)
def _wanted_event_kind(text: str) -> str | None:
    if not text:
        return None
    t = text.strip().lower()

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

def quick_lookup_from_facts(question: str, summary_text: str) -> str | None:
    kind = _wanted_event_kind(question)             # 어떤 이벤트를 찾는지
    if not kind:
        return None
    facts = _parse_facts_from_summary(summary_text) # 요약에서 FACTS dict 추출
    for e in facts.get("이벤트", []):                # 이벤트 목록 순회
        if _normalize_event_kind(e.get("종류")) == _normalize_event_kind(kind):
            date = e.get("날짜") or "날짜 정보 없음"
            desc = e.get("설명") or ""
            return (
                "[MODE: LOOKUP]\n"
                f"- {kind} 날짜: {date}\n"
                + (f"- 메모: {desc}\n" if desc else "")
            )
    return None


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
