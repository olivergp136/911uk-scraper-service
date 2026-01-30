import os
import json
import time
import threading
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL = os.environ.get("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "60"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "25"))
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "500"))
WRITE_BATCH_SIZE = int(os.environ.get("WRITE_BATCH_SIZE", "500"))
SLEEP_BETWEEN_MEMBERS = float(os.environ.get("SLEEP_BETWEEN_MEMBERS", "0.15"))

# Optional: enable repair pass (second model call) when contradictions are detected
ENABLE_REPAIR_PASS = os.environ.get("ENABLE_REPAIR_PASS", "true").strip().lower() in ("1", "true", "yes", "y")
REPAIR_MODEL = os.environ.get("REPAIR_MODEL", MODEL)  # can set to cheaper model if you want

app = FastAPI(title="911uk Signature Parser Service")

RUN_LOCK = threading.Lock()
ACTIVE_RUN: Optional[Dict[str, Any]] = None
STOP_FLAG = False

# ----------------------------
# OpenAI client (lazy init)
# ----------------------------

_openai_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

# ----------------------------
# API models
# ----------------------------

class StartRunRequest(BaseModel):
    start_member_id: int = Field(default=1, ge=1)
    max_member_id: int = Field(..., ge=1)

# ----------------------------
# Supabase helpers
# ----------------------------

def sb_headers(prefer: str = "return=representation") -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }

def sb_insert_parse_run(start_member_id: int, max_member_id: int) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/parse_runs"
    payload = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "start_member_id": start_member_id,
        "max_member_id": max_member_id,
        "last_processed_member_id": start_member_id - 1,
        "ok_count": 0,
        "error_count": 0,
        "model": MODEL,
    }
    r = requests.post(url, headers=sb_headers(), data=json.dumps(payload), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase insert parse_runs failed: {r.status_code} {r.text}")
    return r.json()[0]

def sb_update_parse_run(run_id: str, patch: Dict[str, Any]) -> None:
    url = f"{SUPABASE_URL}/rest/v1/parse_runs?id=eq.{run_id}"
    r = requests.patch(
        url,
        headers=sb_headers(prefer="return=minimal"),
        data=json.dumps(patch),
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase update parse_runs failed: {r.status_code} {r.text}")

def sb_fetch_members_page(after_member_id: int, max_member_id: int, limit: int) -> List[Dict[str, Any]]:
    """
    Fetch a page of members ordered by member_id asc, starting at after_member_id.
    Only rows with signature_raw not null.
    """
    url = f"{SUPABASE_URL}/rest/v1/members"
    params = {
        "select": "member_id,run_id,signature_raw",
        "member_id": f"gte.{after_member_id}",
        "signature_raw": "not.is.null",
        "order": "member_id.asc",
        "limit": str(limit),
    }
    params["member_id"] = f"gte.{after_member_id}"
    params["and"] = f"(member_id.lte.{max_member_id})"

    r = requests.get(url, headers=sb_headers(prefer="return=representation"), params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase fetch members failed: {r.status_code} {r.text}")
    return r.json()

def sb_upsert_member_cars(rows: List[Dict[str, Any]]) -> None:
    """
    Upsert into member_cars using your unique index ux_member_cars_dedupe.

    IMPORTANT:
    We explicitly set the conflict target via on_conflict to avoid 409 duplicate key crashes.
    """
    if not rows:
        return

    url = (
        f"{SUPABASE_URL}/rest/v1/member_cars"
        f"?on_conflict=member_id,ownership,make,model,variant,source_text"
    )
    headers = sb_headers(prefer="resolution=merge-duplicates,return=minimal")
    r = requests.post(url, headers=headers, data=json.dumps(rows), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert member_cars failed: {r.status_code} {r.text}")

# ----------------------------
# Prompt (LLM extraction)
# ----------------------------

SYSTEM_PROMPT = r"""
You are an expert interpreter of enthusiast forum signatures with deep knowledge of car models, Porsche generations/variants, and enthusiast shorthand.

Your task:
Parse the raw forum signature text and extract 0..N owned cars into structured records.

--------------------------------
OUTPUT RULES (STRICT)
--------------------------------
- Output MUST be valid JSON.
- Output MUST contain a top-level object with key: "cars".
- Each car MUST include "source_text" which is a VERBATIM substring supporting that car.
- Do NOT output explanatory text or commentary.
- Never invent facts: if missing, use null and reduce confidence.

--------------------------------
CAR DETECTION & SEGMENTATION
--------------------------------
Signatures may list multiple cars using newlines, pipes (|), bullets, semicolons, commas, or headings.

Rules:
- Split into logical car chunks.
- Avoid over-splitting when details continue across lines.
- Ignore non-vehicle lines (URLs, slogans, quotes, ads).

--------------------------------
PORSCHE DETECTION (COMMON)
--------------------------------
A chunk is a CAR if it contains any of:
- Porsche platform tokens (964, 993, 996, 997, 991, 992, 986, 987, 981, 718, 982, 957, 958, 930)
- Porsche models (911, Boxster, Cayman, Cayenne, Panamera, Macan, 944, 968, 356)
- Porsche variants (C2, C4, C2S, C4S, GT3, GT2, Turbo, Turbo S, GTS, RS, Spyder, Targa, Dakar, Sport Classic, S/T)

--------------------------------
NON-PORSCHE CARS (MUST HANDLE)
--------------------------------
Many signatures include non-Porsche cars too. You MUST extract those as separate car records.

A chunk is also a CAR if it contains clear non-Porsche vehicle cues such as:
- A known non-Porsche make name (BMW, Mercedes, Audi, VW, Ford, Ferrari, Aston Martin, Jaguar, Land Rover, Mini, Lotus, Nissan, Toyota, Honda, Mazda, Subaru, Renault, Peugeot, Alfa Romeo, Volvo, Tesla, etc.)
- A model+platform pattern (e.g., "E46 M3", "F80 M3", "R53 Cooper S", "C6 RS6", "W204 C63")
- Common trim/performance badges (AMG, M3, M5, RS, GTI, ST, Type R, WRX, Vantage, DB9, F430, etc.)

IMPORTANT (implicit makes):
Sometimes the brand is not written. If the model name strongly implies a make, infer it.
Examples:
- "V8 Vantage" => Aston Martin
- "DB9" / "DB11" / "Vanquish" => Aston Martin
- "F430" / "458" / "488" / "360" => Ferrari
- "Gallardo" / "Huracan" => Lamborghini
- "Range Rover" / "Defender" / "Discovery" => Land Rover
- "M3" / "M5" with E/F/G codes => BMW
Only infer make if the model/trim is strongly distinctive; otherwise leave make null and lower confidence.

--------------------------------
OWNERSHIP (IMPORTANT)
--------------------------------
Default: "Current"
Only "Sold" if explicit tokens in that chunk: ex, ex-, sold, gone, previous, formerly, prior, used to own, was my, had, (sold)
"Unknown" only if the signature is essentially non-car content.

--------------------------------
FIELDS
--------------------------------
- make: explicit brand if present; otherwise infer only when very distinctive (see implicit makes above).
- model:
  - Porsche: MUST be Family + (Platform) e.g. "911 (993)" / "Boxster (987.2)" / "Cayenne (957)". NEVER put year in model.
  - Non-Porsche: use the primary model line, optionally with platform in parentheses if explicit (e.g., "M3 (E46)", "C63 (W204)").
- variant: trim/edition/derivative only (e.g. "Carrera 4S", "GT3 RS", "Turbo S", "Competition", "AMG", "Type R").
- notes: MUST include any model years, colours, engine sizes, gearbox, option codes, modifications, "since 2003", etc.
  (If present anywhere in the chunk, put it in notes.)

--------------------------------
CHEAT SHEET: 911 GENERATIONS (YEAR -> PLATFORM)
--------------------------------
When a year is present AND the chunk clearly refers to a 911-family car (911/Carrera/Turbo/GT3/GT2/RS/Targa),
use the correct platform:
- 1964–1973 => Early 911
- 1974–1989 => G-Series (Impact bumper) (Turbo often called 930)
- 1989–1994 => 964
- 1994–1998 => 993
- 1998–2001 => 996.1
- 2002–2004 => 996.2
- 2004–2008 => 997.1
- 2009–2012 => 997.2
- 2011–2015 => 991.1
- 2016–2019 => 991.2
- 2019–2024 => 992.1
- 2024+      => 992.2

--------------------------------
EVIDENCE (REQUIRED)
--------------------------------
For each car, include an "evidence" object that captures the key tokens you used (strings or null):
- year_token: e.g. "1988", "97", "'02", "MY97" (but NOT "996"/"993" platform codes)
- platform_token: e.g. "993", "996.2", "987.2", "957" etc if present or strongly implied
- ownership_token: e.g. "ex", "sold", "gone" if present; otherwise null
- notes_tokens: short list of tokens you considered notes-worthy (colours, gearbox, engine size, codes, mods)

IMPORTANT:
- NEVER treat Porsche platform codes like 996/993/997 as a model year.

--------------------------------
FINAL OUTPUT
--------------------------------
Return:
{
  "cars": [
    {
      "ownership": "Current"|"Sold"|"Unknown",
      "make": string|null,
      "model": string|null,
      "variant": string|null,
      "notes": string|null,
      "source_text": string,
      "confidence": number (0..1),
      "evidence": {
        "year_token": string|null,
        "platform_token": string|null,
        "ownership_token": string|null,
        "notes_tokens": [string]
      }
    }
  ]
}
"""

def call_ai(signature_raw: str) -> Dict[str, Any]:
    user_prompt = f"Signature:\n---\n{signature_raw}\n---\nExtract the cars from this signature."

    resp = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
    )

    out = resp.choices[0].message.content or ""
    if not out.strip():
        return {"cars": []}
    return json.loads(out)

def repair_ai(signature_raw: str, bad_car_obj: Dict[str, Any], issues: List[str]) -> Optional[Dict[str, Any]]:
    """
    Second-pass repair for a single car object when we detect contradictions.
    """
    if not ENABLE_REPAIR_PASS:
        return None

    prompt = {
        "signature_raw": signature_raw,
        "car": bad_car_obj,
        "issues": issues,
        "instructions": [
            "Fix the car object so it satisfies the rules.",
            "Do NOT invent facts; only correct contradictions.",
            "Do NOT put years in model; put them in notes.",
            "Default ownership to Current unless explicit sold/ex evidence exists.",
            "Never treat Porsche platform codes like 996/993/997/987 as years.",
            "Return ONLY a JSON object with keys: ownership, make, model, variant, notes, source_text, confidence, evidence.",
        ],
    }

    resp = get_client().chat.completions.create(
        model=REPAIR_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict validator/repair tool for structured car extraction JSON."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    out = resp.choices[0].message.content or ""
    if not out.strip():
        return None
    fixed = json.loads(out)
    if not isinstance(fixed, dict):
        return None
    return fixed

# ----------------------------
# Helpers
# ----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None

def ensure_notes_contains(notes: Optional[str], addition: Optional[str]) -> Optional[str]:
    add = (addition or "").strip()
    if not add:
        return notes
    n = (notes or "").strip()
    if not n:
        return add
    if add.lower() in n.lower():
        return n
    return f"{add}; {n}"

def ensure_notes_contains_many(notes: Optional[str], tokens: List[str]) -> Optional[str]:
    out = notes
    for t in tokens:
        out = ensure_notes_contains(out, t)
    return out

# ----------------------------
# Constraint layer (minimal, but prevents “impossible” outputs)
# ----------------------------

PLATFORM_TOKENS = {
    "930", "964", "993",
    "996", "996.1", "996.2",
    "997", "997.1", "997.2",
    "991", "991.1", "991.2",
    "992", "992.1", "992.2",
    "986", "987", "987.1", "987.2",
    "981",
    "718", "982",
    "957", "958",
}

SOLD_WORDS_RE = re.compile(r"\b(ex|ex-|sold|gone|previous|formerly|prior|used to own|was my|had|\(sold\))\b", re.I)
MODEL_YEAR_IN_PARENS_RE = re.compile(r"\(\s*(19\d{2}|20\d{2})\s*\)")

def normalize_year_token_to_int(year_token: Optional[str]) -> Optional[int]:
    """
    Minimal normalizer for model-provided evidence.year_token.
    Reject platform-like tokens (e.g. 996).
    """
    if not year_token:
        return None
    t = year_token.strip().replace("’", "'").upper()

    # hard reject Porsche platform codes as years
    if t in {p.upper() for p in PLATFORM_TOKENS}:
        return None

    # 4-digit year
    m4 = re.fullmatch(r"(19\d{2}|20\d{2})", t)
    if m4:
        y = int(m4.group(1))
        return y if 1960 <= y <= 2035 else None

    # 2-digit year tokens: '97, 97, MY97, 97MY
    m2 = re.fullmatch(r"(?:MY)?'?(?P<yy>\d{2})(?:MY)?", t)
    if m2:
        yy = int(m2.group("yy"))
        if 70 <= yy <= 99:
            return 1900 + yy
        if 0 <= yy <= 29:
            return 2000 + yy
    return None

def is_911_family_text(source_text: str, model: Optional[str], variant: Optional[str]) -> bool:
    st = (source_text or "").lower()
    m = (model or "").lower()
    v = (variant or "").lower()
    cues = ["911", "carrera", "turbo", "gt3", "gt2", "rs", "targa", "dakar", "sport classic", "s/t"]
    return any(c in st for c in cues) or m.startswith("911") or any(c in v for c in cues)

def platform_from_year_for_911(year: int) -> Optional[str]:
    if 1964 <= year <= 1973:
        return "911 (Early 911)"
    if 1974 <= year <= 1989:
        if 1978 <= year <= 1983:
            return "911 (SC)"
        if 1984 <= year <= 1989:
            return "911 (3.2 Carrera)"
        return "911 (G-Series)"
    if 1989 <= year <= 1994:
        if year == 1994:
            return "911 (964 / 993)"
        return "911 (964)"
    if 1994 <= year <= 1998:
        return "911 (993)"
    if 1998 <= year <= 2001:
        return "911 (996.1)"
    if 2002 <= year <= 2004:
        return "911 (996.2)"
    if 2004 <= year <= 2008:
        return "911 (997.1)"
    if 2009 <= year <= 2012:
        return "911 (997.2)"
    if 2011 <= year <= 2015:
        return "911 (991.1)"
    if 2016 <= year <= 2019:
        return "911 (991.2)"
    if 2019 <= year <= 2024:
        return "911 (992.1)"
    if year >= 2024:
        return "911 (992.2)"
    return None

def move_year_out_of_model(model: Optional[str], notes: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not model:
        return model, notes
    m = model.strip()
    m2 = MODEL_YEAR_IN_PARENS_RE.search(m)
    if not m2:
        return model, notes
    yr = m2.group(1)
    m = MODEL_YEAR_IN_PARENS_RE.sub("", m).strip()
    m = re.sub(r"\s+", " ", m).strip()
    m = m.replace("()", "").strip()
    notes = ensure_notes_contains(notes, yr)
    return (m or None), (notes or None)

def normalize_ownership(ownership: Optional[str], evidence_ownership_token: Optional[str], source_text: str) -> str:
    """
    Business rule: Current unless explicit sold/ex evidence exists.
    """
    tok = (evidence_ownership_token or "").strip().lower()
    if tok and SOLD_WORDS_RE.search(tok):
        return "Sold"
    if SOLD_WORDS_RE.search(source_text or ""):
        return "Sold"
    return "Current"

def apply_gt3_dot2_hint(model: Optional[str], source_text: str, platform_token: Optional[str]) -> Optional[str]:
    st = (source_text or "").lower()
    if "gt3.2" not in st and "gt3 .2" not in st:
        return model
    pt = (platform_token or "").strip().lower()
    if pt.startswith("996"):
        return "911 (996.2)"
    if pt.startswith("997"):
        return "911 (997.2)"
    if pt.startswith("991"):
        return "911 (991.2)"
    if pt.startswith("992"):
        return "911 (992.2)"
    # fallback: look in source text
    if "996" in st:
        return "911 (996.2)"
    if "997" in st:
        return "911 (997.2)"
    if "991" in st:
        return "911 (991.2)"
    if "992" in st:
        return "911 (992.2)"
    return model

def validate_and_fix_car(car: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], bool]:
    """
    Returns: (fixed_car, issues, needs_repair_pass)
    Minimal “rail” layer: enforce invariants + fix contradictions.
    """
    issues: List[str] = []
    needs_repair = False

    ownership = safe_str(car.get("ownership"))
    make = safe_str(car.get("make"))
    model = safe_str(car.get("model"))
    variant = safe_str(car.get("variant"))
    notes = safe_str(car.get("notes"))
    source_text = safe_str(car.get("source_text")) or ""
    confidence = clamp01(car.get("confidence") or 0.0)

    evidence = car.get("evidence") if isinstance(car.get("evidence"), dict) else {}
    year_token = safe_str(evidence.get("year_token")) if isinstance(evidence, dict) else None
    platform_token = safe_str(evidence.get("platform_token")) if isinstance(evidence, dict) else None
    ownership_token = safe_str(evidence.get("ownership_token")) if isinstance(evidence, dict) else None
    notes_tokens = evidence.get("notes_tokens") if isinstance(evidence, dict) else []
    if not isinstance(notes_tokens, list):
        notes_tokens = []

    # Must have evidence substring
    if not source_text.strip():
        return {}, ["missing_source_text"], False

    # 1) Ownership: force default Current unless sold evidence exists
    fixed_ownership = normalize_ownership(ownership, ownership_token, source_text)
    if ownership and ownership != fixed_ownership:
        issues.append(f"ownership_corrected:{ownership}->{fixed_ownership}")
    ownership = fixed_ownership

    # 2) Never year in model
    model, notes = move_year_out_of_model(model, notes)

    # 3) Put notes tokens into notes
    clean_notes_tokens = [str(t).strip() for t in notes_tokens if str(t).strip()]
    if clean_notes_tokens:
        notes = ensure_notes_contains_many(notes, clean_notes_tokens)

    # 4) Year token normalization (trust model evidence, but block platform tokens)
    year_int = normalize_year_token_to_int(year_token)
    if year_int:
        notes = ensure_notes_contains(notes, str(year_int))

    # 5) Guard: year_token must not equal platform_token
    if year_token and platform_token and year_token.strip().lower() == platform_token.strip().lower():
        issues.append("year_token_equals_platform_token")
        needs_repair = True
        year_int = None

    # 6) GT3.2 hint
    model = apply_gt3_dot2_hint(model, source_text, platform_token)

    # 7) 911 year->platform sanity correction (ONLY for clear 911-family)
    if year_int and is_911_family_text(source_text, model, variant):
        expected = platform_from_year_for_911(year_int)
        if expected:
            if not model:
                model = expected
                issues.append(f"model_set_from_year:{expected}")
            else:
                if model.lower().startswith("911") and expected not in model:
                    issues.append(f"model_corrected_from_year:{model}->{expected}")
                    model = expected

    # 8) If the chunk explicitly contains a Porsche platform token, do NOT let year inference override it.
    st_lower = source_text.lower()
    explicit_platform = None
    for tok in sorted(PLATFORM_TOKENS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(tok.lower())}\b", st_lower):
            explicit_platform = tok
            break
    if explicit_platform and (model or "").lower().startswith("911"):
        if explicit_platform not in (model or ""):
            # This indicates the model probably output a mismatched platform
            issues.append(f"explicit_platform_present:{explicit_platform} model={model}")
            needs_repair = True

    # 9) Clamp confidence down if we corrected things
    if issues:
        confidence = clamp01(min(confidence, 0.75))

    fixed = {
        "ownership": ownership,
        "make": make,
        "model": model,
        "variant": variant,
        "notes": notes,
        "source_text": source_text,
        "confidence": confidence,
        "evidence": {
            "year_token": year_token,
            "platform_token": platform_token,
            "ownership_token": ownership_token,
            "notes_tokens": clean_notes_tokens,
        },
    }
    return fixed, issues, needs_repair

# ----------------------------
# Main run loop
# ----------------------------

def run_parse(run_id: str, start_member_id: int, max_member_id: int):
    global STOP_FLAG, ACTIVE_RUN

    ok = 0
    errors = 0
    last_processed = start_member_id - 1

    def progress_log():
        print(f"[parse {run_id}] last={last_processed} ok={ok} errors={errors}", flush=True)
        sb_update_parse_run(
            run_id,
            {
                "last_processed_member_id": last_processed,
                "ok_count": ok,
                "error_count": errors,
                "status": "running",
            },
        )

    try:
        print(f"[parse {run_id}] START {start_member_id}->{max_member_id} model={MODEL} repair={ENABLE_REPAIR_PASS}", flush=True)

        cursor = start_member_id
        pending_rows: List[Dict[str, Any]] = []

        while cursor <= max_member_id:
            with RUN_LOCK:
                if STOP_FLAG:
                    sb_update_parse_run(
                        run_id,
                        {
                            "status": "stopped",
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "last_processed_member_id": last_processed,
                            "ok_count": ok,
                            "error_count": errors,
                        },
                    )
                    ACTIVE_RUN = {"active": False, "status": "stopped", "run_id": run_id}
                    print(f"[parse {run_id}] STOPPED at member_id={last_processed}", flush=True)
                    return

            members = sb_fetch_members_page(cursor, max_member_id, PAGE_SIZE)
            if not members:
                break

            for m in members:
                with RUN_LOCK:
                    if STOP_FLAG:
                        break

                member_id = int(m["member_id"])
                sig = (m.get("signature_raw") or "")
                src_run_id = m.get("run_id")

                last_processed = member_id

                if not sig.strip():
                    continue

                last_err = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        result = call_ai(sig)
                        cars = result.get("cars", []) if isinstance(result, dict) else []

                        for car in cars:
                            if not isinstance(car, dict):
                                continue

                            if not safe_str(car.get("source_text")):
                                continue

                            fixed_car, issues, needs_repair = validate_and_fix_car(car)

                            if needs_repair:
                                repaired = repair_ai(sig, fixed_car, issues)
                                if repaired and isinstance(repaired, dict):
                                    fixed2, issues2, _ = validate_and_fix_car(repaired)
                                    fixed_car = fixed2 if fixed2 else fixed_car

                            if not fixed_car:
                                continue

                            pending_rows.append(
                                {
                                    "member_id": member_id,
                                    "run_id": src_run_id,
                                    "ownership": safe_str(fixed_car.get("ownership")) or "Current",
                                    "make": safe_str(fixed_car.get("make")),
                                    "model": safe_str(fixed_car.get("model")),
                                    "variant": safe_str(fixed_car.get("variant")),
                                    "notes": safe_str(fixed_car.get("notes")),
                                    "source_text": safe_str(fixed_car.get("source_text")) or "",
                                    "confidence": clamp01(fixed_car.get("confidence") or 0.0),
                                    "parsed_at": datetime.now(timezone.utc).isoformat(),
                                }
                            )

                        ok += 1
                        last_err = None
                        break

                    except Exception as e:
                        last_err = f"{type(e).__name__}: {str(e)[:400]}"
                        if attempt == MAX_RETRIES:
                            break
                        time.sleep(1.0 * attempt)

                if last_err:
                    errors += 1
                    msg = f"member_id={member_id}: {last_err}"
                    print(f"[parse {run_id}] ERROR {msg}", flush=True)
                    sb_update_parse_run(
                        run_id,
                        {"last_error": msg, "error_count": errors, "last_processed_member_id": last_processed},
                    )

                if member_id % PROGRESS_EVERY == 0:
                    if pending_rows:
                        for i in range(0, len(pending_rows), WRITE_BATCH_SIZE):
                            sb_upsert_member_cars(pending_rows[i : i + WRITE_BATCH_SIZE])
                        pending_rows = []
                    progress_log()

                time.sleep(SLEEP_BETWEEN_MEMBERS)

            cursor = int(members[-1]["member_id"]) + 1

            if pending_rows:
                for i in range(0, len(pending_rows), WRITE_BATCH_SIZE):
                    sb_upsert_member_cars(pending_rows[i : i + WRITE_BATCH_SIZE])
                pending_rows = []

        if pending_rows:
            for i in range(0, len(pending_rows), WRITE_BATCH_SIZE):
                sb_upsert_member_cars(pending_rows[i : i + WRITE_BATCH_SIZE])

        sb_update_parse_run(
            run_id,
            {
                "status": "done",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "last_processed_member_id": last_processed,
                "ok_count": ok,
                "error_count": errors,
            },
        )
        print(f"[parse {run_id}] DONE last={last_processed} ok={ok} errors={errors}", flush=True)
        with RUN_LOCK:
            ACTIVE_RUN = {"active": False, "status": "done", "run_id": run_id}

    except Exception as e:
        errors += 1
        msg = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"[parse {run_id}] FAILED {msg}", flush=True)
        try:
            sb_update_parse_run(
                run_id,
                {
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "last_processed_member_id": last_processed,
                    "ok_count": ok,
                    "error_count": errors,
                    "last_error": msg,
                },
            )
        finally:
            with RUN_LOCK:
                ACTIVE_RUN = {"active": False, "status": "failed", "run_id": run_id, "last_error": msg}

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def home():
    return {
        "service": "911uk signature parser",
        "endpoints": {"POST /start": {}, "POST /stop": {}, "GET /active": {}},
        "config": {
            "model": MODEL,
            "repair_enabled": ENABLE_REPAIR_PASS,
            "repair_model": REPAIR_MODEL,
            "temperature": TEMPERATURE,
            "progress_every": PROGRESS_EVERY,
            "page_size": PAGE_SIZE,
        },
    }

@app.get("/active")
def active():
    with RUN_LOCK:
        return ACTIVE_RUN or {"active": False}

@app.post("/start")
def start(req: StartRunRequest):
    global ACTIVE_RUN, STOP_FLAG

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500, detail="Supabase env vars not set")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    if req.start_member_id > req.max_member_id:
        raise HTTPException(status_code=400, detail="start_member_id must be <= max_member_id")

    with RUN_LOCK:
        if ACTIVE_RUN and ACTIVE_RUN.get("status") == "running":
            raise HTTPException(status_code=400, detail="A run is already in progress")
        STOP_FLAG = False

    run = sb_insert_parse_run(req.start_member_id, req.max_member_id)
    run_id = run["id"]

    with RUN_LOCK:
        ACTIVE_RUN = {
            "active": True,
            "status": "running",
            "run_id": run_id,
            "start_member_id": req.start_member_id,
            "max_member_id": req.max_member_id,
        }

    t = threading.Thread(target=run_parse, args=(run_id, req.start_member_id, req.max_member_id), daemon=True)
    t.start()

    return {"ok": True, "run_id": run_id, "start_member_id": req.start_member_id, "max_member_id": req.max_member_id}

@app.post("/stop")
def stop():
    global STOP_FLAG
    with RUN_LOCK:
        if not ACTIVE_RUN or ACTIVE_RUN.get("status") == "running":
            STOP_FLAG = True
            ACTIVE_RUN["status"] = "stopping"
            return {"ok": True, "message": "Stopping soon (after current member finishes)."}
        raise HTTPException(status_code=400, detail="No active run")
