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
    """
    if not rows:
        return
    url = f"{SUPABASE_URL}/rest/v1/member_cars"
    headers = sb_headers(prefer="resolution=merge-duplicates,return=minimal")
    r = requests.post(url, headers=headers, data=json.dumps(rows), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert member_cars failed: {r.status_code} {r.text}")

# ----------------------------
# Response schema + system prompt
# ----------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "cars": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ownership": {"type": "string", "enum": ["Current", "Sold", "Unknown"]},
                    "make": {"type": ["string", "null"]},
                    "model": {"type": ["string", "null"]},
                    "variant": {"type": ["string", "null"]},
                    "notes": {"type": ["string", "null"]},
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["ownership", "make", "model", "variant", "notes", "source_text", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cars"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = r"""
You are an expert interpreter of enthusiast forum signatures with deep knowledge of Porsche models, generations, trims, variants, and community shorthand.

Your task:
Parse the raw forum signature text and extract 0..N owned cars into structured records.

--------------------------------
OUTPUT RULES (STRICT)
--------------------------------
- Output MUST be valid JSON matching the provided schema.
- Output MUST contain a top-level object with key: "cars".
- Each car MUST include "source_text" which is a VERBATIM substring supporting that car.
- Do NOT output explanatory text or commentary.
- Never hallucinate missing facts.
- If unsure, use null values and lower confidence.

--------------------------------
OWNERSHIP STATUS RULES
--------------------------------
Default:
- Assume ownership = "Current"

Set ownership = "Sold" ONLY if explicit evidence exists in the signature chunk:
- tokens: ex, ex-, sold, gone, previous, formerly, prior, old car, used to own, was my, had, (sold)
- headings: "Previous:", "Ex:", "Sold:", "Gone:"

Set ownership = "Unknown" ONLY if the entire signature is non-car content (links/jokes/quotes only).

--------------------------------
CAR DETECTION & SEGMENTATION
--------------------------------
Signatures may list multiple cars using newlines, pipes (|), bullets, semicolons, commas, or headings.

Rules:
- Split into logical car chunks.
- Avoid over-splitting when details continue across lines.
- Ignore non-vehicle lines (URLs, slogans, quotes, ads).

A chunk is considered a CAR if it contains any of:
- Porsche generation/platform tokens (964, 993, 996, 997, 991, 992, 986, 987, 981, 718, 982, 957, 958, SC, G-series, 930)
- Porsche models (911, Boxster, Cayman, Cayenne, Panamera, Macan, 944, 968, 356)
- Porsche variants/trims (C2, C4, C2S, C4S, GT3, GT2, Turbo, Turbo S, GTS, RS, Spyder, Targa, Dakar, Sport Classic, S/T)

--------------------------------
MAKE DETECTION
--------------------------------
- If Porsche shorthand appears (993, C4S, GT3, Turbo, Boxster, Cayman etc) => make="Porsche"
- Mixed brands => separate records per car.

--------------------------------
MODEL NORMALIZATION (CRITICAL)
--------------------------------
Model MUST be normalized as: Family + (Generation/Platform)
Examples:
- "911 (993)"
- "911 (996.2)"
- "Boxster (987.2)"
- "Cayman (981)"
- "Cayenne (957)"
- "944", "968", "356" (no parentheses needed)
DO NOT place year in model.

--------------------------------
VARIANT NORMALIZATION
--------------------------------
Variant should contain the production variant/trim only (not the platform code, not the year).
Examples: "Carrera 4S", "Carrera S", "Turbo S", "GT3", "GT3 RS", "GT2", "GTS", "Targa", "Dakar", "Sport Classic", "S/T".

Shorthand mapping:
C2->Carrera 2; C4->Carrera 4; C2S->Carrera 2S; C4S->Carrera 4S.

Important:
- RS / GT3 / GT2 / S/T / Dakar are variants (keep in Variant).
- Turbo S is a distinct production model/variant (not cosmetic pack).

--------------------------------
NOTES FIELD RULES (VERY IMPORTANT)
--------------------------------
Put ALL of the following into notes (if present in the signature chunk):
- Model year (e.g., 1988, 2011, '97)
- Colours (e.g., Guards Red, Ocean Blue, Basalt Black, etc)
- Gearbox (Manual, PDK, Tiptronic)
- Engine sizes (2.9, 3.4, 3.6, 3.8, 4.0)
- Options/codes (X50, X51, PCCB, etc)
- Modifications (remap, coilovers, exhaust, suspension, aero, widebody, etc)
- ownership-ish extra context ("since 2003") but NOT the ownership label itself

DO NOT put in notes:
- Platform codes (993/996/997/987/etc) if already captured in Model
- Repeated trim names

--------------------------------
CHEAT SHEET: 911 GENERATIONS (YEAR -> MODEL PLATFORM)
--------------------------------
Use these rules when year is known AND the chunk is clearly a 911-family car:
- 1964–1973 => Original/Early 911
- 1974–1989 => G-Series / Impact bumper (incl 911 SC, 911 Carrera 3.2; Turbo=930)
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

"Gen 1 / Gen 2" terminology applies mainly to 996 / 997 / 991 / 992.

--------------------------------
FINAL OUTPUT BEHAVIOR
--------------------------------
- Produce one JSON car object per detected owned vehicle, preserving order.
- Use null for missing fields.
- Always include source_text.
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

# ----------------------------
# Small helpers
# ----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def safe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None

# ----------------------------
# Deterministic post-processing (hard rules)
# ----------------------------

# 4-digit years we reasonably expect in signatures; you can widen later if needed.
FOUR_DIGIT_YEAR_RE = re.compile(r"\b(19[6-9]\d|20[0-2]\d)\b")  # 1960–2029
# 2-digit year tokens like 97, '97, 97MY, MY97
TWO_DIGIT_YEAR_RE = re.compile(r"(?:\bMY)?(\d{2})(?:\bMY|\b)", re.IGNORECASE)

# Year accidentally placed in model like "911 (1988)" or "911(1988)"
MODEL_YEAR_IN_PARENS_RE = re.compile(r"\(\s*(19[6-9]\d|20[0-2]\d)\s*\)")

SOLD_TOKENS_RE = re.compile(r"\b(ex|ex-|sold|gone|previous|formerly|prior|old car|used to own|was my|had)\b", re.I)

# Lightweight enrichment lists (deterministic "pull into notes" help)
GEARBOX_TOKENS = ["pdk", "manual", "tiptronic", "tip", "cvt"]
COMMON_COLOUR_TOKENS = [
    "guards red", "ocean blue", "basalt black", "arctic silver", "carrera white",
    "gt silver", "seal grey", "speed yellow", "racing yellow", "polar silver",
    "midnight blue", "black", "white", "silver", "grey", "gray", "red", "blue", "green", "yellow"
]
MOD_TOKENS = [
    "x50", "x51", "pccb", "coilover", "coilovers", "remap", "mapped", "exhaust",
    "suspension", "short shift", "aero", "splitter", "ducktail", "roll cage", "cage",
    "bucket", "buckets", "carbon", "lw", "lightweight", "clubsport", "cs", "widebody",
]

def normalize_ownership(_raw: Optional[str], source_text: str) -> str:
    """
    Business rule:
    - Default to Current unless explicit sold/ex tokens appear in the supporting source_text.
    """
    st = (source_text or "").strip()
    if SOLD_TOKENS_RE.search(st):
        return "Sold"
    return "Current"

def extract_year_from_text(text: str) -> Optional[int]:
    """
    Extract a likely model year from source_text.
    Prefer 4-digit years.
    If only 2-digit appears: 70–99 => 19xx, 00–29 => 20xx
    """
    t = (text or "")

    m4 = FOUR_DIGIT_YEAR_RE.search(t)
    if m4:
        return int(m4.group(1))

    # Look for 2-digit with common formats:
    # "'97", "97", "MY97", "97MY"
    m = re.search(r"(?:\bMY)?('?)(\d{2})(?:\bMY|\b)", t, flags=re.IGNORECASE)
    if m:
        yy = int(m.group(2))
        if 70 <= yy <= 99:
            return 1900 + yy
        if 0 <= yy <= 29:
            return 2000 + yy

    return None

def ensure_notes_contains(notes: Optional[str], addition: str) -> Optional[str]:
    add = (addition or "").strip()
    if not add:
        return notes
    n = (notes or "").strip()
    if not n:
        return add
    # prevent obvious duplicates
    if add.lower() in n.lower():
        return n
    return f"{add}; {n}"

def move_year_out_of_model(model: Optional[str], notes: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    If model contains a literal year in parentheses (e.g. 911 (1988)), remove it and push year into notes.
    """
    if not model:
        return model, notes

    m = model.strip()
    m2 = MODEL_YEAR_IN_PARENS_RE.search(m)
    if m2:
        yr = m2.group(1)
        m = MODEL_YEAR_IN_PARENS_RE.sub("", m).strip()
        m = re.sub(r"\s+", " ", m).strip()
        m = m.replace("()", "").strip()
        notes = ensure_notes_contains(notes, yr)
        return (m or None), (notes or None)

    return model, notes

def apply_gt3_dot2_hint(model: Optional[str], source_text: str) -> Optional[str]:
    """
    If someone writes 'GT3.2' (common enthusiast shorthand), interpret '.2' as Gen 2 for the platform if present.
    """
    st = (source_text or "").lower()
    if "gt3.2" in st or "gt3 .2" in st:
        if "996" in st:
            return "911 (996.2)"
        if "997" in st:
            return "911 (997.2)"
        if "991" in st:
            return "911 (991.2)"
        if "992" in st:
            return "911 (992.2)"
    return model

def is_911_family(source_text: str, model: Optional[str], variant: Optional[str]) -> bool:
    st = (source_text or "").lower()
    m = (model or "").lower()
    v = (variant or "").lower()
    return (
        any(tok in st for tok in ["911", "carrera", "turbo", "gt3", "gt2", "rs", "targa"])
        or "911" in m
        or any(tok in v for tok in ["carrera", "turbo", "gt3", "gt2", "rs", "targa"])
    )

def enforce_911_platform_from_year(
    model: Optional[str],
    notes: Optional[str],
    source_text: str,
    variant: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    If year is present and the chunk is clearly 911-family, FORCE the correct platform from year.
    This prevents errors like 1997 => 996 or 1988 => 964/993.
    """
    year = extract_year_from_text(source_text)
    if not year or not is_911_family(source_text, model, variant):
        return model, notes

    # Always store the year in notes
    notes = ensure_notes_contains(notes, str(year))

    # Year -> platform mapping (aligned to your cheat sheet + sensible overlaps)
    if 1964 <= year <= 1973:
        return "911 (Early 911)", notes
    if 1974 <= year <= 1989:
        # More specific subtypes (SC/3.2) are variant-ish; model stays generic.
        # But you wanted 1984–1989 to be 3.2 Carrera explicitly, so we do that:
        if 1984 <= year <= 1989:
            return "911 (3.2 Carrera)", notes
        if 1978 <= year <= 1983:
            return "911 (SC)", notes
        return "911 (G-Series)", notes
    if 1989 <= year <= 1994:
        if year == 1994:
            return "911 (964 / 993)", notes
        return "911 (964)", notes
    if 1994 <= year <= 1998:
        return "911 (993)", notes
    if 1998 <= year <= 2001:
        return "911 (996.1)", notes
    if 2002 <= year <= 2004:
        return "911 (996.2)", notes
    if 2004 <= year <= 2008:
        return "911 (997.1)", notes
    if 2009 <= year <= 2012:
        return "911 (997.2)", notes
    if 2011 <= year <= 2015:
        return "911 (991.1)", notes
    if 2016 <= year <= 2019:
        return "911 (991.2)", notes
    if 2019 <= year <= 2024:
        return "911 (992.1)", notes
    if year >= 2024:
        return "911 (992.2)", notes

    return model, notes

def force_extracted_details_into_notes(notes: Optional[str], source_text: str) -> Optional[str]:
    """
    Deterministically ensure we capture obvious "notes-ish" details that the model might forget:
    - year
    - gearbox tokens
    - obvious colour tokens
    - common option/mod tokens
    """
    st = (source_text or "")
    lower = st.lower()

    # year
    year = extract_year_from_text(st)
    if year:
        notes = ensure_notes_contains(notes, str(year))

    # gearbox
    for g in GEARBOX_TOKENS:
        if re.search(rf"\b{re.escape(g)}\b", lower):
            notes = ensure_notes_contains(notes, g.upper() if g == "pdk" else g.title())

    # colours (best-effort; we keep as found token)
    for c in COMMON_COLOUR_TOKENS:
        if c in lower:
            notes = ensure_notes_contains(notes, c.title() if c.islower() else c)

    # mods/options
    for t in MOD_TOKENS:
        if re.search(rf"\b{re.escape(t)}\b", lower):
            notes = ensure_notes_contains(notes, t.upper() if t.startswith("x") else t)

    return notes

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
        print(f"[parse {run_id}] START {start_member_id}->{max_member_id} model={MODEL}", flush=True)

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
                            make = safe_str(car.get("make"))
                            model = safe_str(car.get("model"))
                            variant = safe_str(car.get("variant"))
                            notes = safe_str(car.get("notes"))
                            source_text = safe_str(car.get("source_text")) or ""
                            confidence = clamp01(float(car.get("confidence") or 0.0))

                            if not source_text.strip():
                                continue

                            # Hard business rules
                            ownership = normalize_ownership(safe_str(car.get("ownership")), source_text)

                            # Keep model clean
                            model, notes = move_year_out_of_model(model, notes)

                            # Interpret GT3.2 etc
                            model = apply_gt3_dot2_hint(model, source_text)

                            # Force year -> platform for 911-family (prevents the big "this should be simple" misses)
                            model, notes = enforce_911_platform_from_year(model, notes, source_text, variant)

                            # Ensure year/colour/mods/gearbox live in notes (best-effort)
                            notes = force_extracted_details_into_notes(notes, source_text)

                            pending_rows.append(
                                {
                                    "member_id": member_id,
                                    "run_id": src_run_id,
                                    "ownership": ownership,
                                    "make": make,
                                    "model": model,
                                    "variant": variant,
                                    "notes": notes,
                                    "source_text": source_text,
                                    "confidence": confidence,
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
        if not ACTIVE_RUN or ACTIVE_RUN.get("status") != "running":
            raise HTTPException(status_code=400, detail="No active run")
        STOP_FLAG = True
        ACTIVE_RUN["status"] = "stopping"
    return {"ok": True, "message": "Stopping soon (after current member finishes)."}
