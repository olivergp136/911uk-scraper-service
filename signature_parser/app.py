import os
import json
import time
import threading
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
PAGE_SIZE = int(os.environ.get("PAGE_SIZE", "500"))          # Supabase fetch page size
WRITE_BATCH_SIZE = int(os.environ.get("WRITE_BATCH_SIZE", "500"))
SLEEP_BETWEEN_MEMBERS = float(os.environ.get("SLEEP_BETWEEN_MEMBERS", "0.15"))

app = FastAPI(title="911uk Signature Parser Service")

RUN_LOCK = threading.Lock()
ACTIVE_RUN: Optional[Dict[str, Any]] = None
STOP_FLAG = False

client = OpenAI(api_key=OPENAI_API_KEY)

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
    We only want rows with signature_raw not null.
    """
    url = f"{SUPABASE_URL}/rest/v1/members"
    params = {
        "select": "member_id,run_id,signature_raw",
        "member_id": f"gte.{after_member_id}",
        "signature_raw": "not.is.null",
        "order": "member_id.asc",
        "limit": str(limit),
    }
    # Upper bound
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
# Porsche-savvy interpretation rules (for the model)
# ----------------------------

RESPONSE_SCHEMA = {
    "name": "signature_car_extract",
    "strict": True,
    "schema": {
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
    },
}

SYSTEM_PROMPT = """You are an expert interpreter of enthusiast forum signatures with deep knowledge of Porsche shorthand.
Your task: extract each distinct car mentioned in the signature into structured fields.
The signature may contain multiple cars, cars on the same line, and cars spread across lines. Preserve meaning of line breaks.

Hard rules:
- Output MUST be valid JSON matching the provided schema (no prose).
- Extract 0..N cars. If none, return {"cars": []}.
- ownership must be exactly one of: Current, Sold, Unknown.
  - Use Sold only when clearly indicated (sold, gone, ex-, previous, formerly, "was", etc).
  - Use Current only when clearly indicated (current, now, my, have, "daily", etc).
  - Otherwise Unknown.
- Never invent details. If unsure, use nulls and/or lower confidence.
- notes: put all extra info here (year, colours, gearbox, options, mods, locations, etc).
- source_text MUST be a verbatim substring from the signature that supports the extracted car (may include line breaks).

Porsche-specific guidance:
- If Porsche is implied by shorthand (e.g., 964/993/996/997/991/992, C2S/C4S, GT3, Turbo S), set make="Porsche".
- model should be searchable:
  - Examples: "911 (997.2)", "911 (997.1)", "911 (964 / 993)", "Cayman (987.2)", "Boxster (986)".
- variant should be normalized where possible:
  - C2/C4/C2S/C4S => Carrera 2 / Carrera 4 / Carrera 2 S / Carrera 4 S (when clearly a 911 context)
  - "TT" in 911 context usually means Turbo (e.g., 993TT => 993 Turbo)
- Gen 1 / Gen 2 inference for 996/997/991/992:
  Use .1/.2 only when you can be confident (explicitly stated OR deducible from year ranges below):
  - 996.1 ~ 1998–2001; 996.2 ~ 2002–2004
  - 997.1 ~ 2004–2008; 997.2 ~ 2009–2012
  - 991.1 ~ 2012–2015; 991.2 ~ 2016–2019
  - 992.1 ~ 2019–2023; 992.2 ~ 2024–present
  If uncertain, omit .1/.2.
- Crossover ambiguity:
  If a year could map to two generations (e.g., 1994 could be late 964 or early 993), include both in model like "911 (964 / 993)".
"""

def call_ai(signature_raw: str) -> Dict[str, Any]:
    """
    Calls the model with Structured Outputs (JSON Schema), so it returns strict JSON.
    """
    user_prompt = f"Signature:\n---\n{signature_raw}\n---\nExtract the cars from this signature."

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": RESPONSE_SCHEMA,
            }
        },
        temperature=TEMPERATURE,
    )

    out = resp.output_text or ""
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
                # No more rows with signature_raw; we’re done for this range.
                break

            for m in members:
                with RUN_LOCK:
                    if STOP_FLAG:
                        break

                member_id = int(m["member_id"])
                sig = (m.get("signature_raw") or "")
                src_run_id = m.get("run_id")

                last_processed = member_id

                # Shouldn't happen, but guard.
                if not sig.strip():
                    continue

                # Retry wrapper for AI call
                last_err = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        result = call_ai(sig)
                        cars = result.get("cars", []) if isinstance(result, dict) else []
                        # Build member_cars rows
                        for car in cars:
                            ownership = safe_str(car.get("ownership")) or "Unknown"
                            make = safe_str(car.get("make"))
                            model = safe_str(car.get("model"))
                            variant = safe_str(car.get("variant"))
                            notes = safe_str(car.get("notes"))
                            source_text = safe_str(car.get("source_text")) or ""
                            confidence = clamp01(float(car.get("confidence") or 0.0))

                            # Must have evidence
                            if not source_text.strip():
                                continue

                            pending_rows.append(
                                {
                                    "member_id": member_id,
                                    "run_id": src_run_id,  # link back to scrape run
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
                    sb_update_parse_run(run_id, {"last_error": msg, "error_count": errors, "last_processed_member_id": last_processed})

                # Periodic write & progress
                if member_id % PROGRESS_EVERY == 0:
                    if pending_rows:
                        # write in chunks
                        for i in range(0, len(pending_rows), WRITE_BATCH_SIZE):
                            sb_upsert_member_cars(pending_rows[i : i + WRITE_BATCH_SIZE])
                        pending_rows = []
                    progress_log()

                # Gentle pacing (this is for the AI API stability/cost, not 911uk)
                time.sleep(SLEEP_BETWEEN_MEMBERS)

            # Advance cursor to next ID after the last member in this page
            cursor = int(members[-1]["member_id"]) + 1

            # Flush pending rows occasionally between pages too
            if pending_rows:
                for i in range(0, len(pending_rows), WRITE_BATCH_SIZE):
                    sb_upsert_member_cars(pending_rows[i : i + WRITE_BATCH_SIZE])
                pending_rows = []

        # Final flush + done
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
