import os
import time
import json
import random
import threading
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_URL = "https://911uk.com"
LONDON = ZoneInfo("Europe/London")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
FORUM_USER = os.environ.get("FORUM_USER", "")
FORUM_PASS = os.environ.get("FORUM_PASS", "")

# Politeness: robots says crawl-delay: 5. We'll stay above that.
MIN_DELAY_SECONDS = float(os.environ.get("MIN_DELAY_SECONDS", "6.0"))  # >= 5
MAX_DELAY_SECONDS = float(os.environ.get("MAX_DELAY_SECONDS", "10.0"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# How often to write progress to Supabase + emit a log line
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "10"))

# Cutoff: 01/01/2024 00:00 Europe/London
DEFAULT_CUTOFF_LONDON = datetime(2024, 1, 1, 0, 0, 0, tzinfo=LONDON)
CUTOFF_EPOCH = int(DEFAULT_CUTOFF_LONDON.timestamp())

app = FastAPI(title="911uk Scraper Service")

# In-memory run state (source of truth stored in Supabase; this is just for live status)
RUN_LOCK = threading.Lock()
ACTIVE_RUN: Optional[Dict[str, Any]] = None
STOP_FLAG = False


class StartRunRequest(BaseModel):
    # Start from this member id (so you can resume at e.g. 33000)
    start_member_id: int = Field(default=1, ge=1)
    # Stop at this member id
    max_member_id: int = Field(..., ge=1)


def sb_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def sb_insert_run(start_member_id: int, max_member_id: int) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/scrape_runs"
    payload = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "start_member_id": start_member_id,
        "max_member_id": max_member_id,
        "last_processed_member_id": start_member_id - 1,
        "cutoff_london": DEFAULT_CUTOFF_LONDON.astimezone(timezone.utc).isoformat(),
    }
    r = requests.post(url, headers=sb_headers(), data=json.dumps(payload), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase insert run failed: {r.status_code} {r.text}")
    return r.json()[0]


def sb_update_run(run_id: str, patch: Dict[str, Any]) -> None:
    url = f"{SUPABASE_URL}/rest/v1/scrape_runs?id=eq.{run_id}"
    r = requests.patch(url, headers=sb_headers(), data=json.dumps(patch), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase update run failed: {r.status_code} {r.text}")


def sb_upsert_member(row: Dict[str, Any]) -> None:
    url = f"{SUPABASE_URL}/rest/v1/members"
    # Upsert by PK (member_id)
    headers = sb_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    r = requests.post(url, headers=headers, data=json.dumps(row), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase upsert member failed: {r.status_code} {r.text}")


def polite_sleep():
    delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
    time.sleep(delay)


def small_jitter():
    # Tiny pause between "profile" -> "about" like a real user click
    time.sleep(random.uniform(0.2, 0.8))


def fetch(session: requests.Session, url: str) -> requests.Response:
    # Basic retry with backoff for transient failures
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)

            # 429 / 403 handling: slow down hard
            if r.status_code == 429:
                print(f"[fetch] 429 on {url} (attempt {attempt}) -> backing off", flush=True)
                time.sleep(60 * min(attempt, 5))  # 60s, 120s, 180s...
                continue

            if r.status_code == 403:
                # Real 403 responses (Cloudflare / permission / rate limiting)
                print(f"[fetch] 403 on {url} (attempt {attempt}) -> cooling off 30m", flush=True)
                time.sleep(60 * 30)
                continue

            return r
        except requests.RequestException as e:
            print(f"[fetch] exception on {url} (attempt {attempt}): {type(e).__name__}", flush=True)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("Unreachable")


def extract_xf_token(html: str) -> Optional[str]:
    """
    XenForo often includes CSRF in <html data-csrf="..."> or meta tags.
    We'll look for common patterns.
    """
    soup = BeautifulSoup(html, "html.parser")

    html_tag = soup.find("html")
    if html_tag and html_tag.get("data-csrf"):
        return html_tag.get("data-csrf")

    meta = soup.find("meta", attrs={"name": "csrf-token"})
    if meta and meta.get("content"):
        return meta.get("content")

    inp = soup.find("input", attrs={"name": "_xfToken"})
    if inp and inp.get("value"):
        return inp.get("value")

    return None


def login(session: requests.Session) -> None:
    home = fetch(session, f"{BASE_URL}/")
    token = extract_xf_token(home.text)
    if not token:
        raise RuntimeError("Could not extract CSRF token (_xfToken/data-csrf) from homepage")

    post_url = f"{BASE_URL}/login/login/"
    data = {
        "_xfToken": token,
        "login": FORUM_USER,
        "password": FORUM_PASS,
        "remember": "1",
        "_xfRedirect": f"{BASE_URL}/",
    }
    r = session.post(post_url, data=data, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    if r.status_code >= 400:
        raise RuntimeError(f"Login POST failed: {r.status_code}")

    chk = fetch(session, f"{BASE_URL}/")
    if "Log in" in chk.text and "Stay logged in" in chk.text:
        raise RuntimeError("Login may have failed (homepage still looks logged-out)")


def is_not_found(html: str) -> bool:
    return (
        'data-template="error"' in html
        or ("page not found" in html.lower() and "requested page could not be found" in html.lower())
    )


def is_private_profile(html: str) -> bool:
    lower = html.lower()
    return (
        "this member limits who may view their full profile" in lower
        or "limits who may view their full profile" in lower
    )


def parse_profile_main(html: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")

    username = None
    u = soup.select_one(".memberHeader-name .username")
    if u:
        username = u.get_text(strip=True)

    def find_timestamp(label: str) -> Optional[int]:
        dt_nodes = soup.find_all("dt")
        for dt in dt_nodes:
            if dt.get_text(strip=True).lower() == label.lower():
                dd = dt.find_next_sibling("dd")
                if not dd:
                    continue
                t = dd.find("time")
                if t and t.get("data-timestamp"):
                    try:
                        return int(t.get("data-timestamp"))
                    except ValueError:
                        return None
        return None

    joined_epoch = find_timestamp("Joined")
    last_seen_epoch = find_timestamp("Last seen")

    location = None
    blurb = soup.select_one(".memberHeader-blurb")
    if blurb:
        text = blurb.get_text(" ", strip=True)
        lower = text.lower()
        if "from " in lower:
            idx = lower.rfind("from ")
            location = text[idx + 5 :].strip()
        else:
            location = text.strip() or None

    return username, joined_epoch, last_seen_epoch, location


def parse_signature_about(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    headers = soup.select("h4.block-textHeader")
    for h in headers:
        if h.get_text(strip=True).lower() == "signature":
            wrapper = h.find_next("div", class_="bbWrapper")
            if wrapper:
                txt = wrapper.get_text("\n", strip=True)
                cleaned = "\n".join([line.strip() for line in txt.splitlines() if line.strip()])
                return cleaned if cleaned else None
            return None
    return None


def epoch_to_utc_ts(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def run_scrape(run_id: str, start_member_id: int, max_member_id: int):
    global STOP_FLAG, ACTIVE_RUN

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "911uk-research-scraper/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )

    ok = skipped_inactive = skipped_no_sig = not_found = errors = 0

    def update_progress(member_id: int):
        sb_update_run(
            run_id,
            {
                "last_processed_member_id": member_id,
                "ok_count": ok,
                "skipped_inactive_count": skipped_inactive,
                "skipped_no_signature_count": skipped_no_sig,
                "not_found_count": not_found,
                "error_count": errors,
            },
        )
        print(
            f"[run {run_id}] member_id={member_id} ok={ok} inactive={skipped_inactive} "
            f"no_sig={skipped_no_sig} not_found={not_found} errors={errors}",
            flush=True,
        )

    try:
        print(
            f"[run {run_id}] START start_member_id={start_member_id} max_member_id={max_member_id} cutoff_epoch={CUTOFF_EPOCH}",
            flush=True,
        )
        login(session)
        print(f"[run {run_id}] LOGIN OK", flush=True)

        for member_id in range(start_member_id, max_member_id + 1):
            with RUN_LOCK:
                if STOP_FLAG:
                    print(f"[run {run_id}] STOP requested -> stopping at member_id={member_id}", flush=True)
                    sb_update_run(
                        run_id,
                        {
                            "status": "stopped",
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "last_processed_member_id": member_id - 1,
                            "ok_count": ok,
                            "skipped_inactive_count": skipped_inactive,
                            "skipped_no_signature_count": skipped_no_sig,
                            "not_found_count": not_found,
                            "error_count": errors,
                        },
                    )
                    ACTIVE_RUN = {"active": False, "status": "stopped", "run_id": run_id}
                    return

            did_raise = False
            try:
                r = fetch(session, f"{BASE_URL}/members/{member_id}/")

                if is_private_profile(r.text):
                    errors += 1
                    msg = f"member_id={member_id}: private_profile"
                    print(f"[run {run_id}] ERROR {msg}", flush=True)
                    sb_update_run(
                        run_id,
                        {
                            "last_processed_member_id": member_id,
                            "error_count": errors,
                            "last_error": msg,
                        },
                    )
                    if member_id % PROGRESS_EVERY == 0:
                        update_progress(member_id)
                    continue

                if r.status_code == 404 or is_not_found(r.text):
                    not_found += 1
                    if member_id % PROGRESS_EVERY == 0:
                        update_progress(member_id)
                    continue

                username, joined_epoch, last_seen_epoch, location = parse_profile_main(r.text)

                if not last_seen_epoch:
                    skipped_inactive += 1
                    if member_id % PROGRESS_EVERY == 0:
                        update_progress(member_id)
                    continue

                if last_seen_epoch < CUTOFF_EPOCH:
                    skipped_inactive += 1
                    if member_id % PROGRESS_EVERY == 0:
                        update_progress(member_id)
                    continue

                small_jitter()
                a = fetch(session, f"{BASE_URL}/members/{member_id}/about")

                if is_private_profile(a.text):
                    errors += 1
                    msg = f"member_id={member_id}: private_profile_about"
                    print(f"[run {run_id}] ERROR {msg}", flush=True)
                    sb_update_run(
                        run_id,
                        {
                            "last_processed_member_id": member_id,
                            "error_count": errors,
                            "last_error": msg,
                        },
                    )
                    if member_id % PROGRESS_EVERY == 0:
                        update_progress(member_id)
                    continue

                if a.status_code == 404 or is_not_found(a.text):
                    not_found += 1
                    if member_id % PROGRESS_EVERY == 0:
                        update_progress(member_id)
                    continue

                signature = parse_signature_about(a.text)
                if not signature:
                    skipped_no_sig += 1
                    if member_id % PROGRESS_EVERY == 0:
                        update_progress(member_id)
                    continue

                row = {
                    "member_id": member_id,
                    "username": username,
                    "location": location,
                    "joined_at": epoch_to_utc_ts(joined_epoch) if joined_epoch else None,
                    "last_seen_at": epoch_to_utc_ts(last_seen_epoch),
                    "signature_raw": signature,
                    "run_id": run_id,
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }
                sb_upsert_member(row)
                ok += 1

                if member_id % PROGRESS_EVERY == 0:
                    update_progress(member_id)

            except Exception as e:
                did_raise = True
                errors += 1
                msg = f"member_id={member_id}: {type(e).__name__}: {str(e)[:500]}"
                print(f"[run {run_id}] ERROR {msg}", flush=True)
                sb_update_run(
                    run_id,
                    {
                        "last_processed_member_id": member_id,
                        "error_count": errors,
                        "last_error": msg,
                    },
                )
                time.sleep(20)

            if not did_raise:
                polite_sleep()

        update_progress(max_member_id)
        sb_update_run(
            run_id,
            {
                "status": "done",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "last_processed_member_id": max_member_id,
                "ok_count": ok,
                "skipped_inactive_count": skipped_inactive,
                "skipped_no_signature_count": skipped_no_sig,
                "not_found_count": not_found,
                "error_count": errors,
            },
        )
        print(
            f"[run {run_id}] DONE ok={ok} inactive={skipped_inactive} no_sig={skipped_no_sig} "
            f"not_found={not_found} errors={errors}",
            flush=True,
        )
        with RUN_LOCK:
            ACTIVE_RUN = {"active": False, "status": "done", "run_id": run_id}

    except Exception as e:
        errors += 1
        msg = f"{type(e).__name__}: {str(e)[:500]}"
        print(f"[run {run_id}] FAILED {msg}", flush=True)
        sb_update_run(
            run_id,
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error_count": errors,
                "last_error": msg,
            },
        )
        with RUN_LOCK:
            ACTIVE_RUN = {"active": False, "status": "failed", "run_id": run_id, "last_error": msg}


@app.get("/")
def home():
    return {
        "service": "911uk scraper",
        "endpoints": {
            "POST /start": {"start_member_id": "int (optional, default 1)", "max_member_id": "int"},
            "POST /stop": {},
            "GET /active": {},
        },
        "config": {
            "min_delay_seconds": MIN_DELAY_SECONDS,
            "max_delay_seconds": MAX_DELAY_SECONDS,
            "progress_every": PROGRESS_EVERY,
            "mode": "one_polite_sleep_per_member",
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
    if not FORUM_USER or not FORUM_PASS:
        raise HTTPException(status_code=500, detail="Forum credentials env vars not set")

    if req.start_member_id > req.max_member_id:
        raise HTTPException(status_code=400, detail="start_member_id must be <= max_member_id")

    with RUN_LOCK:
        if ACTIVE_RUN and ACTIVE_RUN.get("status") == "running":
            raise HTTPException(status_code=400, detail="A run is already in progress")
        STOP_FLAG = False

    run = sb_insert_run(req.start_member_id, req.max_member_id)
    run_id = run["id"]

    with RUN_LOCK:
        ACTIVE_RUN = {
            "active": True,
            "status": "running",
            "run_id": run_id,
            "start_member_id": req.start_member_id,
            "max_member_id": req.max_member_id,
            "cutoff_london": DEFAULT_CUTOFF_LONDON.isoformat(),
            "progress_every": PROGRESS_EVERY,
        }

    t = threading.Thread(target=run_scrape, args=(run_id, req.start_member_id, req.max_member_id), daemon=True)
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
