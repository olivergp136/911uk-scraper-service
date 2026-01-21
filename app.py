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
from pydantic import BaseModel

BASE_URL = "https://911uk.com"
LONDON = ZoneInfo("Europe/London")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
FORUM_USER = os.environ.get("FORUM_USER", "")
FORUM_PASS = os.environ.get("FORUM_PASS", "")

# Politeness: robots says crawl-delay: 5. We'll stay above that.
MIN_DELAY_SECONDS = float(os.environ.get("MIN_DELAY_SECONDS", "6.0"))  # >= 5
MAX_DELAY_SECONDS = float(os.environ.get("MAX_DELAY_SECONDS", "10.0"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "10"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

# Cutoff: 01/01/2024 00:00 Europe/London
DEFAULT_CUTOFF_LONDON = datetime(2024, 1, 1, 0, 0, 0, tzinfo=LONDON)
CUTOFF_EPOCH = int(DEFAULT_CUTOFF_LONDON.timestamp())

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    # We only raise at runtime when endpoints are used, so deploy can still boot.
    pass

app = FastAPI(title="911uk Scraper Service")

# In-memory run state (source of truth stored in Supabase; this is just for live status)
RUN_LOCK = threading.Lock()
ACTIVE_RUN: Optional[Dict[str, Any]] = None
STOP_FLAG = False


class StartRunRequest(BaseModel):
    max_member_id: int


def sb_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def sb_insert_run(max_member_id: int) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/scrape_runs"
    payload = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_member_id": max_member_id,
        "last_processed_member_id": 0,
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


def fetch(session: requests.Session, url: str) -> requests.Response:
    # Basic retry with backoff for transient failures
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            # 429 / 403 handling: slow down hard
            if r.status_code == 429:
                time.sleep(60 * min(attempt, 5))  # 60s, 120s, 180s...
                continue
            if r.status_code == 403:
                # Treat as "pause-worthy" to avoid hammering
                time.sleep(60 * 30)
                continue
            return r
        except requests.RequestException:
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

    # Fallback: hidden input _xfToken
    inp = soup.find("input", attrs={"name": "_xfToken"})
    if inp and inp.get("value"):
        return inp.get("value")

    return None


def login(session: requests.Session) -> None:
    """
    Login via POST /login/login using the field names you found.
    Must fetch a page first to get CSRF token + cookies.
    """
    # Step 1: load homepage to obtain cookies + token
    home = fetch(session, f"{BASE_URL}/")
    token = extract_xf_token(home.text)
    if not token:
        raise RuntimeError("Could not extract CSRF token (_xfToken/data-csrf) from homepage")

    # Step 2: submit login form
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

    # Step 3: sanity check by fetching a page that typically shows logged-in state
    # We’ll just ensure we don’t see the login form prompt.
    chk = fetch(session, f"{BASE_URL}/")
    if "Log in" in chk.text and "Stay logged in" in chk.text:
        # Not perfect, but a decent indicator in absence of a dedicated endpoint.
        raise RuntimeError("Login may have failed (homepage still looks logged-out)")


def is_not_found(html: str) -> bool:
    # Based on your uploaded "not found" source:
    return ('data-template="error"' in html) or ("Page not found" in html and "requested page could not be found" in html.lower())


def parse_profile_main(html: str) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    """
    Returns: username, joined_epoch, last_seen_epoch, location
    """
    soup = BeautifulSoup(html, "html.parser")

    # Username: common XenForo pattern
    username = None
    u = soup.select_one(".memberHeader-name .username")
    if u:
        username = u.get_text(strip=True)

    # Joined / Last seen: find <dt> text, then next <dd> time[data-timestamp]
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

    # Location: you had "From London" style.
    # Common XenForo: .memberHeader-blurb contains location link or plain text.
    location = None
    blurb = soup.select_one(".memberHeader-blurb")
    if blurb:
        text = blurb.get_text(" ", strip=True)
        # Often like "From London" or "Yas Marina · From London"
        # We'll extract the part after "From " if present.
        lower = text.lower()
        if "from " in lower:
            idx = lower.rfind("from ")
            location = text[idx + 5:].strip()
        else:
            location = text.strip() or None

    return username, joined_epoch, last_seen_epoch, location


def parse_signature_about(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")

    # Find header "Signature" and take next bbWrapper block
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


def run_scrape(run_id: str, max_member_id: int):
    global STOP_FLAG

    session = requests.Session()
    session.headers.update({
        "User-Agent": "911uk-research-scraper/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    ok = skipped_inactive = skipped_no_sig = not_found = errors = 0

    try:
        login(session)

        for member_id in range(1, max_member_id + 1):
            with RUN_LOCK:
                if STOP_FLAG:
                    sb_update_run(run_id, {
                        "status": "stopped",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "last_processed_member_id": member_id - 1,
                        "ok_count": ok,
                        "skipped_inactive_count": skipped_inactive,
                        "skipped_no_signature_count": skipped_no_sig,
                        "not_found_count": not_found,
                        "error_count": errors,
                    })
                    return

            # MAIN PROFILE
            try:
                r = fetch(session, f"{BASE_URL}/members/{member_id}/")
                polite_sleep()
                if r.status_code == 404 or is_not_found(r.text):
                    not_found += 1
                    if member_id % 50 == 0:
                        sb_update_run(run_id, {"last_processed_member_id": member_id, "not_found_count": not_found})
                    continue

                username, joined_epoch, last_seen_epoch, location = parse_profile_main(r.text)

                # if no last seen, treat as skip (or could count as error)
                if not last_seen_epoch:
                    skipped_inactive += 1
                    if member_id % 50 == 0:
                        sb_update_run(run_id, {"last_processed_member_id": member_id, "skipped_inactive_count": skipped_inactive})
                    continue

                if last_seen_epoch < CUTOFF_EPOCH:
                    skipped_inactive += 1
                    if member_id % 50 == 0:
                        sb_update_run(run_id, {"last_processed_member_id": member_id, "skipped_inactive_count": skipped_inactive})
                    continue

                # ABOUT PAGE (SIGNATURE)
                a = fetch(session, f"{BASE_URL}/members/{member_id}/about")
                polite_sleep()
                if a.status_code == 404 or is_not_found(a.text):
                    # weird, but treat as not found
                    not_found += 1
                    if member_id % 50 == 0:
                        sb_update_run(run_id, {"last_processed_member_id": member_id, "not_found_count": not_found})
                    continue

                signature = parse_signature_about(a.text)
                if not signature:
                    skipped_no_sig += 1
                    if member_id % 50 == 0:
                        sb_update_run(run_id, {"last_processed_member_id": member_id, "skipped_no_signature_count": skipped_no_sig})
                    continue

                # COMMIT TO DB (only now)
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

                if member_id % 50 == 0:
                    sb_update_run(run_id, {
                        "last_processed_member_id": member_id,
                        "ok_count": ok,
                        "skipped_inactive_count": skipped_inactive,
                        "skipped_no_signature_count": skipped_no_sig,
                        "not_found_count": not_found,
                        "error_count": errors,
                    })

            except Exception as e:
                errors += 1
                sb_update_run(run_id, {
                    "last_processed_member_id": member_id,
                    "error_count": errors,
                    "last_error": f"member_id={member_id}: {type(e).__name__}: {str(e)[:500]}",
                })
                # Cool-off to avoid repeated rapid failures
                time.sleep(20)

        # Finished
        sb_update_run(run_id, {
            "status": "done",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "last_processed_member_id": max_member_id,
            "ok_count": ok,
            "skipped_inactive_count": skipped_inactive,
            "skipped_no_signature_count": skipped_no_sig,
            "not_found_count": not_found,
            "error_count": errors,
        })

    except Exception as e:
        sb_update_run(run_id, {
            "status": "failed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_count": errors + 1,
            "last_error": f"{type(e).__name__}: {str(e)[:500]}",
        })


@app.get("/")
def home():
    return {
        "service": "911uk scraper",
        "endpoints": {
            "POST /start": {"max_member_id": "int"},
            "POST /stop": {},
            "GET /active": {},
        }
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

    with RUN_LOCK:
        if ACTIVE_RUN and ACTIVE_RUN.get("status") == "running":
            raise HTTPException(status_code=400, detail="A run is already in progress")
        STOP_FLAG = False

    run = sb_insert_run(req.max_member_id)
    run_id = run["id"]

    with RUN_LOCK:
        ACTIVE_RUN = {
            "active": True,
            "status": "running",
            "run_id": run_id,
            "max_member_id": req.max_member_id,
            "cutoff_london": DEFAULT_CUTOFF_LONDON.isoformat(),
        }

    t = threading.Thread(target=run_scrape, args=(run_id, req.max_member_id), daemon=True)
    t.start()

    return {"ok": True, "run_id": run_id}


@app.post("/stop")
def stop():
    global STOP_FLAG
    with RUN_LOCK:
        if not ACTIVE_RUN or ACTIVE_RUN.get("status") != "running":
            raise HTTPException(status_code=400, detail="No active run")
        STOP_FLAG = True
        ACTIVE_RUN["status"] = "stopping"
    return {"ok": True, "message": "Stopping soon (after current member finishes)."}
