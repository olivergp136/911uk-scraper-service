from fastapi import FastAPI, HTTPException
import threading
from typing import Optional, Dict, Any

app = FastAPI(title="911uk Signature Parser")

# Simple in-memory run state (same pattern as scraper)
RUN_LOCK = threading.Lock()
ACTIVE_RUN: Optional[Dict[str, Any]] = None
STOP_FLAG = False


@app.get("/")
def home():
    return {
        "service": "911uk signature parser",
        "status": "ok",
        "endpoints": {
            "POST /start": {},
            "POST /stop": {},
            "GET /active": {}
        }
    }


@app.get("/active")
def active():
    with RUN_LOCK:
        return ACTIVE_RUN or {"active": False}


@app.post("/start")
def start():
    global ACTIVE_RUN, STOP_FLAG
    with RUN_LOCK:
        if ACTIVE_RUN and ACTIVE_RUN.get("status") == "running":
            raise HTTPException(status_code=400, detail="Run already active")
        STOP_FLAG = False
        ACTIVE_RUN = {
            "active": True,
            "status": "running"
        }
    return {"ok": True, "message": "Parser run started (stub)"}


@app.post("/stop")
def stop():
    global STOP_FLAG
    with RUN_LOCK:
        if not ACTIVE_RUN or ACTIVE_RUN.get("status") != "running":
            raise HTTPException(status_code=400, detail="No active run")
        STOP_FLAG = True
        ACTIVE_RUN["status"] = "stopped"
    return {"ok": True, "message": "Parser run stopped"}

