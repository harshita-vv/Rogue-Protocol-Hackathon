"""Execute actions chosen by actions.decide. Identification stays in actions.py."""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import actions
import telegram_bot

SCHEDULE_COLUMNS = ["log_id", "crew_id", "start", "end", "job"]


def load_crew_map(path):
    frame = pd.read_csv(path)
    required = {"module", "crew_id", "role", "contact"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"crew map is missing {sorted(missing)}")
    return frame.drop_duplicates("module").set_index("module", drop=False)


def explain_fault(row):
    """Fault brief for the demo: why this row was acted on, and a job title."""
    module = str(row.get("module", "unknown"))
    status = str(row.get("status", ""))
    message = str(row.get("message", ""))
    lowered = message.lower()
    reasons = [f"{module} was classified as a genuine fault."]
    if status in {"CRIT", "ERROR"}:
        reasons.append(f"Status is {status}.")
    if any(phrase in lowered for phrase in ("all is well", "nothing to see", "need not worry")):
        reasons.append("The message uses calming language, which hid real faults in Round 1.")
    if "[" in message and module not in message:
        reasons.append("The module named inside the message does not match this log.")
    if "o2" in lowered or "seal" in lowered or "scrubber" in lowered:
        job = f"Life-support check on {module}"
    elif "volt" in lowered or "breaker" in lowered or "brownout" in lowered:
        job = f"Power check on {module}"
    elif "dish" in lowered or "carrier" in lowered or "antenna" in lowered:
        job = f"Antenna check on {module}"
    elif "queue" in lowered or "packet" in lowered or "bus" in lowered:
        job = f"Bus check on {module}"
    else:
        job = f"Inspect {module}"
    return " ".join(reasons), job


def crew_load(schedule):
    """Custom warning: how many open jobs each specialist already holds."""
    if schedule.empty:
        return pd.DataFrame(columns=["crew_id", "open_jobs", "warning"])
    counts = schedule.groupby("crew_id").size().rename("open_jobs").reset_index()
    counts["warning"] = np_where_busy(counts["open_jobs"])
    return counts.sort_values("open_jobs", ascending=False)


def np_where_busy(counts):
    return ["busy" if int(count) >= 3 else "ok" for count in counts]


def overlap_warnings(schedule):
    """Flag a crew whose booked windows overlap, and crews carrying three or more jobs."""
    if schedule.empty:
        return []
    warnings = []
    for crew_id, group in schedule.groupby("crew_id"):
        times = group.copy()
        times["start_at"] = pd.to_datetime(times["start"], errors="coerce")
        times["end_at"] = pd.to_datetime(times["end"], errors="coerce")
        times = times.dropna(subset=["start_at", "end_at"]).sort_values("start_at")
        previous_end = None
        for row in times.itertuples(index=False):
            if previous_end is not None and row.start_at < previous_end:
                warnings.append(f"{crew_id}: {row.log_id} overlaps an earlier job")
            previous_end = row.end_at if previous_end is None else max(previous_end, row.end_at)
        if len(group) >= 3:
            warnings.append(f"{crew_id} has {len(group)} open jobs")
    return warnings


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return int(response.status)


def _outbox_ids(path):
    file_path = Path(path)
    if not file_path.is_file():
        return set()
    found = set()
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("accepted") is False:
            continue
        delivery = str(record.get("delivery", ""))
        if delivery.startswith("email-failed") or delivery.startswith("telegram-failed"):
            continue
        found.add(str(record.get("log_id", "")))
    return found


def delivery_accepted(delivery):
    text = str(delivery)
    return (
        text.startswith("email-dispatched")
        or text.startswith("telegram-dispatched")
        or text.startswith("local-outbox-only")
    )


def _email_delivery(payload, endpoint):
    if not endpoint:
        return "local-outbox-only (ROGUE_EMAIL_URL is not set)"
    last_error = "URLError"
    for _attempt in (1, 2):
        try:
            return f"email-dispatched HTTP {_post_json(endpoint, payload)}"
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc.__class__.__name__
    return f"email-failed {last_error}; saved to local outbox"


def _telegram_delivery(payload):
    """Send on Telegram when a token exists. Two tries. A missing chat id is not a sent message."""
    if not telegram_bot.configured():
        return None
    try:
        destination, _kind = telegram_bot.discover_chat_id()
    except telegram_bot.TelegramError as exc:
        return f"telegram-failed {exc}"
    if not destination:
        return "telegram-failed no chat id; open @RogueProtocolAlertBot, press Start, and send a message"
    try:
        message_id = telegram_bot.send_message(telegram_bot.alert_text(payload), destination)
        return f"telegram-dispatched message_id={message_id}"
    except telegram_bot.TelegramError as exc:
        return f"telegram-failed {exc}; saved to local outbox"


def send_alert(payload, outbox_path, url=None):
    """Deliver a fault alert. An explicit email url keeps tests off Telegram.

    Telegram is the live channel when TELEGRAM_BOT_TOKEN is set and no email url
    was forced. A missing token falls through to email or the labeled local outbox.
    """
    destination = Path(outbox_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if url is None:
        delivery = _telegram_delivery(payload)
        if delivery is None:
            delivery = _email_delivery(payload, os.environ.get("ROGUE_EMAIL_URL", "").strip())
    else:
        delivery = _email_delivery(payload, str(url).strip())
    record = {**payload, "delivery": delivery, "accepted": delivery_accepted(delivery)}
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return delivery


def load_schedule(path):
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return pd.DataFrame(columns=SCHEDULE_COLUMNS)
    frame = pd.read_csv(file_path)
    if list(frame.columns) != SCHEDULE_COLUMNS:
        raise ValueError(f"schedule columns must be {SCHEDULE_COLUMNS}")
    return frame


def _blank(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _next_window(schedule, crew_id, timestamp):
    start = pd.to_datetime(timestamp, errors="coerce")
    supplied = start
    if pd.isna(start):
        start = pd.Timestamp("2187-04-15 03:12:00")
    end = start + pd.Timedelta(minutes=30)
    crew = schedule[schedule["crew_id"].astype(str) == str(crew_id)]
    shifted = False
    if not crew.empty:
        occupied_end = pd.to_datetime(crew["end"], errors="coerce").max()
        if pd.notna(occupied_end) and start < occupied_end:
            start = occupied_end
            end = start + pd.Timedelta(minutes=30)
            shifted = True
    return start, end, shifted and pd.notna(supplied)


def _load_json(path, fallback):
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return fallback
    return json.loads(file_path.read_text(encoding="utf-8"))


def _save_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _draft(payload, job, start, end, delivery):
    contact = payload.get("contact", "")
    header = "DRAFT ONLY — contact is not on the official roster."
    if contact and not str(contact).startswith("REPLACE_"):
        header = "ALERT DRAFT"
    return "\n".join(
        [
            header,
            f"TO: {contact} ({payload.get('crew_id', '')})",
            f"ROLE: {payload.get('role', '')}",
            f"SUBJECT: NINE fault {payload.get('log_id')} / {payload.get('module')}",
            payload.get("explanation", ""),
            f"LOG: {payload.get('message', '')}",
            f"JOB: {job} | {start} → {end}",
            f"DELIVERY: {delivery}",
        ]
    )


def run_response(cases, crew_map, schedule_path, outbox_path, memory, repair_status_path, email_url=None):
    """Run decided actions. A failed live alert does not store or book that fault."""
    schedule = load_schedule(schedule_path)
    repair_status = _load_json(repair_status_path, {})
    # Schedule and outbox block a second alert. Memory keeps past incidents searchable
    # and does not block a fresh demonstration run.
    seen = set(schedule["log_id"].astype(str)) | _outbox_ids(outbox_path)
    incidents = []
    alerts = []
    counts = {
        "rows": 0,
        "matched": 0,
        "faults": 0,
        "false_alarms": 0,
        "malformed": 0,
        "unmatched": 0,
        "notified": 0,
        "notify_failed": 0,
        "remembered": 0,
        "scheduled": 0,
        "follow_ups": 0,
        "duplicates": 0,
        "held": 0,
    }

    for row in cases.reset_index(drop=True).to_dict(orient="records"):
        counts["rows"] += 1
        log_id = _blank(row.get("log_id")) or "(missing)"
        verdict = actions.parse_verdict(row.get("verdict"))
        matched = row.get("_merge", "both") == "both" and _blank(row.get("module"))
        incident = {
            "log_id": log_id,
            "module": _blank(row.get("module")) or "—",
            "kind": "malformed",
            "actions": "—",
            "specialist": "—",
            "crew_id": "—",
            "notification": "not sent",
            "memory": "not stored",
            "repair": "not scheduled",
            "follow_up": "—",
            "explanation": "",
            "message": _blank(row.get("message")),
            "draft": "",
            "job": "",
            "start": "",
            "end": "",
            "note": "malformed-skipped",
            "overlap": "",
            "verdict_value": "" if verdict is None else str(verdict),
            "verdict_source": "Round 1 model verdict" if verdict is not None else "not a model verdict",
            "action_source": "deterministic rule in actions.py",
            "evidence": _blank(row.get("message")),
            "log_status": _blank(row.get("status")) or "not recorded",
            "log_time": _blank(row.get("timestamp")),
            "executed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "error": "",
        }
        if not log_id or log_id == "(missing)" or verdict is None:
            counts["malformed"] += 1
            incidents.append(incident)
            continue
        if not matched:
            incident["kind"] = "unmatched"
            incident["note"] = "log-not-found"
            counts["unmatched"] += 1
            incidents.append(incident)
            continue
        counts["matched"] += 1
        chosen = actions.decide(verdict)
        incident["actions"] = " → ".join(chosen) if chosen else "—"
        if verdict == 0:
            incident["kind"] = "false alarm"
            incident["notification"] = "not sent"
            incident["memory"] = "not stored"
            incident["repair"] = "not scheduled"
            incident["note"] = "stand-down"
            counts["false_alarms"] += 1
            incidents.append(incident)
            continue
        counts["faults"] += 1
        incident["kind"] = "true fault"
        if log_id in seen:
            incident["note"] = "duplicate-skipped"
            incident["notification"] = "already sent"
            incident["memory"] = "already stored"
            incident["repair"] = "already scheduled"
            counts["duplicates"] += 1
            incidents.append(incident)
            continue
        if email_url is None and telegram_bot.configured() and counts["notified"] >= 4:
            incident["note"] = "held"
            incident["notification"] = "not sent — one press sends 4 Telegram alerts"
            incident["memory"] = "not stored"
            incident["repair"] = "not scheduled"
            counts["held"] += 1
            incidents.append(incident)
            continue
        module = _blank(row.get("module"))
        if module not in crew_map.index:
            incident["note"] = f"no crew for {module or '(blank)'}"
            incident["notification"] = "not sent — no crew mapping"
            counts["notify_failed"] += 1
            incidents.append(incident)
            continue
        crew = crew_map.loc[module]
        crew_id = str(crew["crew_id"])
        contact = str(crew["contact"])
        incident["specialist"] = str(crew["role"])
        incident["crew_id"] = crew_id
        if crew_id.startswith("REPLACE_") or contact.startswith("REPLACE_"):
            incident["specialist"] = f"{crew['role']} (roster id not set)"
        explanation, job = explain_fault(row)
        start, end, shifted = _next_window(schedule, crew_id, row.get("timestamp"))
        payload = {
            "log_id": log_id,
            "module": module,
            "crew_id": crew_id,
            "role": str(crew["role"]),
            "contact": contact,
            "explanation": explanation,
            "message": _blank(row.get("message")),
            "status": _blank(row.get("status")) or "not recorded",
            "action": incident["actions"],
            "job": job,
            "window": f"{start.strftime('%Y-%m-%d %H:%M:%S')} → {end.strftime('%Y-%m-%d %H:%M:%S')}",
        }
        delivery = send_alert(payload, outbox_path, email_url)
        if str(delivery).startswith("telegram-failed") or str(delivery).startswith("email-failed"):
            incident["error"] = delivery
        incident["explanation"] = explanation
        incident["job"] = job
        incident["notification"] = delivery
        incident["draft"] = _draft(payload, job, start, end, delivery)
        incident["start"] = start.strftime("%Y-%m-%d %H:%M:%S")
        incident["end"] = end.strftime("%Y-%m-%d %H:%M:%S")
        if shifted:
            incident["overlap"] = "Start moved so this crew does not overlap an open job."
        if not delivery_accepted(delivery):
            incident["note"] = "notify-failed"
            incident["memory"] = "not stored"
            incident["repair"] = "not scheduled"
            counts["notify_failed"] += 1
            incidents.append(incident)
            continue
        stored = memory.add({**payload, "job": job, "action": "remember"})
        incident["memory"] = "stored" if stored else "already stored"
        if stored:
            counts["remembered"] += 1
        schedule = pd.concat(
            [
                schedule,
                pd.DataFrame(
                    [
                        {
                            "log_id": log_id,
                            "crew_id": crew_id,
                            "start": incident["start"],
                            "end": incident["end"],
                            "job": job,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        repair_status[log_id] = {
            "status": "follow-up queued",
            "follow_up_at": incident["end"],
            "job": job,
            "crew_id": crew_id,
            "note": "Next check is the scheduled end. No live repair feed is connected.",
        }
        incident["repair"] = "scheduled"
        incident["follow_up"] = incident["end"]
        incident["note"] = "booked"
        counts["notified"] += 1
        counts["scheduled"] += 1
        counts["follow_ups"] += 1
        seen.add(log_id)
        alerts.append({**payload, "delivery": delivery, "job": job, "start": incident["start"], "end": incident["end"]})
        incidents.append(incident)

    destination = Path(schedule_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before_rows = len(load_schedule(schedule_path)) if destination.is_file() else 0
    schedule.to_csv(destination, index=False)
    _save_json(repair_status_path, repair_status)
    telegram_sent = any(str(item.get("delivery", "")).startswith("telegram-dispatched") for item in alerts)
    if telegram_sent and len(schedule) > before_rows:
        schedule_delivery = telegram_bot.deliver_schedule(destination, destination.parent / "telegram_state.json")
    elif telegram_sent:
        schedule_delivery = "schedule unchanged; CSV was not uploaded again"
    else:
        schedule_delivery = "schedule CSV was not sent to Telegram"
    if telegram_sent:
        connector = "telegram-dispatched"
    elif email_url is not None:
        connector = "local-outbox-only (email override)" if not str(email_url).strip() else "email endpoint"
    elif telegram_bot.configured() and not telegram_bot.chat_id():
        connector = "telegram-failed no chat id; open @RogueProtocolAlertBot, press Start, and send a message"
    elif telegram_bot.configured() and counts["notify_failed"] and not counts["notified"]:
        connector = "telegram-failed; nothing was marked sent"
    else:
        endpoint = os.environ.get("ROGUE_EMAIL_URL", "").strip()
        if not endpoint:
            connector = "local-outbox-only (ROGUE_EMAIL_URL is not set)"
        elif counts["notify_failed"] and not counts["notified"]:
            connector = "email endpoint failed; nothing was marked sent"
        else:
            connector = "ROGUE_EMAIL_URL accepted the alert POST"
    stages = _stages(counts, connector)
    return {
        "schedule": schedule,
        "alerts": alerts,
        "incidents": incidents,
        "stages": stages,
        "counts": counts,
        "overlaps": overlap_warnings(schedule),
        "connector": connector,
        "schedule_delivery": schedule_delivery,
        "backend": getattr(memory, "backend", ""),
    }


def _stages(counts, connector):
    ingest_state = "done" if counts["rows"] else "failed"
    identify_state = "done" if counts["rows"] else "waiting"
    decide_state = "done" if counts["faults"] or counts["false_alarms"] else "waiting"
    if counts["notify_failed"] and not counts["notified"]:
        notify_state = "failed"
    elif counts["notified"] or counts["false_alarms"]:
        notify_state = "done"
    else:
        notify_state = "waiting"
    text = str(connector)
    if text.startswith("telegram-dispatched"):
        notify_mode = "telegram"
    elif "chat id missing" in text:
        notify_mode = "outbox, no chat"
    elif text.startswith("local-outbox"):
        notify_mode = "demo outbox"
    elif text.startswith("telegram-failed"):
        notify_mode = "telegram failed"
    else:
        notify_mode = "endpoint"
    return {
        "ingest": {"state": ingest_state, "detail": f"{counts['rows']} verdicts · {counts['matched']} logs"},
        "identify": {
            "state": identify_state,
            "detail": f"{counts['faults']} faults · {counts['false_alarms']} false alarms",
        },
        "decide": {
            "state": decide_state,
            "detail": "repair only for verdict 1",
        },
        "notify": {
            "state": notify_state,
            "detail": f"{counts['notified']} sent · {notify_mode}"
            + (f" · {counts['held']} held" if counts.get("held") else ""),
        },
        "remember": {"state": "done" if counts["remembered"] else "waiting", "detail": f"{counts['remembered']} stored"},
        "schedule": {"state": "done" if counts["scheduled"] else "waiting", "detail": f"{counts['scheduled']} booked"},
        "follow_up": {
            "state": "done" if counts["follow_ups"] else "waiting",
            "detail": f"{counts['follow_ups']} queued at job end",
        },
    }


def book_faults(logs, predictions, crew_map, schedule_path, outbox_path, memory, email_url=None):
    """Act only on verdict 1. A log id already alerted or booked is skipped."""
    frame = logs.reset_index(drop=True).copy()
    if len(frame) != len(predictions):
        raise ValueError("logs and predictions have different lengths")
    frame["verdict"] = list(predictions)
    frame["_merge"] = "both"
    result = run_response(
        frame,
        crew_map,
        schedule_path,
        outbox_path,
        memory,
        Path(schedule_path).parent / "repair_status.json",
        email_url=email_url,
    )
    notes = [
        {"log_id": item["log_id"], "status": item["note"]}
        for item in result["incidents"]
        if item.get("note")
    ]
    return result["schedule"], result["alerts"], notes
