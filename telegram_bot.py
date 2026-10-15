"""Telegram delivery for @RogueProtocolAlertBot. The token stays in the environment."""

import hashlib
import json
import os
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


class TelegramError(Exception):
    pass


def load_env(path=None):
    file_path = Path(path) if path else ENV_PATH
    if not file_path.is_file():
        return
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _redact(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    cleaned = str(text)
    if token:
        cleaned = cleaned.replace(token, "[token]")
    return cleaned


def _token():
    load_env()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set")
    return token


def configured():
    load_env()
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN", "").strip())


def chat_id():
    load_env()
    return os.environ.get("TELEGRAM_CHAT_ID", "").strip()


def _api(method, payload=None, timeout=20):
    url = f"https://api.telegram.org/bot{_token()}/{method}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise TelegramError(_redact(f"HTTP {exc.code} {detail}")) from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TelegramError(_redact(f"{exc.__class__.__name__}")) from None
    if not body.get("ok"):
        raise TelegramError(_redact(body.get("description") or "Telegram API error"))
    return body.get("result")


def get_me():
    result = _api("getMe", timeout=10)
    return {
        "id": result.get("id"),
        "username": result.get("username") or "",
        "name": result.get("first_name") or "",
    }


def _chat_from_update(update):
    for key in ("message", "edited_message", "channel_post", "my_chat_member"):
        payload = update.get(key) or {}
        chat = payload.get("chat") or {}
        if chat.get("id") is not None:
            return str(chat["id"]), str(chat.get("type") or "")
    return "", ""


def discover_chat_id():
    """Use a real getUpdates result. Never invent a chat id."""
    if chat_id():
        return chat_id(), "environment"
    updates = _api("getUpdates", {"timeout": 0}, timeout=15)
    found = ""
    kind = ""
    for update in updates or []:
        candidate, chat_type = _chat_from_update(update)
        if candidate:
            found, kind = candidate, chat_type
    if not found:
        return "", ""
    remember_chat_id(found)
    return found, kind or "discovered"


def remember_chat_id(value):
    text = str(value).strip()
    if not text:
        return
    os.environ["TELEGRAM_CHAT_ID"] = text
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.is_file() else []
    kept = [line for line in lines if not line.startswith("TELEGRAM_CHAT_ID=")]
    kept.append(f"TELEGRAM_CHAT_ID={text}")
    ENV_PATH.write_text("\n".join(kept) + "\n", encoding="utf-8")


def send_message(text, destination=None):
    target = str(destination or chat_id()).strip()
    if not target:
        raise TelegramError("TELEGRAM_CHAT_ID is not set")
    result = _api("sendMessage", {"chat_id": target, "text": text[:4000]}, timeout=8)
    return int(result.get("message_id"))


def send_document(path, caption):
    target = chat_id()
    if not target:
        raise TelegramError("TELEGRAM_CHAT_ID is not set")
    file_path = Path(path)
    content = file_path.read_bytes()
    boundary = "----NINE" + uuid.uuid4().hex
    chunks = []
    for key, value in (("chat_id", target), ("caption", caption[:900])):
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    chunks.append(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"{file_path.name}\"\r\n"
            "Content-Type: text/csv\r\n\r\n"
        ).encode()
        + content
        + b"\r\n"
    )
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{_token()}/sendDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise TelegramError(_redact(f"HTTP {exc.code} {detail}")) from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise TelegramError(_redact(exc.__class__.__name__)) from None
    if not payload.get("ok"):
        raise TelegramError(_redact(payload.get("description") or "sendDocument failed"))
    return int((payload.get("result") or {}).get("message_id") or 0)


def deliver_schedule(schedule_path, state_path):
    """Upload the real schedule file once per content hash."""
    file_path = Path(schedule_path)
    if not file_path.is_file():
        return "telegram-document-failed schedule file is missing"
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    state_file = Path(state_path)
    state = {}
    if state_file.is_file():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    if state.get("schedule_sha256") == digest:
        return "schedule already delivered; CSV was not uploaded again"
    try:
        message_id = send_document(
            file_path,
            "NINE schedule.csv. Columns are log_id,crew_id,start,end,job. REPLACE_ crew ids are placeholders.",
        )
    except TelegramError as exc:
        return f"telegram-document-failed {exc}"
    state["schedule_sha256"] = digest
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return f"telegram-document-sent message_id={message_id}"


def alert_text(payload):
    crew = str(payload.get("crew_id", ""))
    roster = "Crew id is a placeholder, not an official roster id." if crew.startswith("REPLACE_") else "Crew id comes from crew_map.csv."
    return "\n".join(
        [
            f"NINE fault {payload.get('log_id')}",
            f"Module: {payload.get('module')}",
            f"Severity: {payload.get('status') or 'not recorded'}",
            f"Evidence: {str(payload.get('message', ''))[:700]}",
            "Verdict source: Round 1 model file.",
            f"Action: {payload.get('action')} (deterministic rule, not a second model).",
            f"Specialist: {payload.get('role')} / {crew}",
            roster,
            f"Job: {payload.get('job')} | {payload.get('window')}",
            "Repair feed: follow-up is the scheduled end. Completion is not confirmed.",
        ]
    )
