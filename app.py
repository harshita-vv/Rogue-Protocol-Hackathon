"""NINE // AUTONOMOUS RESPONSE DESK. Run: streamlit run app.py"""

import hashlib
import html
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

import actions
import agent
import data_io
import memory as memory_store
import telegram_bot

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
CREW_PATH = ROOT / "crew_map.csv"
SCHEDULE_PATH = OUTPUT / "schedule.csv"
OUTBOX_PATH = OUTPUT / "outbox.jsonl"
MEMORY_PATH = OUTPUT / "incidents.json"
REPAIR_PATH = OUTPUT / "repair_status.json"
LAST_RUN_PATH = OUTPUT / "last_run.json"

STAGES = (
    ("ingest", "INGEST", "↓"),
    ("identify", "IDENTIFY", "◎"),
    ("decide", "DECIDE", "⬡"),
    ("notify", "NOTIFY", "✉"),
    ("remember", "REMEMBER", "◌"),
    ("schedule", "SCHEDULE", "▤"),
    ("follow_up", "FOLLOW UP", "↻"),
)

IDLE_STAGES = {
    key: {"state": "waiting", "detail": "waiting for a verdict file"}
    for key, _label, _icon in STAGES
}


@st.cache_data(show_spinner=False)
def load_catalog():
    candidates = []
    env = os.environ.get("ROGUE_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend([Path("/tmp/rogue-protocol"), ROOT, Path.cwd()])
    for path in candidates:
        if not all((path / name).is_file() for name in ("train.csv", "val.csv", "test.csv")):
            continue
        train, val, test = data_io.load_splits(str(path))
        frames = []
        for name, frame in (("train", train), ("val", val), ("test", test)):
            copy = frame.copy()
            copy["split"] = name
            frames.append(copy)
        return pd.concat(frames, ignore_index=True), str(path)
    raise FileNotFoundError(
        "Ship logs were not found. Set ROGUE_DATA_DIR to the folder with train.csv, val.csv, and test.csv."
    )


def metric_table(cases):
    """Score only rows that carry a real ground-truth label. Predictions never fill that column."""
    if "ground_truth" not in cases.columns:
        return None
    labels = []
    predictions = []
    for row in cases.itertuples(index=False):
        label = actions.parse_verdict(getattr(row, "ground_truth", None))
        prediction = actions.parse_verdict(getattr(row, "verdict", None))
        if label is None or prediction is None:
            continue
        labels.append(label)
        predictions.append(prediction)
    if not labels:
        return None
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def live_sample(verdicts):
    """One fault and one false alarm. A full file must not block the page or flood Telegram."""
    column = actions.prediction_column(verdicts)
    if column is None or "log_id" not in verdicts.columns:
        return verdicts.head(0)
    marked = verdicts.copy()
    marked["_parsed"] = marked[column].map(actions.parse_verdict)
    faults = marked[marked["_parsed"] == 1].head(1)
    quiet = marked[marked["_parsed"] == 0].head(1)
    sample = pd.concat([faults, quiet], ignore_index=True)
    return sample.drop(columns=["_parsed"])


def execute(verdicts, catalog, source):
    verdicts = live_sample(verdicts)
    cases = actions.attach_logs(verdicts, catalog)
    crew = agent.load_crew_map(CREW_PATH)
    store = memory_store.IncidentMemory(MEMORY_PATH)
    result = agent.run_response(
        cases,
        crew,
        SCHEDULE_PATH,
        OUTBOX_PATH,
        store,
        REPAIR_PATH,
    )
    result["metrics"] = metric_table(cases)
    result["log_source"] = source
    result["backend"] = store.backend
    slim = {key: value for key, value in result.items() if key != "schedule"}
    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_PATH.write_text(json.dumps(slim), encoding="utf-8")
    return result


def load_last_run():
    if not LAST_RUN_PATH.is_file():
        return None
    try:
        return json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def inject_theme():
    st.markdown(
        """
        <style>
        .stApp { background: #070b12; }
        h1 { letter-spacing: 0.04em; font-weight: 650; }
        .stage { border: 1px solid #1c3b34; background: #0d151c; padding: 0.55rem 0.45rem;
                 min-height: 7.2rem; border-radius: 2px; }
        .stage.done { border-color: #3DDC97; }
        .stage.failed { border-color: #ff5d73; }
        .stage.waiting { opacity: 0.5; }
        .stage-icon { color: #3DDC97; font-size: 1.05rem; }
        .stage-name { font-size: 0.68rem; letter-spacing: 0.08em; margin-top: 0.2rem; }
        .stage-state { font-size: 0.72rem; text-transform: uppercase; color: #d7e2ea; }
        .stage-detail { font-size: 0.72rem; color: #8ea3af; margin-top: 0.25rem; }
        div[data-testid="stMetric"] { background: #101820; border: 1px solid #1c3b34; padding: 0.4rem 0.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tracker(stages):
    columns = st.columns(len(STAGES))
    for column, (key, label, icon) in zip(columns, STAGES):
        info = stages.get(key, {"state": "waiting", "detail": "—"})
        state = html.escape(str(info.get("state", "waiting")))
        detail = html.escape(str(info.get("detail", "—")))
        column.markdown(
            f"""
            <div class="stage {state}">
              <div class="stage-icon">{icon}</div>
              <div class="stage-name">{label}</div>
              <div class="stage-state">{state}</div>
              <div class="stage-detail">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def telegram_alert_count():
    if not OUTBOX_PATH.is_file():
        return 0
    count = 0
    for line in OUTBOX_PATH.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("delivery", "")).startswith("telegram-dispatched"):
            count += 1
    return count


@st.cache_data(ttl=60, show_spinner=False)
def telegram_identity():
    telegram_bot.load_env()
    if not telegram_bot.configured():
        return {"ok": False, "detail": "TELEGRAM_BOT_TOKEN is not set"}
    try:
        identity = telegram_bot.get_me()
    except telegram_bot.TelegramError as exc:
        return {"ok": False, "detail": str(exc)}
    return {"ok": True, "username": identity.get("username") or ""}


def status_row(result):
    counts = (result or {}).get("counts", {})
    connector = (result or {}).get("connector") or "Waiting for a verdict file."
    booked = len(agent.load_schedule(SCHEDULE_PATH)) if SCHEDULE_PATH.is_file() else 0
    columns = st.columns(4)
    columns[0].metric("Processed logs", counts.get("matched", 0))
    columns[1].metric("Detected faults", counts.get("faults", 0))
    columns[2].metric("Telegram alerts", telegram_alert_count())
    columns[3].metric("Scheduled repairs", booked)
    st.caption(connector)
    if result and result.get("schedule_delivery"):
        st.caption(f"Schedule file: {result['schedule_delivery']}")


def incident_frame(incidents, faults_only):
    rows = incidents or []
    if faults_only:
        rows = [row for row in rows if row.get("kind") == "true fault"]
    if not rows:
        return pd.DataFrame(
            columns=["log_id", "verdict", "evidence", "action", "specialist", "status", "notification"]
        )
    return pd.DataFrame(
        [
            {
                "log_id": row.get("log_id"),
                "verdict": row.get("verdict_value", ""),
                "evidence": str(row.get("evidence") or "")[:90],
                "action": row.get("actions"),
                "specialist": row.get("specialist"),
                "status": row.get("note"),
                "notification": row.get("notification"),
            }
            for row in rows
        ]
    )


def main():
    telegram_bot.load_env()
    st.set_page_config(page_title="NINE // AUTONOMOUS RESPONSE DESK", layout="wide")
    inject_theme()
    st.title("NINE // AUTONOMOUS RESPONSE DESK")
    st.caption("Verdict file in. One press sends at most 4 Telegram alerts, then the schedule file. False alarms stay silent.")

    try:
        catalog, source = load_catalog()
        st.caption(f"Ship logs attached from `{source}`. Upload a verdict CSV only.")
    except FileNotFoundError as exc:
        catalog, source = None, ""
        st.error(str(exc))

    if LAST_RUN_PATH.is_file():
        stamp = LAST_RUN_PATH.stat().st_mtime
        if st.session_state.get("last_run_mtime") != stamp:
            st.session_state["result"] = load_last_run()
            st.session_state["last_run_mtime"] = stamp
    elif "result" not in st.session_state:
        st.session_state["result"] = None

    upload = st.file_uploader("Verdict CSV", type="csv")
    if upload is not None:
        raw = upload.getvalue()
        st.session_state["verdict_raw"] = raw
        st.session_state["verdict_name"] = upload.name
        try:
            preview = pd.read_csv(io.BytesIO(raw))
            column = actions.prediction_column(preview)
            fault_count = 0
            if column:
                fault_count = sum(actions.parse_verdict(value) == 1 for value in preview[column])
            st.caption(f"{upload.name} is loaded, {len(preview)} rows, {fault_count} predicted faults. Nothing is sent until you press the button.")
            st.info("One press sends the first fault only, plus one false alarm on screen. The page should be back in a few seconds.")
        except Exception as exc:
            st.error(f"Could not read the verdict file: {exc}")
    run_clicked = st.button("RUN RESPONSE SEQUENCE", type="primary")
    if run_clicked and catalog is not None:
        raw = st.session_state.get("verdict_raw")
        if not raw:
            st.warning("Upload a verdict CSV. The ship logs are already attached.")
        else:
            try:
                verdicts = pd.read_csv(io.BytesIO(raw))
            except Exception as exc:
                st.error(f"Could not read the verdict file: {exc}")
                verdicts = None
            if verdicts is not None:
                with st.spinner("Sending 1 Telegram alert, then the schedule file."):
                    try:
                        st.session_state["result"] = execute(verdicts, catalog, source)
                        st.session_state["last_run_mtime"] = LAST_RUN_PATH.stat().st_mtime if LAST_RUN_PATH.is_file() else None
                    except Exception as exc:
                        st.error(f"Response sequence stopped: {exc}")
    elif run_clicked:
        st.warning("Ship logs are not attached, so the sequence cannot start.")

    result = st.session_state.get("result") or {}
    status_row(result)
    tracker(result.get("stages") or IDLE_STAGES)
    report = result.get("metrics")
    if report:
        with st.expander("Ground-truth score", expanded=False):
            columns = st.columns(8)
            for column, key in zip(columns, report):
                value = report[key]
                column.metric(key, f"{value:.3f}" if isinstance(value, float) else value)
    elif result:
        st.caption("No ground-truth labels on these logs, so TP/FP/TN/FN stay hidden.")

    incidents = result.get("incidents") or []
    faults = [row for row in incidents if row.get("kind") == "true fault"]
    incidents_tab, draft_tab, memory_tab, schedule_tab, tools_tab = st.tabs(
        ["Incidents", "Alert draft", "Memory", "Schedule", "Tools"]
    )

    with incidents_tab:
        st.caption(f"{len(faults)} faults on the board. {result.get('counts', {}).get('false_alarms', 0)} false alarms stood down and were not booked.")
        show_all = st.toggle("Include false alarms", value=False)
        table = incident_frame(incidents, faults_only=not show_all)
        choices = [row["log_id"] for row in (incidents if show_all else faults)]
        if not table.empty:
            st.dataframe(table.head(12), width="stretch", height=280, hide_index=True)
        if choices:
            selected_id = st.selectbox("Open one incident", choices)
            selected = next(row for row in incidents if row["log_id"] == selected_id)
            with st.expander(f"{selected_id} · {selected.get('kind')}", expanded=True):
                st.caption(f"Verdict source: {selected.get('verdict_source', 'Round 1 model verdict')}. Action source: {selected.get('action_source', 'deterministic rule')}.")
                st.write(selected.get("explanation") or "No fault brief. This row was not treated as a repair.")
                st.write(f"Evidence: {selected.get('evidence') or '—'}")
                st.write(
                    f"Log status `{selected.get('log_status', 'not recorded')}` at `{selected.get('log_time') or '—'}`."
                )
                timeline = pd.DataFrame(
                    [
                        {"stage": "identify", "result": selected.get("kind"), "at": selected.get("executed_at")},
                        {"stage": "decide", "result": selected.get("actions"), "at": selected.get("executed_at")},
                        {"stage": "notify", "result": selected.get("notification"), "at": selected.get("executed_at")},
                        {"stage": "remember", "result": selected.get("memory"), "at": selected.get("executed_at")},
                        {"stage": "schedule", "result": selected.get("repair"), "at": selected.get("start")},
                        {"stage": "follow up", "result": selected.get("follow_up"), "at": selected.get("end")},
                    ]
                )
                st.dataframe(timeline, width="stretch", hide_index=True)
                if selected.get("error"):
                    st.error(selected["error"])
                if selected.get("overlap"):
                    st.warning(selected["overlap"])
                if selected.get("job"):
                    st.write(f"Job `{selected['job']}` from {selected.get('start')} to {selected.get('end')}.")
                query = selected.get("explanation") or selected.get("message") or ""
                if query and MEMORY_PATH.is_file():
                    found = [
                        item
                        for item in memory_store.IncidentMemory(MEMORY_PATH).search(query, limit=4)
                        if str(item.get("log_id")) != str(selected_id)
                    ]
                    if found:
                        st.write("Similar stored incidents")
                        found_frame = pd.DataFrame(found)
                        keep = [column for column in ("log_id", "module", "job") if column in found_frame.columns]
                        st.dataframe(found_frame[keep], width="stretch", hide_index=True)
                    else:
                        st.caption("No similar incident is stored yet.")
        else:
            st.caption("No incidents yet.")

    with draft_tab:
        drafts = [row for row in faults if row.get("draft")]
        if not drafts:
            st.caption("A draft appears here after a fault is accepted for notification.")
        else:
            picked = st.selectbox("Draft", [row["log_id"] for row in drafts], key="draft-id")
            body = next(row["draft"] for row in drafts if row["log_id"] == picked)
            st.code(body)
            st.caption(next(row["notification"] for row in drafts if row["log_id"] == picked))

    with memory_tab:
        stored = 0
        if MEMORY_PATH.is_file():
            try:
                stored = len(json.loads(MEMORY_PATH.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                stored = 0
        st.caption(
            f"Backend: {(result.get('backend') or 'tfidf')}. {stored} past incidents stay searchable across fresh runs."
        )
        query = st.text_input("Similar-incident search")
        if query and MEMORY_PATH.is_file():
            found = memory_store.IncidentMemory(MEMORY_PATH).search(query)
            if found:
                st.dataframe(pd.DataFrame(found), width="stretch", height=280, hide_index=True)
            else:
                st.caption("No stored incident matched.")
        elif query:
            st.caption("Memory is empty.")

    with schedule_tab:
        if SCHEDULE_PATH.is_file():
            schedule = agent.load_schedule(SCHEDULE_PATH)
            st.dataframe(schedule, width="stretch", height=280, hide_index=True)
            st.download_button(
                "Download schedule.csv",
                schedule.to_csv(index=False),
                file_name="schedule.csv",
                mime="text/csv",
            )
            warnings = agent.overlap_warnings(schedule)
            for warning in warnings:
                st.warning(warning)
            if not warnings:
                st.caption("No crew overlaps on the current schedule.")
        else:
            st.caption("schedule.csv is not written yet.")

    with tools_tab:
        st.subheader("Telegram")
        identity = telegram_identity()
        if identity.get("ok"):
            st.write(f"getMe succeeded for @{identity.get('username')}. The token is not shown.")
        else:
            st.error(identity.get("detail") or "Telegram is not configured.")
        if st.button("Check Telegram chat"):
            try:
                found, kind = telegram_bot.discover_chat_id()
            except telegram_bot.TelegramError as exc:
                st.error(str(exc))
            else:
                if found:
                    st.success(f"Chat discovered ({kind}). Run the response sequence to send alerts.")
                else:
                    st.warning("No chat yet. Open @RogueProtocolAlertBot, press Start, and send a message. Then press this button again.")
        if telegram_bot.chat_id():
            st.write(f"Destination chat id: {telegram_bot.chat_id()}.")
        else:
            st.warning("Open @RogueProtocolAlertBot, press Start, and send a message. No chat id will be guessed.")
        st.write(f"Telegram alerts confirmed in the outbox: {telegram_alert_count()}.")
        st.write("n8n and LangChain are not installed. Alerts use the Telegram Bot API directly.")
        st.write("Crew roster: replace every `REPLACE_` crew id and contact in `crew_map.csv`. Those values are not official ids.")
        st.write("Repair feed: follow-up is the scheduled end time in `outputs/repair_status.json`. No live completion signal is connected.")
        if CREW_PATH.is_file():
            st.dataframe(pd.read_csv(CREW_PATH), width="stretch", hide_index=True)
        if SCHEDULE_PATH.is_file():
            st.dataframe(agent.crew_load(agent.load_schedule(SCHEDULE_PATH)), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
