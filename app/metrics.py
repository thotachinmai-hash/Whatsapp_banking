from datetime import datetime
from app.logger import get_logger

logger = get_logger(__name__)

metrics_store = {
    "total_messages": 0,
    "voice_messages": 0,
    "text_messages": 0,
    "unsupported_messages": 0,
    "transcription_calls": 0,
    "transcription_errors": 0,
    "agent_calls": 0,
    "agent_errors": 0,
    "tool_calls": {},
    "whatsapp_sends": 0,
    "whatsapp_errors": 0,
    "total_response_time_ms": 0,
    "request_count": 0,
    "errors": 0,
    "events": [],
    # Per-stage timing for the latency target (<10s end-to-end) — one entry
    # per named stage (transcription, agent, tts, whatsapp_send, ...), so
    # /metrics can show where time is actually going without needing to
    # grep individual "duration=" log lines.
    "stage_durations": {},
}


def _track_stage(stage: str, duration_ms: float) -> None:
    bucket = metrics_store["stage_durations"].setdefault(stage, {"count": 0, "total_ms": 0.0})
    bucket["count"] += 1
    bucket["total_ms"] += duration_ms


def log_message_received(phone_number: str, msg_type: str, trace_id: str):
    metrics_store["total_messages"] += 1
    if msg_type == "voice":
        metrics_store["voice_messages"] += 1
    elif msg_type == "text":
        metrics_store["text_messages"] += 1
    else:
        metrics_store["unsupported_messages"] += 1

    metrics_store["events"].append({
        "timestamp": datetime.now().isoformat(),
        "type": "message_received",
        "trace_id": trace_id,
        "phone": phone_number[-4:],  # Only last 4 digits for privacy
        "msg_type": msg_type
    })
    logger.info(f"[{trace_id}] METRIC | message_received | type={msg_type}")


def log_transcription(success: bool, duration_ms: float, trace_id: str):
    metrics_store["transcription_calls"] += 1
    if not success:
        metrics_store["transcription_errors"] += 1
    _track_stage("stt", duration_ms)
    logger.info(f"[{trace_id}] METRIC | transcription | success={success} | duration={duration_ms:.2f}ms")


def log_agent_call(success: bool, duration_ms: float, tools_called: list, trace_id: str):
    metrics_store["agent_calls"] += 1
    if not success:
        metrics_store["agent_errors"] += 1
    for tool in tools_called:
        metrics_store["tool_calls"][tool] = metrics_store["tool_calls"].get(tool, 0) + 1
    _track_stage("agent", duration_ms)
    logger.info(f"[{trace_id}] METRIC | agent_call | success={success} | duration={duration_ms:.2f}ms | tools={tools_called}")


def log_tts(success: bool, duration_ms: float, trace_id: str):
    _track_stage("tts", duration_ms)
    logger.info(f"[{trace_id}] METRIC | tts | success={success} | duration={duration_ms:.2f}ms")


def log_media_upload(success: bool, duration_ms: float, trace_id: str):
    """Isolated to the two-step Graph media-upload+send call
    (app/services/whatsapp.py::send_voice_message) — kept separate from
    "tts" (synthesis) and "whatsapp_send" (plain text) so a slow voice
    reply can be attributed to the right stage instead of one blended
    number."""
    _track_stage("media_upload", duration_ms)
    logger.info(f"[{trace_id}] METRIC | media_upload | success={success} | duration={duration_ms:.2f}ms")


def log_tool_call(tool_name: str, duration_ms: float, trace_id: str):
    """One bucket per tool name — search_bank_documents is RAG, every
    other tool is a DB read/write, so this naturally gives the RAG-vs-
    DB/tools split without hardcoding a category list here. tools.py's
    own per-call "TOOL | name | duration=" log line already exists for
    per-request debugging; this feeds the same number into the
    aggregated /metrics endpoint so the bottleneck can be seen at a
    glance instead of grepped for."""
    _track_stage(f"tool:{tool_name}", duration_ms)


def log_whatsapp_send(success: bool, trace_id: str, duration_ms: float | None = None):
    metrics_store["whatsapp_sends"] += 1
    if not success:
        metrics_store["whatsapp_errors"] += 1
    if duration_ms is not None:
        _track_stage("whatsapp_send", duration_ms)
    duration_part = f" | duration={duration_ms:.2f}ms" if duration_ms is not None else ""
    logger.info(f"[{trace_id}] METRIC | whatsapp_send | success={success}{duration_part}")


def log_request(duration_ms: float, trace_id: str):
    metrics_store["request_count"] += 1
    metrics_store["total_response_time_ms"] += duration_ms
    _track_stage("request_total", duration_ms)
    logger.info(f"[{trace_id}] METRIC | request_complete | duration={duration_ms:.2f}ms")


def get_metrics() -> dict:
    avg_response = 0
    if metrics_store["request_count"] > 0:
        avg_response = round(
            metrics_store["total_response_time_ms"] / metrics_store["request_count"], 2
        )
    return {
        "total_messages": metrics_store["total_messages"],
        "voice_messages": metrics_store["voice_messages"],
        "text_messages": metrics_store["text_messages"],
        "unsupported_messages": metrics_store["unsupported_messages"],
        "transcription_calls": metrics_store["transcription_calls"],
        "transcription_errors": metrics_store["transcription_errors"],
        "agent_calls": metrics_store["agent_calls"],
        "agent_errors": metrics_store["agent_errors"],
        "tool_calls": metrics_store["tool_calls"],
        "whatsapp_sends": metrics_store["whatsapp_sends"],
        "whatsapp_errors": metrics_store["whatsapp_errors"],
        "avg_response_time_ms": avg_response,
        "total_requests": metrics_store["request_count"],
        "avg_stage_duration_ms": {
            stage: round(bucket["total_ms"] / bucket["count"], 2)
            for stage, bucket in metrics_store["stage_durations"].items()
            if bucket["count"] > 0
        },
        "recent_events": metrics_store["events"][-20:]
    }
