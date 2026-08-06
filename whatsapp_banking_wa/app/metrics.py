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
    "events": []
}


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
    logger.info(f"[{trace_id}] METRIC | transcription | success={success} | duration={duration_ms:.2f}ms")


def log_agent_call(success: bool, duration_ms: float, tools_called: list, trace_id: str):
    metrics_store["agent_calls"] += 1
    if not success:
        metrics_store["agent_errors"] += 1
    for tool in tools_called:
        metrics_store["tool_calls"][tool] = metrics_store["tool_calls"].get(tool, 0) + 1
    logger.info(f"[{trace_id}] METRIC | agent_call | success={success} | duration={duration_ms:.2f}ms | tools={tools_called}")


def log_whatsapp_send(success: bool, trace_id: str):
    metrics_store["whatsapp_sends"] += 1
    if not success:
        metrics_store["whatsapp_errors"] += 1
    logger.info(f"[{trace_id}] METRIC | whatsapp_send | success={success}")


def log_request(duration_ms: float, trace_id: str):
    metrics_store["request_count"] += 1
    metrics_store["total_response_time_ms"] += duration_ms
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
        "recent_events": metrics_store["events"][-20:]
    }
