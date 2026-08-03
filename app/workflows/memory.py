import json
import uuid
from datetime import datetime
from typing import Any

from app.logger import get_logger
from app.memory import redis_client
from app.workflows.constants import STATUS_ACTIVE

logger = get_logger(__name__)

WORKFLOW_TTL = 3600  # 1 hour


def create_workflow_model(
    workflow_type: str,
    step: str,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a new workflow model.
    """

    now = datetime.utcnow().isoformat()

    return {
        "workflow_id": str(uuid.uuid4()),
        "type": workflow_type,
        "status": STATUS_ACTIVE,
        "step": step,
        "data": data or {},
        "created_at": now,
        "updated_at": now,
    }


def create_workflow(
    phone_number: str,
    workflow: dict[str, Any],
) -> None:
    """
    Store a new workflow in Redis.
    """

    try:
        redis_client.setex(
            f"workflow:{phone_number}",
            WORKFLOW_TTL,
            json.dumps(workflow),
        )

        logger.info(
            f"Workflow created | phone={phone_number} | type={workflow['type']}"
        )

    except Exception as e:
        logger.error(
            f"Workflow creation failed | phone={phone_number} | error={e}"
        )


def get_workflow(
    phone_number: str,
) -> dict[str, Any] | None:
    """
    Retrieve workflow from Redis.
    """

    try:
        data = redis_client.get(f"workflow:{phone_number}")

        if not data:
            return None

        return json.loads(data)

    except Exception as e:
        logger.error(
            f"Workflow retrieval failed | phone={phone_number} | error={e}"
        )
        return None


def update_workflow(
    phone_number: str,
    workflow: dict[str, Any],
) -> None:
    """
    Update an existing workflow.
    """

    try:
        workflow["updated_at"] = datetime.utcnow().isoformat()

        redis_client.setex(
            f"workflow:{phone_number}",
            WORKFLOW_TTL,
            json.dumps(workflow),
        )

        logger.info(
            f"Workflow updated | phone={phone_number}"
        )

    except Exception as e:
        logger.error(
            f"Workflow update failed | phone={phone_number} | error={e}"
        )


def delete_workflow(
    phone_number: str,
) -> None:
    """
    Delete workflow from Redis.
    """

    try:
        redis_client.delete(f"workflow:{phone_number}")

        logger.info(
            f"Workflow deleted | phone={phone_number}"
        )

    except Exception as e:
        logger.error(
            f"Workflow deletion failed | phone={phone_number} | error={e}"
        )


def set_workflow_step(
    phone_number: str,
    step: str,
) -> None:
    """
    Update only the workflow step.
    """

    workflow = get_workflow(phone_number)

    if not workflow:
        return

    workflow["step"] = step

    update_workflow(phone_number, workflow)


def update_workflow_data(
    phone_number: str,
    data: dict[str, Any],
) -> None:
    """
    Merge new data into workflow data.
    """

    workflow = get_workflow(phone_number)

    if not workflow:
        return

    workflow.setdefault("data", {})
    workflow["data"].update(data)

    update_workflow(phone_number, workflow)


def clear_workflow_data(phone_number: str, *keys: str) -> None:
    """Remove temporary workflow data while keeping the workflow active."""
    workflow = get_workflow(phone_number)
    if not workflow:
        return
    for key in keys:
        workflow.get("data", {}).pop(key, None)
    update_workflow(phone_number, workflow)


def complete_workflow(
    phone_number: str,
) -> None:
    """
    Mark workflow as completed and remove it.
    """

    delete_workflow(phone_number)
