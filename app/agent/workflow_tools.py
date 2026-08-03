from app.workflows.memory import create_workflow, create_workflow_model
from app.workflows.constants import (
    WORKFLOW_CHEQUE,
    STEP_UPLOAD_CHEQUE,
)


def start_cheque_workflow(phone_number: str) -> str:
    """
    Starts a cheque deposit workflow.
    """

    workflow = create_workflow_model(
        workflow_type=WORKFLOW_CHEQUE,
        step=STEP_UPLOAD_CHEQUE,
    )

    create_workflow(phone_number, workflow)

    return (
        "Cheque deposit started successfully.\n\n"
        "Please upload a clear image of the cheque."
    )