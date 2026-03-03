from __future__ import annotations

from pathlib import Path
from typing import Any

from researchclaw.config import ResearchClawConfig
from researchclaw.fsm.states import State
from researchclaw.models import TrialMeta


def handle_merge_loop(
    trial_dir: Path,
    meta: TrialMeta,
    config: ResearchClawConfig,
    chat_interface: Any,
) -> State:
    """Handle the MERGE_LOOP state (stub).

    Merge functionality is not yet implemented. Prints a message and
    returns to DECIDE.
    """
    if chat_interface is not None:
        chat_interface.send(
            "Merge is not yet implemented. Returning to decision menu."
        )
    return State.DECIDE
