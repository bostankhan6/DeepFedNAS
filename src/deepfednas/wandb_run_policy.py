"""Pure policy helpers for choosing W&B run identity during training resume."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class WandbRunPolicy:
    """Resolved W&B identity for one training process."""

    run_id: Optional[str]
    run_name: str
    source: str
    predecessor_run_id: Optional[str] = None
    segment_start_round: Optional[int] = None


def resolve_wandb_run_policy(
    *,
    resume_training: bool,
    fresh_run_on_resume: bool,
    requested_run_id: Optional[str],
    checkpoint_run_id: Optional[str],
    run_name: str,
    checkpoint_next_round: Optional[int],
) -> WandbRunPolicy:
    """Resolve whether to create or resume a W&B run.

    Training-state resume and W&B-run resume are deliberately independent.
    Starting a fresh W&B segment avoids resumed-run heartbeat defects in some
    W&B SDK releases while preserving the checkpoint continuation exactly.
    """
    if fresh_run_on_resume and not resume_training:
        raise ValueError(
            "--wandb_fresh_run_on_resume requires --resume_training"
        )
    if fresh_run_on_resume and requested_run_id:
        raise ValueError(
            "--wandb_fresh_run_on_resume cannot be combined with "
            "--wandb_run_id_resume"
        )

    if fresh_run_on_resume:
        if checkpoint_next_round is None:
            raise ValueError(
                "A fresh W&B resume segment requires checkpoint next-round metadata"
            )
        return WandbRunPolicy(
            run_id=None,
            run_name=f"{run_name}_segment_r{int(checkpoint_next_round)}",
            source="fresh_resume_segment",
            predecessor_run_id=checkpoint_run_id,
            segment_start_round=int(checkpoint_next_round),
        )

    if requested_run_id:
        return WandbRunPolicy(
            run_id=requested_run_id,
            run_name=run_name,
            source="explicit",
        )
    if resume_training and checkpoint_run_id:
        return WandbRunPolicy(
            run_id=checkpoint_run_id,
            run_name=run_name,
            source="checkpoint",
        )
    return WandbRunPolicy(run_id=None, run_name=run_name, source="new")
