"""Persistent summaries of training runs for experiments and reporting."""

import csv
from datetime import datetime, timezone
from pathlib import Path


RUN_SUMMARY_FIELDS = [
    "timestamp_utc", "mode", "duration_seconds", "duration_minutes",
    "epochs_completed", "epochs_requested", "best_validation_loss",
    "train_samples", "validation_samples", "batch_size",
    "gradient_accumulation_steps", "backbone_learning_rate",
    "head_learning_rate", "weight_decay", "scheduler_first_cycle_epochs",
    "freeze_backbone_epochs",
]


def save_training_summary(output_path: str, summary: dict) -> Path:
    """Append one paper-friendly training-run summary and return its CSV path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {field: summary.get(field, "") for field in RUN_SUMMARY_FIELDS}
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path
