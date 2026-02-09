import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH (fixes "No module named thread")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from thread.phase_3_correlation._3_3_predictions import handle_realtime_prediction  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to kafka_message_dump.json")
    ap.add_argument("--resample", type=int, default=60, help="Resample seconds (informational; pipeline reads config too)")
    ap.add_argument("--dry_run", action="store_true", help="Don't write anything; only run predictions")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        msgs = json.load(f)

    # The dump can be list or {"messages":[...]}
    if isinstance(msgs, dict) and "messages" in msgs:
        msgs = msgs["messages"]
    if not isinstance(msgs, list):
        raise ValueError("Dump must be a list of messages or {'messages': [...]}")

    for i, message in enumerate(msgs, start=1):
        # Minimal scope setup: pid-level
        pid = message.get("joOpId") or message.get("pid") or f"dump_{i}"
        out = handle_realtime_prediction(
            message=message,
            scope="pid",
            scope_id=pid,
            group_by_stock=True,
            resample_seconds=args.resample,
            dry_run=True,
        )
        print(f"[{i}/{len(msgs)}] scope_id={pid} -> {out}")

        if args.dry_run:
            continue


if __name__ == "__main__":
    main()
