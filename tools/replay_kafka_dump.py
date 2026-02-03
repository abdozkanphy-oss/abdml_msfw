"""Replay a Kafka JSON dump through the realtime prediction code.

Usage (from repo root):
    python tools/replay_kafka_dump.py --path /mnt/data/kafka_message_dump.json --resample 60

This runs in DRY-RUN mode (no Cassandra writes). It helps validate:
- message parsing
- batch id mapping (prod_order_reference_no)
- model buffer + resampling path
"""

import argparse
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = ROOT / "thread" / "phase_3_correlation" / "_3_3_predictions.py"

spec = importlib.util.spec_from_file_location("_3_3_predictions", str(PRED_PATH))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

handle_realtime_prediction = mod.handle_realtime_prediction

def _normalize_message(msg: dict) -> dict:
    """Apply the same reference mapping we use in the Kafka consumer."""
    if not isinstance(msg, dict):
        return {}

    # Canonical batch id
    jo_ref = msg.get("joRef")
    if jo_ref not in (None, "", "None"):
        msg["prod_order_reference_no"] = jo_ref
        msg.setdefault("job_order_reference_no", jo_ref)

    ref_no = msg.get("refNo")
    if ref_no not in (None, "", "None", 0, "0"):
        msg["job_order_reference_no"] = ref_no

    # Ensure output_stock fields exist (optional)
    if "output_stock_no" not in msg or "output_stock_name" not in msg:
        pl = msg.get("prodList")
        if isinstance(pl, list) and pl and isinstance(pl[0], dict):
            msg.setdefault("output_stock_no", pl[0].get("stNo"))
            msg.setdefault("output_stock_name", pl[0].get("stNm"))

    # operation fields
    msg.setdefault("operationname", msg.get("opNm"))
    msg.setdefault("operationno", msg.get("opNo"))
    msg.setdefault("operationtaskcode", msg.get("opTc"))

    # Timestamp: prefer measDt if present
    if not msg.get("crDt"):
        out = msg.get("outVals")
        if isinstance(out, list) and out and isinstance(out[0], dict) and out[0].get("measDt"):
            msg["crDt"] = out[0]["measDt"]

    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to kafka_message_dump.json")
    ap.add_argument("--resample", type=int, default=60, help="Resample seconds (60 default)")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--min_train_points", type=int, default=60)
    ap.add_argument("--algo", type=str, default="LSTM")
    args = ap.parse_args()

    p = Path(args.path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and "messages" in payload:
        messages = payload["messages"]
    elif isinstance(payload, list):
        messages = payload
    else:
        raise ValueError("JSON must be a list of messages or {messages: [...]}.")

    for i, raw in enumerate(messages, start=1):
        msg = _normalize_message(dict(raw))
        batch_id = msg.get("prod_order_reference_no") or msg.get("joRef") or msg.get("refNo") or "0"

        status = handle_realtime_prediction(
            message=msg,
            lookback=args.lookback,
            epochs=args.epochs,
            min_train_points=args.min_train_points,
            algorithm=args.algo,
            seed_history=None,
            scope="batch",
            scope_id=str(batch_id),
            group_by_stock=True,
            resample_seconds=args.resample,
            dry_run=True,
        )

        print(f"[{i}/{len(messages)}] batch={batch_id} ok={status.get('ok')} wrote={status.get('wrote')} reason={status.get('reason')}")


if __name__ == "__main__":
    main()
