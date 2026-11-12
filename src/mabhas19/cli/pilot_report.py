"""Closed-pilot (M2) review and report, from the logs only (no engine call, no model).

``python -m mabhas19.cli.pilot_report report [--log-dir data/chat]``
    metrics to ``<log-dir>/reports/m2_report_<time>.{json,md}`` and the Markdown on stdout;
``python -m mabhas19.cli.pilot_report pending``
    requests without a review (ID, participant, response type, question, feedback);
``python -m mabhas19.cli.pilot_report review --request <id> --correct <judgement>
    --answerable <judgement> --triage <class> [--wrong-value] [--unsupported-number]
    [--note "..."] [--reviewer name]``
    (judgement: yes, no, uncertain or not_applicable)
    appends one manual review to ``reviews.jsonl``; the request log is never changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mabhas19.api.chatlog import JUDGEMENTS, LOG_DIR, TRIAGE_CLASSES, ChatLog, read_jsonl
from mabhas19.api.pilot_metrics import markdown, metrics

Events = list[dict[str, Any]]


def _logs(directory: Path) -> tuple[Events, Events, Events]:
    return (
        read_jsonl(directory / "requests.jsonl"),
        read_jsonl(directory / "feedback.jsonl"),
        read_jsonl(directory / "reviews.jsonl"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report")
    sub.add_parser("pending")
    rv = sub.add_parser("review")
    rv.add_argument("--request", required=True)
    rv.add_argument("--correct", required=True, choices=JUDGEMENTS)
    rv.add_argument("--answerable", required=True, choices=JUDGEMENTS)
    rv.add_argument("--triage", required=True, choices=TRIAGE_CLASSES)
    rv.add_argument("--wrong-value", action="store_true")
    rv.add_argument("--unsupported-number", action="store_true")
    rv.add_argument("--note", default="")
    rv.add_argument("--reviewer", default="")
    args = parser.parse_args(argv)
    requests, feedback, reviews = _logs(args.log_dir)

    if args.command == "review":
        if args.request not in {str(r.get("request_id")) for r in requests}:
            print(f"no logged request {args.request}")
            return 2
        try:
            record = ChatLog(args.log_dir).review(
                {
                    "request_id": args.request,
                    "answer_correct": args.correct,
                    "answerable": args.answerable,
                    "triage": args.triage,
                    "wrong_regulatory_value": args.wrong_value,
                    "unsupported_numeric_claim": args.unsupported_number,
                    "note": args.note,
                },
                reviewer=args.reviewer,
            )
        except ValueError as exc:
            print(f"review not recorded: {exc}")
            return 2
        print(json.dumps(record, ensure_ascii=False))
        return 0

    m = metrics(requests, feedback, reviews)
    if args.command == "pending":
        rated = {str(f.get("request_id")): f.get("rating") for f in feedback}
        for r in requests:
            if r.get("kind") == "request" and r["request_id"] in m["unreviewed"]:
                print(
                    r["request_id"],
                    r.get("participant"),
                    r.get("response_type"),
                    rated.get(r["request_id"], "-"),
                    r.get("question"),
                    sep="\t",
                )
        return 0

    out = args.log_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (out / f"m2_report_{stamp}.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    text = markdown(m)
    (out / f"m2_report_{stamp}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
