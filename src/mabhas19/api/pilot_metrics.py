"""Closed-pilot (M2) metrics from the logs alone: no engine call, no model, nothing rewritten.

Inputs are the three append-only logs of :mod:`mabhas19.api.chatlog`. For each request the
latest feedback and the latest review count. Safety is reported first: wrong regulatory values
and unsupported numeric claims found by manual review, and provenance violations recorded by
the composer at answer time (each was already shown to the user as a refusal).
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from mabhas19.api.chatlog import TRIAGE_CLASSES

ANSWERED = ("structured_value", "prose")
UNANSWERED = ("clarification", "refusal", "out_of_scope")
RESPONSE_TYPES = (*ANSWERED, *UNANSWERED)


def _latest(events: list[dict[str, Any]], known: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in sorted(events, key=lambda x: str(x.get("timestamp", ""))):
        rid = str(e.get("request_id", ""))
        if rid in known:
            out[rid] = e
    return out


def _share(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def metrics(
    requests: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    asked = [r for r in requests if r.get("kind") == "request"]
    errors = [r for r in requests if r.get("kind") == "error"]
    by_id = {str(r["request_id"]): r for r in asked}
    known = set(by_id)
    latest_fb = _latest(feedback, known)
    latest_rv = _latest(reviews, known)
    n = len(asked)

    per_participant = Counter(str(r.get("participant") or "anonymous") for r in asked)
    types = Counter(str(r.get("response_type")) for r in asked)
    ratings = Counter(str(f.get("rating")) for f in latest_fb.values())
    notes = sum(bool(str(f.get("note") or "").strip()) for f in latest_fb.values())

    reviewed = {rid: rv for rid, rv in latest_rv.items()}
    answered = {rid: rv for rid, rv in reviewed.items() if by_id[rid]["response_type"] in ANSWERED}
    unanswered = {rid: rv for rid, rv in reviewed.items() if rid not in answered}
    correct = Counter(str(rv.get("answer_correct")) for rv in answered.values())
    answerable = Counter(str(rv.get("answerable")) for rv in unanswered.values())
    triage = Counter(str(rv.get("triage")) for rv in reviewed.values())

    wrong = sorted(rid for rid, rv in reviewed.items() if rv.get("wrong_regulatory_value"))
    unsupported = sorted(rid for rid, rv in reviewed.items() if rv.get("unsupported_numeric_claim"))
    violations = sorted(rid for rid, r in by_id.items() if r.get("provenance_violations"))
    latencies = sorted(float(r.get("latency_ms") or 0) for r in asked)

    # clarification flows: a follow-up names the request it continues
    asked_back = {rid for rid, r in by_id.items() if r.get("response_type") == "clarification"}
    follow_ups = [r for r in asked if (r.get("follow_up") or {}).get("parent_request_id")]
    continued = {str(r["follow_up"]["parent_request_id"]) for r in follow_ups} & asked_back
    reached = [r for r in follow_ups if r.get("response_type") in ANSWERED]

    return {
        "safety": {
            "wrong_regulatory_values": len(wrong),
            "wrong_regulatory_value_requests": wrong,
            "unsupported_numeric_claims": len(unsupported),
            "unsupported_numeric_claim_requests": unsupported,
            "provenance_violations": len(violations),
            "provenance_violation_requests": violations,
            "hard_criterion": (
                "investigate now"
                if wrong
                else ("no wrong value in reviewed answers" if reviewed else "nothing reviewed yet")
            ),
            "reviewed_share": _share(len(reviewed), n),
        },
        "usage": {
            "participants": len(per_participant),
            "questions": n,
            "questions_per_participant": dict(per_participant.most_common()),
            "errors": len(errors),
        },
        "answer_rate": {
            t: {"count": types.get(t, 0), "share": _share(types.get(t, 0), n)}
            for t in RESPONSE_TYPES
        },
        "clarifications": {
            "asked": len(asked_back),
            "continued_by_user": len(continued),
            "follow_ups": len(follow_ups),
            "follow_ups_answered": len(reached),
        },
        "feedback": {
            "requests_with_feedback": len(latest_fb),
            "feedback_rate": _share(len(latest_fb), n),
            "up": ratings.get("up", 0),
            "down": ratings.get("down", 0),
            "negative_rate": _share(ratings.get("down", 0), len(latest_fb)),
            "with_note": notes,
        },
        "reviewed_quality": {
            "reviewed": len(reviewed),
            "answered_reviewed": len(answered),
            "correct": correct.get("yes", 0),
            "incorrect": correct.get("no", 0),
            "uncertain": correct.get("uncertain", 0),
            "unanswered_reviewed": len(unanswered),
            "unanswered_but_answerable": answerable.get("yes", 0),
            "genuinely_unsupported": answerable.get("no", 0),
            "unanswered_uncertain": answerable.get("uncertain", 0),
        },
        "failure_taxonomy": {c: triage.get(c, 0) for c in TRIAGE_CLASSES},
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p90": latencies[int(0.9 * (len(latencies) - 1))] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "unreviewed": sorted(known - set(reviewed)),
    }


def markdown(m: dict[str, Any]) -> str:
    s, u, f, q = m["safety"], m["usage"], m["feedback"], m["reviewed_quality"]
    lines = [
        "# M2 closed-pilot report",
        "",
        "## Safety",
        f"- wrong regulatory values (reviewed): **{s['wrong_regulatory_values']}** "
        f"{s['wrong_regulatory_value_requests'] or ''}",
        f"- unsupported numeric claims (reviewed): **{s['unsupported_numeric_claims']}** "
        f"{s['unsupported_numeric_claim_requests'] or ''}",
        f"- provenance violations (refused at answer time): **{s['provenance_violations']}**",
        f"- hard criterion: {s['hard_criterion']} (reviewed share {s['reviewed_share']})",
        "",
        "## Usage",
        f"- participants {u['participants']}, questions {u['questions']}, errors {u['errors']}",
        f"- per participant: {u['questions_per_participant']}",
        "",
        "## Answer rate",
        *(f"- {t}: {v['count']} ({v['share']})" for t, v in m["answer_rate"].items()),
        "",
        "## Clarifications",
        f"- asked {m['clarifications']['asked']}, continued by the user "
        f"{m['clarifications']['continued_by_user']}; follow-ups "
        f"{m['clarifications']['follow_ups']}, of which answered "
        f"{m['clarifications']['follow_ups_answered']}",
        "",
        "## Feedback",
        f"- with feedback {f['requests_with_feedback']} (rate {f['feedback_rate']}); "
        f"up {f['up']}, down {f['down']} (negative rate {f['negative_rate']}); "
        f"notes {f['with_note']}",
        "",
        "## Reviewed quality",
        f"- reviewed {q['reviewed']}: answered {q['answered_reviewed']} (correct {q['correct']}, "
        f"incorrect {q['incorrect']}, uncertain {q['uncertain']}); unanswered "
        f"{q['unanswered_reviewed']} (answerable {q['unanswered_but_answerable']}, genuinely "
        f"unsupported {q['genuinely_unsupported']}, uncertain {q['unanswered_uncertain']})",
        "",
        "## Failure taxonomy (reviewed)",
        *(f"- {c}: {v}" for c, v in m["failure_taxonomy"].items()),
        "",
        f"## Latency (ms): {m['latency_ms']}",
        "",
        f"Unreviewed requests: {len(m['unreviewed'])}",
    ]
    return "\n".join(lines) + "\n"
