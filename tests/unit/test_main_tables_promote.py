from __future__ import annotations

import json
from typing import Any

import yaml

from mabhas19.tables.app02 import SOURCE_ISSUES_BY_ROW
from mabhas19.tables.app02_promote import APP02_SI_0009
from mabhas19.tables.main_extract import ISSUE_ISFAHAN, ISSUE_KERMAN, final_flags
from mabhas19.tables.main_tables_promote import (
    NEW_ISSUES,
    app02_with_backlink,
    issues_with_new,
    validate_isfahan_backlink,
)

EXISTING = """schema_version: 1
issues:
  - id: SI-0001
    status: unresolved
    title: "x"
"""


def test_new_issue_ids_do_not_reuse_existing_ones() -> None:
    assert (ISSUE_ISFAHAN, ISSUE_KERMAN) == ("SI-0009", "SI-0010")


def test_issues_are_appended_unresolved_and_idempotent() -> None:
    text = issues_with_new(EXISTING)
    assert text.startswith(EXISTING.rstrip("\n"))
    issues = {i["id"]: i for i in yaml.safe_load(text)["issues"]}
    assert issues[ISSUE_ISFAHAN]["status"] == issues[ISSUE_KERMAN]["status"] == "unresolved"
    assert "typo" not in issues[ISSUE_KERMAN]["description"].lower()
    assert "۴۸۸۴۱۰" in issues[ISSUE_KERMAN]["description"]
    assert issues_with_new(text) == text
    assert NEW_ISSUES.count("former_candidate_id") == 2


def test_percent_order_is_kept_and_flagged_as_accepted_not_open() -> None:
    audit = {"flags": [], "value_class": "label", "text_raw": "%20 از کمتر"}
    f = final_flags(audit, "کمتر از %۲۰")
    assert f["accepted_flags"] == ["percent_sign_order_bidi"]
    assert f["open_flags"] == []


def test_corrected_brackets_resolve_the_audit_flag() -> None:
    audit = {"flags": ["mirrored_brackets", "multi_line"], "value_class": "label", "text_raw": "x"}
    f = final_flags(audit, "حداكثر دما (درجه سانتیگراد)")
    assert f["resolved_flags"] == ["mirrored_brackets"]
    assert f["flags"] == ["multi_line"]
    assert f["open_flags"] == []


def _app02(refs: list[str], code: str = "4B") -> str:
    rec = {"id": "app02-p002-r07", "climate_class": {"code": code}, "source_issue_refs": refs,
           "verification": {"status": "machine_corroborated"}}  # fmt: skip
    other = {"id": "app02-p002-r01", "climate_class": {"code": "0B"}, "source_issue_refs": []}
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in (other, rec)) + "\n"


def test_app02_backlink_adds_only_the_issue_reference() -> None:
    before = _app02([])
    after, meta = app02_with_backlink(before, "source_issues:\n- 'SI-0001: x'\n")
    assert '"source_issue_refs": ["SI-0009"]' in after
    assert '"code": "4B"' in after
    assert APP02_SI_0009 in yaml.safe_load(meta)["source_issues"]
    assert app02_with_backlink(after, meta) == (after, meta)
    main = {"city": {"normalized": "اصفهان"}, "climate_class": {"code": "3B"},
            "source_issue_refs": ["SI-0009"]}  # fmt: skip
    built: dict[str, Any] = {"19-3-2": ([], [main], {})}
    issues = issues_with_new(EXISTING)
    assert validate_isfahan_backlink(built, before, after, issues) == []
    assert validate_isfahan_backlink(built, before, _app02(["SI-0009"], "3B"), issues)


def test_app02_rebuild_keeps_the_backlink() -> None:
    assert SOURCE_ISSUES_BY_ROW == {"app02-p002-r07": ["SI-0009"]}
