"""Final candidate manifests: every pinned value is recomputed and drift is reported."""

from __future__ import annotations

from pathlib import Path

import yaml

from mabhas19.retrieval.candidate import check_manifest, live_candidate


def test_live_candidate_pins_the_architecture() -> None:
    live = live_candidate()
    assert live["bm25"] == {"k1": 1.5, "b": 0.75}
    assert live["fusion"]["rrf_k"] == 60
    assert live["chunks"]["count"] == 420
    assert live["embedding"]["model_id"] == "intfloat/multilingual-e5-small"
    assert live["embedding"]["dimensions"] == 384
    assert live["code"]["router_version"] == "v3.2"
    assert live["code"]["constraint_parser_version"] == "ledger-1"
    assert live["evaluator"]["version"] == "corrected-2"
    assert live["safety"] == {"wrong_regulatory_values_expected": 0}
    assert "src/mabhas19/retrieval/constraints.py" in live["code"]["modules_sha256"]


def test_any_pinned_difference_is_reported(tmp_path: Path) -> None:
    live = live_candidate()
    live["fusion"]["rrf_k"] = 61
    live["scope"]["scope_map_sha256"] = "0" * 64
    live["safety"]["wrong_regulatory_values_expected"] = 1
    path = tmp_path / "final2_candidate.yaml"
    path.write_text(yaml.safe_dump({"candidate": live}), encoding="utf-8")
    problems = check_manifest(path)
    assert any(p.startswith("fusion.rrf_k") for p in problems)
    assert any(p.startswith("scope.scope_map_sha256") for p in problems)
    assert any(p.startswith("safety.wrong_regulatory_values_expected") for p in problems)


def test_missing_manifest_refuses(tmp_path: Path) -> None:
    assert check_manifest(tmp_path / "absent.yaml") == ["absent.yaml is missing"]
