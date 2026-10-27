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
    assert live["code"]["router_version"] == "v4.1"
    assert live["code"]["query_plan_version"] == "plan-4"
    assert live["code"]["coverage_version"] == "coverage-2"
    assert live["code"]["coverage_ledger_schema"] == "coverage-ledger-2"
    assert live["code"]["coverage_invariants"] == [f"I{i}" for i in range(1, 11)]
    assert live["code"]["numeric_parser_version"] == "numeric-2"
    assert live["code"]["entity_resolver_version"] == "source-structure-1"
    assert live["code"]["constraint_parser_version"] == "attested-1"
    assert live["code"]["unit_model_version"] == "units-compositional-1"
    assert live["code"]["polarity_version"] == "polarity-corrections-1"
    assert live["code"]["coordination_version"] == "generic-coordination-1"
    assert len(live["code"]["identity_snapshot_sha256"]) == 64
    assert live["evaluator"]["version"] == "corrected-6"
    assert live["evaluator"]["final_validation_version"] == "final-validation-1"
    assert live["evaluator"]["final_validated_sets"] == ["final5", "final6", "final7"]
    scope = live["curated"]["table_scope"]
    assert len(scope) == 14
    assert not [ds for ds, row in scope.items() if "undeclared" in row.values()]
    assert scope["m19-ed5-1404-main.t19-5-1"]["rating"] == {
        "fixed": ["D"],
        "provenance": ["main-p083-b010", "main-p077-b007", "main-p077-b008"],
    }
    assert scope["m19-ed5-1404-app03.max_energy_use_intensity"]["rating"] == "printed"
    assert live["safety"] == {
        "wrong_regulatory_values_expected": 0,
        "wrong_extra_rows_expected": 0,
        "dropped_constraints_expected": 0,
        "failed_certifications_expected": 0,
        "foreign_table_rows_expected": 0,
        "dimension_leaks_expected": 0,
        "identity_substitutions_expected": 0,
        "refusal_bypassed_expected": 0,
    }
    modules = live["code"]["modules_sha256"]
    for m in (
        "constraints",
        "plan",
        "planner",
        "executor",
        "numeric",
        "schema",
        "admission",
        "certification",
        "coverage",
        "units",
        "table_scope",
        "safety_metrics",
        "identity",
        "identity_snapshot",
        "polarity",
        "operations",
        "ratings",
        "final_validation",
    ):
        assert f"src/mabhas19/retrieval/{m}.py" in modules
    assert set(live["scope"]) == {
        "scope_map_sha256",
        "external_domains_sha256",
        "policy_ambiguities_sha256",
    }
    assert len(live["curated"]["structured_schemas_sha256"]) == 64


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


def test_only_the_frozen_set_may_change_after_the_candidate_commit(tmp_path: Path) -> None:
    live = live_candidate()
    path = tmp_path / "final4_candidate.yaml"
    root = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git's empty tree is never an ancestor
    doc = {"candidate_commit": root, "allowed_changes_after": [], "candidate": live}
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert any("not an ancestor" in p for p in check_manifest(path))


def test_final7_is_the_only_writable_manifest() -> None:
    from mabhas19.retrieval.candidate import CURRENT, LABELS, MANIFESTS, SET_FILES  # noqa: PLC0415

    assert CURRENT == "final7"
    assert LABELS[CURRENT] == "Final-7"
    assert set(MANIFESTS) == {"final2", "final3", "final4", "final5", "final6", "final7"}
    assert all("final7" in f for f in SET_FILES)
