import json

from scripts.summarize_selector_ablation import main


def _trace(path, disabled, best_ms, mapping):
    rows = [
        {
            "schema": "cubutterfly-plan-search-trace-v1",
            "search_id": 1,
            "search": "butterfly",
            "phase": "inventory",
            "candidate_count": 4,
            "restricted_candidate_count": 3 if disabled else 4,
            "disabled_candidate_count": 1 if disabled else 0,
            "scheduled_candidate_count": 3 if disabled else 4,
            "search_budget": 0,
            "disabled_mechanisms": disabled,
        },
        {
            "schema": "cubutterfly-plan-search-trace-v1",
            "search_id": 1,
            "search": "butterfly",
            "phase": "candidate_result",
            "outcome": "accepted",
            "kernel_ms": best_ms,
            "mapping": mapping,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_selector_ablation_summary(tmp_path, monkeypatch):
    unrestricted = tmp_path / "unrestricted.jsonl"
    restricted = tmp_path / "restricted.jsonl"
    _trace(unrestricted, [], 1.0, {"kind": "a"})
    _trace(restricted, ["stage-overlap"], 1.25, {"kind": "b"})
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_selector_ablation.py", str(unrestricted), str(restricted), "--output", str(output)],
    )
    main()
    result = json.loads(output.read_text())
    pair = result["paired"][0]
    assert pair["selected_mapping_same"] is False
    assert pair["restricted_regret_vs_unrestricted_best"] == 0.25
    assert result["restricted"]["searches"][0]["disabled_mechanisms"] == ["stage-overlap"]
