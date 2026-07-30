from pathlib import Path


ROOT = Path(__file__).resolve().parent
runner = ROOT / "run_p4_api_r1c_structured_validation.py"
text = runner.read_text(encoding="utf-8")
needle = '''def threshold_acceptance(val_metrics: Mapping[str, Any]) -> dict[str, Any]:
'''
addition = '''def apply_outcome_schema_metrics(
    metrics: dict[str, Any], outcome_rows: list[Mapping[str, Any]]
) -> None:
    total = len(outcome_rows)
    if total != int(metrics["record_count"]):
        raise RuntimeError("E_SCHEMA_METRIC_ROW_COUNT")
    metrics["language_cae_ir"]["schema_valid_rate_first_round"] = (
        sum(bool(row["schema_valid_first_round"]) for row in outcome_rows) / total
    )
    metrics["language_cae_ir"]["schema_valid_rate_after_retry"] = (
        sum(bool(row["schema_valid_after_retry"]) for row in outcome_rows) / total
    )
    metrics["runtime"]["retry_rate"] = (
        sum(int(row["retry_count"]) for row in outcome_rows) / total
    )


def threshold_acceptance(val_metrics: Mapping[str, Any]) -> dict[str, Any]:
'''
if text.count(needle) != 1:
    raise RuntimeError("E_PATCH_FUNCTION_ANCHOR")
text = text.replace(needle, addition)
needle = '''        metrics = accumulators[split].finalize(unique_source_graphs=len(graph_sets[split]))
        metrics["runtime_config"] = config.public_metadata()
'''
replacement = '''        metrics = accumulators[split].finalize(unique_source_graphs=len(graph_sets[split]))
        split_outcomes = [row for row in client.audit_rows if row["split"] == split]
        apply_outcome_schema_metrics(metrics, split_outcomes)
        metrics["runtime_config"] = config.public_metadata()
'''
if text.count(needle) != 1:
    raise RuntimeError("E_PATCH_SPLIT_ANCHOR")
text = text.replace(needle, replacement)
needle = '''    combined_metrics = combined.finalize(unique_source_graphs=len(graph_sets["combined"]))
    combined_metrics["runtime_config"] = config.public_metadata()
'''
replacement = '''    combined_metrics = combined.finalize(unique_source_graphs=len(graph_sets["combined"]))
    apply_outcome_schema_metrics(combined_metrics, client.audit_rows)
    combined_metrics["runtime_config"] = config.public_metadata()
'''
if text.count(needle) != 1:
    raise RuntimeError("E_PATCH_COMBINED_ANCHOR")
runner.write_text(text, encoding="utf-8", newline="\n")

test_path = ROOT / "tests" / "test_p4_api_r1c_runner.py"
test_text = test_path.read_text(encoding="utf-8")
needle = '''

if __name__ == "__main__":
'''
addition = '''
    def test_outcome_schema_metrics_override_placeholder_validity(self):
        metrics = {
            "record_count": 2,
            "language_cae_ir": {
                "schema_valid_rate_first_round": 1.0,
                "schema_valid_rate_after_retry": 1.0,
            },
            "runtime": {"retry_rate": 0.0},
        }
        rows = [
            {
                "schema_valid_first_round": False,
                "schema_valid_after_retry": False,
                "retry_count": 1,
            },
            {
                "schema_valid_first_round": True,
                "schema_valid_after_retry": True,
                "retry_count": 0,
            },
        ]
        r1c.apply_outcome_schema_metrics(metrics, rows)
        self.assertEqual(metrics["language_cae_ir"]["schema_valid_rate_first_round"], 0.5)
        self.assertEqual(metrics["language_cae_ir"]["schema_valid_rate_after_retry"], 0.5)
        self.assertEqual(metrics["runtime"]["retry_rate"], 0.5)


if __name__ == "__main__":
'''
if test_text.count(needle) != 1:
    raise RuntimeError("E_PATCH_TEST_ANCHOR")
test_text = test_text.replace(needle, addition)
test_path.write_text(test_text, encoding="utf-8", newline="\n")
