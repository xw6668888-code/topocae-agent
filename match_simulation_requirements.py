from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.requirement_matching import match_requirements


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match Chinese static-structural preprocessing requirements to CAE selectable face regions."
    )
    parser.add_argument("--requirements-path", type=Path, required=True, help="JSON file containing sample requirements.")
    parser.add_argument("--output-path", type=Path, required=True, help="Path for matched region output JSON.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    config_relative = (base_dir / path).resolve()
    if config_relative.exists():
        return config_relative
    cwd_relative = (Path.cwd() / path).resolve()
    if cwd_relative.exists():
        return cwd_relative
    return config_relative


def run_matching(requirements_data: dict, requirements_dir: Path) -> dict:
    sample_results = []
    for sample in requirements_data.get("samples", []):
        feature_objects_path = resolve_path(sample["feature_objects_path"], requirements_dir)
        feature_objects_data = load_json(feature_objects_path)
        matched = match_requirements(sample.get("requirements", []), feature_objects_data)
        sample_results.append({
            "sample_id": sample.get("sample_id", feature_objects_data.get("sample_id")),
            "step_path": sample.get("step_path", feature_objects_data.get("source_step_path")),
            "feature_objects_path": str(feature_objects_path),
            **matched,
        })

    return {
        "matching_method": "deterministic_chinese_keyword_to_selectable_region_v1",
        "sample_count": len(sample_results),
        "samples": sample_results,
    }


def main() -> None:
    args = parse_args()
    requirements_data = load_json(args.requirements_path)
    output = run_matching(requirements_data, args.requirements_path.resolve().parent)
    write_json(args.output_path, output)
    print(f"requirement matches written: {args.output_path}")


if __name__ == "__main__":
    main()
