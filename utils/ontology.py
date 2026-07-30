from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class MacroFeatureInfo:
    id: int
    key: str
    enabled: bool
    aliases: List[str]


@dataclass(frozen=True)
class MacroFeatureResolution:
    id: int
    key: str
    raw_key: str
    matched_by: str


class OntologyMaps:
    def __init__(self, ontology_name: str, ontology_path: Path, macro_features: List[MacroFeatureInfo]):
        self.ontology_name = ontology_name
        self.ontology_path = ontology_path
        self.macro_features = macro_features
        self.macro_id_by_key: Dict[str, int] = {item.key: item.id for item in macro_features}
        self.macro_key_by_id: Dict[int, str] = {item.id: item.key for item in macro_features}
        self.alias_to_key: Dict[str, str] = {}

        for item in macro_features:
            self.alias_to_key[item.key] = item.key
            for alias in item.aliases:
                self.alias_to_key[alias] = item.key

        if "stock" not in self.macro_id_by_key:
            raise ValueError("Ontology macro_feature layer must contain 'stock'.")

    @property
    def stock_id(self) -> int:
        return self.macro_id_by_key["stock"]

    def resolve_macro_feature(self, raw_key: str) -> MacroFeatureResolution:
        canonical_key = self.alias_to_key.get(raw_key)
        if canonical_key is None:
            raise KeyError(
                f"Macro feature '{raw_key}' is not defined in ontology aliases: {self.ontology_path}"
            )

        matched_by = "key" if canonical_key == raw_key else "alias"
        return MacroFeatureResolution(
            id=self.macro_id_by_key[canonical_key],
            key=canonical_key,
            raw_key=raw_key,
            matched_by=matched_by,
        )

    def export_label_map(self) -> dict:
        return {
            "ontology_name": self.ontology_name,
            "ontology_path": str(self.ontology_path),
            "macro_feature": {
                "id_to_key": {str(k): v for k, v in sorted(self.macro_key_by_id.items())},
                "key_to_id": dict(sorted(self.macro_id_by_key.items(), key=lambda item: item[1])),
                "alias_to_key": dict(sorted(self.alias_to_key.items())),
            },
        }


def load_ontology_maps(ontology_path: str | Path) -> OntologyMaps:
    path = Path(ontology_path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")

    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        return _build_maps_from_dict(data, path)
    except ModuleNotFoundError:
        return _build_maps_from_fallback_yaml(path)


def load_layer_index(ontology_path: str | Path, layer_name: str) -> dict:
    path = Path(ontology_path)
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")

    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        labels = data["layers"][layer_name]["labels"]
        ontology_name = data.get("ontology_name", "unknown_ontology")
        return _build_layer_index(ontology_name, path, layer_name, labels)
    except ModuleNotFoundError:
        return _build_layer_index_from_fallback_yaml(path, layer_name)


def export_layer_label_map(layer_index: dict) -> dict:
    return {
        "ontology_name": layer_index["ontology_name"],
        "ontology_path": layer_index["ontology_path"],
        layer_index["layer_name"]: {
            "id_to_key": dict(sorted(layer_index["id_to_key"].items(), key=lambda item: int(item[0]))),
            "key_to_id": dict(sorted(layer_index["key_to_id"].items(), key=lambda item: item[1])),
            "alias_to_key": dict(sorted(layer_index["alias_to_key"].items())),
        },
    }


def _build_maps_from_dict(data: dict, ontology_path: Path) -> OntologyMaps:
    ontology_name = data.get("ontology_name", "unknown_ontology")
    macro_feature_layer = data["layers"]["macro_feature"]["labels"]

    macro_features = []
    for item in macro_feature_layer:
        macro_features.append(
            MacroFeatureInfo(
                id=int(item["id"]),
                key=str(item["key"]),
                enabled=bool(item.get("enabled", True)),
                aliases=[str(alias) for alias in item.get("current_code_aliases", [])],
            )
        )

    return OntologyMaps(ontology_name, ontology_path, macro_features)


def _build_layer_index(ontology_name: str, ontology_path: Path, layer_name: str, labels: list) -> dict:
    id_to_key = {}
    key_to_id = {}
    alias_to_key = {}
    enabled_keys = []

    for item in labels:
        label_id = int(item["id"])
        key = str(item["key"])
        id_to_key[str(label_id)] = key
        key_to_id[key] = label_id
        alias_to_key[key] = key
        if bool(item.get("enabled", True)):
            enabled_keys.append(key)
        for alias in item.get("current_code_aliases", []):
            alias_to_key[str(alias)] = key

    return {
        "ontology_name": ontology_name,
        "ontology_path": str(ontology_path),
        "layer_name": layer_name,
        "id_to_key": id_to_key,
        "key_to_id": key_to_id,
        "alias_to_key": alias_to_key,
        "enabled_keys": enabled_keys,
    }


def _build_maps_from_fallback_yaml(ontology_path: Path) -> OntologyMaps:
    ontology_name, labels = _extract_layer_labels_from_fallback_yaml(ontology_path, "macro_feature")
    macro_features: List[MacroFeatureInfo] = []

    for item in labels:
        macro_features.append(
            MacroFeatureInfo(
                id=int(item["id"]),
                key=item["key"],
                enabled=bool(item.get("enabled", True)),
                aliases=list(item.get("current_code_aliases", [])),
            )
        )

    return OntologyMaps(ontology_name, ontology_path, macro_features)


def _build_layer_index_from_fallback_yaml(ontology_path: Path, layer_name: str) -> dict:
    ontology_name, labels = _extract_layer_labels_from_fallback_yaml(ontology_path, layer_name)
    return _build_layer_index(ontology_name, ontology_path, layer_name, labels)


def _extract_layer_labels_from_fallback_yaml(ontology_path: Path, layer_name: str) -> tuple[str, list]:
    ontology_name = "unknown_ontology"
    labels = []

    lines = ontology_path.read_text(encoding="utf-8").splitlines()
    in_target_layer = False
    in_labels = False
    in_aliases = False
    current = None

    def flush_current():
        nonlocal current
        if current is None:
            return
        labels.append(current)
        current = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))

        if indent == 0 and stripped.startswith("ontology_name:"):
            ontology_name = stripped.split(":", 1)[1].strip()
            continue

        if indent == 2 and stripped == f"{layer_name}:":
            in_target_layer = True
            in_labels = False
            in_aliases = False
            flush_current()
            continue

        if in_target_layer and indent == 2 and stripped.endswith(":") and stripped != f"{layer_name}:":
            flush_current()
            in_target_layer = False
            in_labels = False
            in_aliases = False
            continue

        if not in_target_layer:
            continue

        if indent == 4 and stripped == "labels:":
            in_labels = True
            in_aliases = False
            continue

        if not in_labels:
            continue

        if indent == 6 and stripped.startswith("- id:"):
            flush_current()
            in_aliases = False
            current = {
                "id": stripped.split(":", 1)[1].strip(),
                "current_code_aliases": [],
                "enabled": True,
            }
            continue

        if current is None:
            continue

        if indent == 8 and stripped.startswith("key:"):
            current["key"] = stripped.split(":", 1)[1].strip()
            in_aliases = False
            continue

        if indent == 8 and stripped.startswith("enabled:"):
            value = stripped.split(":", 1)[1].strip().lower()
            current["enabled"] = value == "true"
            in_aliases = False
            continue

        if indent == 8 and stripped.startswith("current_code_aliases:"):
            aliases_value = stripped.split(":", 1)[1].strip()
            current["current_code_aliases"] = []
            in_aliases = aliases_value == ""
            continue

        if in_aliases and indent == 10 and stripped.startswith("- "):
            current["current_code_aliases"].append(stripped[2:].strip())
            continue

        if indent <= 6:
            in_aliases = False

    flush_current()

    if not labels:
        raise ValueError(f"Failed to parse layer '{layer_name}' labels from ontology: {ontology_path}")

    return ontology_name, labels
