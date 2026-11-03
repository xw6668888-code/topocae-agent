"""Frozen recognition and source-geometry measurements, without truth or API calls."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PILOT = Path(__file__).resolve().parents[1] / "aei_two_case_pilot"
sys.path.insert(0, str(PILOT))
from build_cases import ROOT, load_case, properties, sha, write_json, xyz
from feature_service import FROZEN_CHECKPOINT, recognize
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Extend.TopologyUtils import TopologyExplorer

OUT = ROOT / "reports/aei_case_suite_v2"


def measured(face, number):
    surface = BRepAdaptor_Surface(face)
    area, center = properties(face)
    row = {"face_number": number, "area_mm2": area, "center_mm": center, "type": str(surface.GetType())}
    from OCC.Core.BRep import BRep_Tool
    row["vertices_mm"] = [xyz(BRep_Tool.Pnt(v)) for v in TopologyExplorer(face).vertices()]
    if surface.GetType() == GeomAbs_Plane:
        normal = xyz(surface.Plane().Axis().Direction())
        if face.Orientation() == TopAbs_REVERSED:
            normal = [-v for v in normal]
        row["normal"] = normal
    if surface.GetType() == GeomAbs_Cylinder:
        cyl = surface.Cylinder()
        row.update(radius_mm=cyl.Radius(), axis=xyz(cyl.Axis().Direction()), origin_mm=xyz(cyl.Axis().Location()),
                   v_range=[surface.FirstVParameter(), surface.LastVParameter()],
                   u_range=[surface.FirstUParameter(), surface.LastUParameter()])
    return row


def main():
    results = []
    for source in sorted((ROOT / "test_part_0901").glob("*.stp")):
        prediction_path = OUT / "recognition" / (source.stem + "_recognition.json")
        if not prediction_path.exists():
            prediction_path = recognize(source, OUT / "recognition")
            log = prediction_path.parent / "inference.log"
            prediction_path.with_suffix(".log").write_bytes(log.read_bytes())
        case = load_case(source=source, prediction_path=prediction_path)
        _, shape, faces, prediction, _ = case
        if prediction["model"]["checkpoint_sha256"].lower() != FROZEN_CHECKPOINT:
            raise RuntimeError("Frozen checkpoint mismatch")
        records = [measured(face, i + 1) for i, face in enumerate(faces)]
        for record, pred in zip(records, prediction["faces"]):
            record.update(predicted_class=pred["class"], semantic_label=pred["semantic_label"])
        row = {"part": source.name, "source_sha256": sha(source), "prediction_sha256": sha(prediction_path),
               "truth_used": False, "model": prediction["model"], "volume_mm3": properties(shape, True)[0],
               "center_of_mass_mm": properties(shape, True)[1],
               "objects": prediction["feature_objects"], "faces": records}
        write_json(OUT / "geometry" / (source.stem + ".json"), row)
        results.append({"part": source.name, "faces": len(faces), "objects": len(prediction["feature_objects"]),
                        "classes": {name: sum(o["class"] == name for o in prediction["feature_objects"])
                                    for name in sorted({o["class"] for o in prediction["feature_objects"]})}})
        write_json(OUT / "recognition_inventory.json", results)
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
