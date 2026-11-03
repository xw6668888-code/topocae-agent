"""Compile two approved engineering operations from frozen P1 predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCC.Core.TopAbs import TopAbs_IN
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Pnt2d
from OCC.Extend.TopologyUtils import TopologyExplorer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/aei_two_case_pilot_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def xyz(value):
    return [float(value.X()), float(value.Y()), float(value.Z())]


def properties(shape, volume=False):
    props = GProp_GProps()
    if volume:
        brepgprop.VolumeProperties(shape, props)
    else:
        brepgprop.SurfaceProperties(shape, props)
    return float(props.Mass()), xyz(props.CentreOfMass())


def read_step(path):
    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError("STEP read failed: " + str(path))
    reader.TransferRoots()
    shape = reader.OneShape()
    if not BRepCheck_Analyzer(shape).IsValid() or len(list(TopologyExplorer(shape).solids())) != 1:
        raise RuntimeError("Expected one valid solid")
    return shape


def load_case(stem=None, *, source=None, prediction_path=None):
    source = Path(source) if source is not None else ROOT / "test_part_0901" / (stem + ".stp")
    prediction_path = Path(prediction_path) if prediction_path is not None else OUT / "recognition" / (stem + "_recognition.json")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if prediction["truth_used"] or prediction["input"]["sha256"].lower() != sha(source):
        raise RuntimeError("Prediction provenance does not match the STEP")
    shape = read_step(source)
    return source, shape, list(TopologyExplorer(shape).faces()), prediction, prediction_path


def select(prediction, class_name):
    objects = [obj for obj in prediction["feature_objects"] if obj["class"] == class_name]
    if not objects or any(not obj["topology_valid"] for obj in objects):
        raise RuntimeError("Missing or invalid predicted objects: " + class_name)
    return objects


def face_record(face, number):
    surface = BRepAdaptor_Surface(face)
    area, center = properties(face)
    u0, u1 = surface.FirstUParameter(), surface.LastUParameter()
    v0, v1 = surface.FirstVParameter(), surface.LastVParameter()
    candidates = []
    for i in range(1, 12):
        for j in range(1, 12):
            u, v = u0 + (u1 - u0) * i / 12, v0 + (v1 - v0) * j / 12
            if BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), 1e-7).State() == TopAbs_IN:
                candidates.append(xyz(surface.Value(u, v)))
    if len(candidates) < 3:
        raise RuntimeError("Insufficient interior samples for face " + str(number))
    # Spread interior witnesses across the trimmed face, away from shared edges.
    chosen = [candidates[len(candidates) // 2]]
    for _ in range(2):
        chosen.append(max(candidates, key=lambda p: min(math.dist(p, q) for q in chosen)))
    return {"occ_face_number": number, "area_mm2": area, "center_mm": center,
            "interior_points_mm": chosen, "surface_type": str(surface.GetType())}


def build_cae(*, case=None, output=None, fixed=None, loaded=None, material=None,
              mesh_mm=2.0, force_N=100.0, force_basis="per_instance", direction=(0., -1., 0.)):
    source, shape, faces, prediction, prediction_path = case or load_case("Angle Bracket")
    output = Path(output) if output is not None else OUT / "cae"
    fixed = select(prediction, "counterbore_hole") if fixed is None else fixed
    loaded = select(prediction, "obround_hole") if loaded is None else loaded
    if not fixed or not loaded:
        raise ValueError("Empty fixed or loaded object selection")
    if force_basis not in {"per_instance", "total"} or force_N <= 0 or mesh_mm <= 0:
        raise ValueError("Invalid load basis, load magnitude, or mesh size")
    norm = math.sqrt(sum(v * v for v in direction))
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError("Invalid force direction")
    direction = [v / norm for v in direction]
    areas = [sum(properties(faces[n - 1])[0] for n in obj["original_face_numbers"]) for obj in loaded]
    forces = [force_N if force_basis == "per_instance" else force_N * area / sum(areas) for area in areas]
    fixed_ids = {n for obj in fixed for n in obj["original_face_numbers"]}
    load_ids = {n for obj in loaded for n in obj["original_face_numbers"]}
    if fixed_ids & load_ids:
        raise RuntimeError("Overlapping fixed and loaded surfaces")
    intent = {
        "source_step": str(source), "source_sha256": sha(source),
        "script_directory": str(output.resolve()),
        "prediction_sha256": sha(prediction_path), "model": prediction["model"],
        "truth_used": False, "units": "mm, N, MPa, tonne, s",
        "analysis": "linear_elastic_static", "global_mesh_mm": mesh_mm,
        "material": material or {"name": "Steel45", "youngs_modulus_mpa": 210000.0,
                     "poisson_ratio": 0.30, "density_tonne_mm3": 7.85e-9},
        "fixed_scope": "all faces of every predicted counterbore, including the shoulder",
        "load_scope": "uniform directional traction on all inner walls of selected obround holes",
        "force_magnitude_N": force_N, "force_basis": force_basis,
        "force_direction": direction, "object_forces_N": forces,
        "fixed_objects": fixed, "loaded_objects": loaded,
        "expected_force_N": [v * sum(forces) for v in direction],
        "source_volume_mm3": properties(shape, volume=True)[0],
        "source_faces": [face_record(face, i + 1) for i, face in enumerate(faces)],
        "mesh_force_balance_relative_tolerance": 0.005,
    }
    write_json(output / "operation_plan.json", intent)
    template = Path(__file__).with_name("abaqus_angle_bracket.template.py").read_text(encoding="utf-8")
    embedded = repr(json.dumps(intent, ensure_ascii=True, separators=(",", ":")))
    generated = template.replace("__EMBEDDED_PLAN_JSON__", embedded)
    script = output / "Angle_Bracket_Abaqus.py"
    script.write_text(generated, encoding="ascii")
    print("CAE script: " + str(script))
    return script


def boolean(kind, first, second):
    op = kind(first, second)
    op.SetFuzzyValue(1e-7)
    op.Build()
    if not op.IsDone():
        raise RuntimeError("Boolean operation failed")
    result = op.Shape()
    if not BRepCheck_Analyzer(result).IsValid():
        raise RuntimeError("Boolean produced an invalid B-Rep")
    return result


def clean(shape):
    builder = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    builder.Build()
    return builder.Shape()


def hole_geometry(obj, faces):
    numbered = [(n, faces[n - 1]) for n in obj["original_face_numbers"]]
    walls = [(n, BRepAdaptor_Surface(f)) for n, f in numbered
             if BRepAdaptor_Surface(f).GetType() == GeomAbs_Cylinder]
    bottoms = [(n, f) for n, f in numbered if BRepAdaptor_Surface(f).GetType() == GeomAbs_Plane]
    if len(numbered) != 2 or len(walls) != 1 or len(bottoms) != 1:
        raise RuntimeError("This pilot supports a full cylindrical blind wall with a planar bottom")
    number, wall = walls[0]
    cylinder = wall.Cylinder()
    radius = float(cylinder.Radius())
    depth = wall.LastVParameter() - wall.FirstVParameter()
    area, bottom = properties(bottoms[0][1])
    if abs(wall.LastUParameter() - wall.FirstUParameter() - 2 * math.pi) > 1e-6:
        raise RuntimeError("Partial/intersecting cylindrical wall is unsupported")
    if abs(area - math.pi * radius**2) > 1e-5:
        raise RuntimeError("Bottom is not a complete circular disk")
    origin, axis = xyz(cylinder.Axis().Location()), xyz(cylinder.Axis().Direction())
    ends = [[origin[k] + axis[k] * v for k in range(3)]
            for v in (wall.FirstVParameter(), wall.LastVParameter())]
    bottom_end = min(ends, key=lambda p: math.dist(p, bottom))
    opening = max(ends, key=lambda p: math.dist(p, bottom))
    if math.dist(bottom_end, bottom) > 1e-5 or depth <= 0:
        raise RuntimeError("Cannot establish blind-hole depth and direction")
    direction = [(bottom[k] - opening[k]) / depth for k in range(3)]
    return {"object_index": obj["object_index"], "original_face_numbers": obj["original_face_numbers"],
            "wall_face_number": number, "radius_before_mm": radius, "depth_mm": depth,
            "opening_center_mm": opening, "bottom_center_mm": bottom, "inward_axis": direction}


def cylinder_for(hole, radius):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*hole["opening_center_mm"]),
                                         gp_Dir(*hole["inward_axis"])), radius, hole["depth_mm"]).Shape()


def build_cad(*, case=None, output=None, selected=None, old_radius=4.0, new_radius=3.0):
    source, original, faces, prediction, prediction_path = case or load_case("Bearing Housing")
    output = Path(output) if output is not None else OUT / "cad"
    if not (math.isfinite(old_radius) and math.isfinite(new_radius) and 0 < new_radius < old_radius):
        raise ValueError("This operator requires a positive radius reduction")
    blind = select(prediction, "blind_hole_round") if selected is None else selected
    holes = [hole_geometry(obj, faces) for obj in blind
             if any(abs(r - old_radius) < 1e-6 for r in obj["geometry_summary"]["positive_radii"])]
    if not holes or (selected is not None and len(holes) != len(selected)):
        raise RuntimeError("No unambiguous matching blind holes")
    if any(abs(h["radius_before_mm"] - old_radius) > 1e-6 for h in holes):
        raise RuntimeError("Measured target radius mismatch")
    changed = original
    rings = []
    for hole in holes:
        plug, bore = cylinder_for(hole, old_radius), cylinder_for(hole, new_radius)
        if abs(properties(boolean(BRepAlgoAPI_Common, original, plug), True)[0]) > 1e-5:
            raise RuntimeError("Original hole cylinder intersects existing material")
        ring = boolean(BRepAlgoAPI_Cut, plug, bore)
        rings.append(ring)
        changed = clean(boolean(BRepAlgoAPI_Fuse, changed, ring))
        if len(list(TopologyExplorer(changed).solids())) != 1:
            raise RuntimeError("Hole edit split the solid")
        hole["radius_after_mm"] = new_radius
    destination = output / (source.stem + "_r" + format(new_radius, "g") + ".stp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    if writer.Transfer(changed, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError("STEP transfer failed")
    if writer.Write(str(destination)) != IFSelect_RetDone:
        raise RuntimeError("STEP write failed")
    reread = read_step(destination)
    before_volume = properties(original, True)[0]
    after_volume = properties(reread, True)[0]
    expected = sum(math.pi * (old_radius**2 - new_radius**2) * h["depth_mm"] for h in holes)
    removed = abs(properties(boolean(BRepAlgoAPI_Cut, original, reread), True)[0])
    outside = boolean(BRepAlgoAPI_Cut, reread, original)
    for ring in rings:
        outside = boolean(BRepAlgoAPI_Cut, outside, ring)
    outside_volume = abs(properties(outside, True)[0])
    target_checks = []
    new_faces = list(TopologyExplorer(reread).faces())
    for hole in holes:
        matches = []
        bottom_matches = []
        center = [(a + b) / 2 for a, b in zip(hole["opening_center_mm"], hole["bottom_center_mm"])]
        for n, face in enumerate(new_faces, 1):
            surface = BRepAdaptor_Surface(face)
            area, com = properties(face)
            if surface.GetType() == GeomAbs_Cylinder and math.dist(com, center) < 1e-5:
                matches.append({"new_face_number": n, "radius_mm": float(surface.Cylinder().Radius()),
                                "depth_mm": float(surface.LastVParameter() - surface.FirstVParameter())})
            if surface.GetType() == GeomAbs_Plane and math.dist(com, hole["bottom_center_mm"]) < 1e-5:
                bottom_matches.append({"new_face_number": n, "area_mm2": area})
        ok = (len(matches) == 1 and abs(matches[0]["radius_mm"] - new_radius) < 1e-6
              and abs(matches[0]["depth_mm"] - hole["depth_mm"]) < 1e-6
              and len(bottom_matches) == 1 and abs(bottom_matches[0]["area_mm2"] - math.pi * new_radius**2) < 1e-5)
        target_checks.append({"object_index": hole["object_index"], "pass": ok,
                              "walls": matches, "bottoms": bottom_matches})
    passed = (abs(after_volume - before_volume - expected) < 1e-4 and removed < 1e-5
              and outside_volume < 1e-5 and all(c["pass"] for c in target_checks)
              and sha(source) == prediction["input"]["sha256"].lower())
    audit = {"status": "PASS" if passed else "FAIL", "truth_used": False,
             "source_step": str(source), "source_sha256": sha(source),
             "prediction_sha256": sha(prediction_path), "model": prediction["model"],
             "output_step": str(destination), "output_sha256": sha(destination),
             "method": "B-Rep boolean union with annular material inside predicted blind holes",
             "target_count": len(holes), "targets": holes, "target_checks": target_checks,
             "volume_before_mm3": before_volume, "volume_after_mm3": after_volume,
             "expected_volume_gain_mm3": expected, "actual_volume_gain_mm3": after_volume - before_volume,
             "removed_original_material_mm3": removed, "added_material_outside_target_rings_mm3": outside_volume,
             "roundtrip_valid_solid_count": len(list(TopologyExplorer(reread).solids())),
             "face_count_before": len(faces), "face_count_after": len(new_faces),
             "source_file_unchanged": sha(source) == prediction["input"]["sha256"].lower()}
    write_json(output / "edit_audit.json", audit)
    if not passed:
        raise RuntimeError("CAD verification failed; see edit_audit.json")
    print(json.dumps({k: audit[k] for k in ("status", "target_count", "actual_volume_gain_mm3", "output_step")}, indent=2))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("cad", "cae", "both"), default="both")
    args = parser.parse_args()
    if args.case in ("cae", "both"):
        build_cae()
    if args.case in ("cad", "both"):
        build_cad()
