"""Pure STEP measurements and local B-Rep primitives adapted from verified kernels.
No feature classifier, object cache, reference table, or part-name dispatch.
"""
from __future__ import annotations
import math
import itertools
from pathlib import Path
from common import sha, write as write_json
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakePrism
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_REVERSED
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Pnt2d, gp_Vec, gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer
PlanError=ValueError
BOOLEAN_EVENTS=[]
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

def clean(shape):
    builder = ShapeUpgrade_UnifySameDomain(shape, True, True, False)
    builder.Build()
    return builder.Shape()
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
def dot(a,b):
    return sum(x*y for x,y in zip(a,b))

def sub(a,b):
    return [x-y for x,y in zip(a,b)]

def add(a,b):
    return [x+y for x,y in zip(a,b)]

def mul(a,s):
    return [x*s for x in a]

def unit(a):
    return mul(a,1/math.hypot(*a))

def cylinder(row):
    if "radius_mm" not in row:
        raise PlanError("noncylindrical_face")
    radius = row["radius_mm"]
    depth = row["v_range"][1]-row["v_range"][0]
    if abs(row["u_range"][1]-row["u_range"][0]-2*math.pi)>1e-6 or abs(row["area_mm2"]-2*math.pi*radius*depth)>1e-4:
        raise PlanError("trimmed_cylinder_not_supported_for_edit")
    return {"face_number":row["face_number"],"radius_mm":radius,"depth_mm":depth,"axis":row["axis"],
            "start_mm":add(row["origin_mm"],mul(row["axis"],row["v_range"][0])),"center_mm":row["center_mm"]}

def thickness(data,obj):
    rows = [data["faces"][n-1] for n in obj["original_face_numbers"]]
    pairs = []
    for a,b in itertools.combinations(rows,2):
        if "normal" not in a or "normal" not in b or dot(a["normal"],b["normal"])>-1+1e-7:
            continue
        displacement = sub(b["center_mm"],a["center_mm"])
        distance = abs(dot(displacement,a["normal"]))
        if distance < 1e-6 or abs(a["area_mm2"]-b["area_mm2"])>1e-4:
            continue
        av, bv = a["vertices_mm"], b["vertices_mm"]
        if len(av)!=len(bv) or any(min(math.dist(add(v,displacement),w) for w in bv)>1e-5 for v in av):
            continue
        pairs.append({"object_index":obj["object_index"],"side_faces":[a["face_number"],b["face_number"]],
                      "thickness_mm":distance,"sweep_vector_mm":displacement,"side_area_mm2":a["area_mm2"],
                      "midpoint_mm":mul(add(a["center_mm"],b["center_mm"]),.5),"normal":a["normal"]})
    if not pairs:
        raise PlanError("parallel_congruent_sides_not_found")
    return max(pairs,key=lambda row:row["side_area_mm2"])
def boolean(kind,first,second):
    for tolerance in (1e-7,1e-6,1e-5):
        op=kind(first,second)
        op.SetFuzzyValue(tolerance)
        op.Build()
        if op.IsDone() and BRepCheck_Analyzer(op.Shape()).IsValid():
            if tolerance>1e-7:
                BOOLEAN_EVENTS.append({"operation":kind.__name__,"fuzzy_tolerance_mm":tolerance})
            return op.Shape()
    raise PlanError("boolean_invalid_at_allowed_tolerances")

def move(shape,vector):
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(*vector))
    return BRepBuilderAPI_Transform(shape,transform,True).Shape()

def tube(row,radius=None,shift=None):
    start = add(row["start_mm"],shift or [0,0,0])
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*start),gp_Dir(*row["axis"])),
                                   radius if radius is not None else row["radius_mm"],row["depth_mm"]).Shape()

def volume(shape):
    return abs(properties(shape,True)[0])

def join(shapes):
    result = shapes[0]
    for shape in shapes[1:]:
        result = boolean(BRepAlgoAPI_Fuse,result,shape)
    return result

def source_data(path):
    shape=read_step(path)
    faces=list(TopologyExplorer(shape).faces())
    rows=[measured(f,i+1) for i,f in enumerate(faces)]
    return shape,faces,rows

def import_face_record(face, number):
    record=face_record(face,number)
    record['boundary_vertices_mm']=[xyz(BRep_Tool.Pnt(v)) for v in TopologyExplorer(face).vertices()]
    return record
