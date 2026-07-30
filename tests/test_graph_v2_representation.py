from __future__ import annotations

from collections import Counter
import copy
import math
import unittest

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Transform
from OCC.Core.BRepFeat import BRepFeat_SplitShape
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Extend.TopologyUtils import TopologyExplorer

from core.graph_v2_1 import GRAPH_V2_SCHEMA, canonical_graph_signature, extract_graph_v2
from utils.graph_v2_validation import compare_graph_v2_invariance, compare_partition_invariance
from utils.training_export_v2_1 import (
    GRAPH_V2_EDGE_FEATURE_SCHEMA,
    GRAPH_V2_NODE_FEATURE_SCHEMA,
    MAX_MODEL_FEATURE_ABS,
    build_graph_v2_node_feature_vector,
    export_graph_v2_training_record,
    fit_training_quantile_transform_from_graphs,
)


def box_with_through_hole():
    box = BRepPrimAPI_MakeBox(gp_Pnt(-10.0, -8.0, 0.0), 20.0, 16.0, 10.0).Shape()
    hole = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0.0, 0.0, -1.0), gp_Dir(0.0, 0.0, 1.0)), 3.0, 12.0
    ).Shape()
    cut = BRepAlgoAPI_Cut(box, hole)
    cut.Build()
    return cut.Shape()


def rotated(shape):
    transform = gp_Trsf()
    transform.SetRotation(gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(1.0, 2.0, 3.0)), 0.73)
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def scaled(shape, factor=3.7):
    transform = gp_Trsf()
    transform.SetScale(gp_Pnt(0.0, 0.0, 0.0), factor)
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def split_top_face_box():
    shape = BRepPrimAPI_MakeBox(gp_Pnt(-10.0, -8.0, 0.0), 20.0, 16.0, 10.0).Shape()
    top_face = None
    for face in TopologyExplorer(shape).faces():
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() != GeomAbs_Plane:
            continue
        plane = surface.Plane()
        if abs(plane.Axis().Direction().Z()) > 0.9 and abs(plane.Location().Z() - 10.0) < 0.1:
            top_face = face
            break
    if top_face is None:
        raise RuntimeError("top face not found")
    splitter = BRepFeat_SplitShape(shape)
    splitter.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(-10.0, 0.0, 10.0), gp_Pnt(10.0, 0.0, 10.0)).Edge(), top_face)
    splitter.Build()
    return shape, splitter.Shape()


class GraphV2RepresentationTests(unittest.TestCase):
    def test_required_relative_node_edge_and_raw_metadata_are_present(self):
        graph = extract_graph_v2(box_with_through_hole())
        self.assertEqual(GRAPH_V2_SCHEMA, graph["schema_version"])
        normalization = graph["part_metadata_v2"]["normalization"]
        raw = graph["part_metadata_v2"]["raw_physical"]
        self.assertGreater(normalization["part_total_area"], 0.0)
        self.assertGreater(normalization["part_bbox_diagonal"], 0.0)
        self.assertGreater(normalization["part_characteristic_length"], 0.0)
        self.assertGreater(raw["solid_volume"], 0.0)

        required_node = {
            "face_area_over_part_total_area",
            "face_area_over_part_bbox_diagonal_sq",
            "radius_over_part_bbox_diagonal",
            "bbox_long_over_part_bbox_diagonal",
            "projected_long_over_part_scale",
            "uv_metric_long_over_part_scale",
            "face_area_over_neighbor_area_mean",
            "local_extent_long_middle_ratio",
            "inner_loop_count",
            "outer_loop_count",
        }
        for node in graph["nodes"]:
            recognition = node["attributes"]["recognition_features_v2"]
            self.assertTrue(required_node.issubset(recognition))
            self.assertTrue(all(math.isfinite(float(value)) for value in recognition.values()))
            self.assertIn("raw_physical_v2", node["attributes"])
        self.assertTrue(any(node["attributes"]["topology_v2"]["inner_loop_count"] > 0 for node in graph["nodes"]))

        required_edge = {
            "edge_length_over_part_bbox_diagonal",
            "circle_radius_over_part_bbox_diagonal",
            "dihedral_angle_over_pi",
            "material_side_inner_evidence",
            "material_side_outer_evidence",
        }
        for edge in graph["edge_attributes"]:
            recognition = edge["recognition_features_v2"]
            self.assertTrue(required_edge.issubset(recognition))
            self.assertTrue(all(math.isfinite(float(value)) for value in recognition.values()))
            self.assertIn("raw_physical_v2", edge)
        self.assertTrue(any(edge["recognition_features_v2"]["material_side_inner_evidence"] > 0.0 for edge in graph["edge_attributes"]))

    def test_arbitrary_rotation_and_uniform_scale_keep_model_features_and_probe_predictions(self):
        shape = box_with_through_hole()
        reference = extract_graph_v2(shape)
        rotation_graph = extract_graph_v2(rotated(shape))
        scale_graph = extract_graph_v2(scaled(shape))
        rotation_result = compare_graph_v2_invariance(reference, rotation_graph)
        scale_result = compare_graph_v2_invariance(reference, scale_graph)
        self.assertGreaterEqual(rotation_result["score"], 0.999999)
        self.assertGreaterEqual(scale_result["score"], 0.999999)
        self.assertTrue(rotation_result["canonical_signature_equal"])
        self.assertTrue(scale_result["canonical_signature_equal"])

        reference_raw = reference["part_metadata_v2"]["raw_physical"]
        scaled_raw = scale_graph["part_metadata_v2"]["raw_physical"]
        self.assertAlmostEqual(reference_raw["total_surface_area"] * 3.7**2, scaled_raw["total_surface_area"], places=5)
        self.assertAlmostEqual(reference_raw["solid_volume"] * 3.7**3, scaled_raw["solid_volume"], places=5)
        self.assertAlmostEqual(
            reference["part_metadata_v2"]["normalization"]["part_characteristic_length"] * 3.7,
            scale_graph["part_metadata_v2"]["normalization"]["part_characteristic_length"],
            places=5,
        )

    def test_legal_face_partition_is_recovered_by_non_destructive_canonical_patches(self):
        reference_shape, split_shape = split_top_face_box()
        reference = extract_graph_v2(reference_shape)
        partitioned = extract_graph_v2(split_shape)
        result = compare_partition_invariance(reference, partitioned)
        self.assertEqual(6, result["reference_face_count"])
        self.assertEqual(7, result["partitioned_face_count"])
        self.assertEqual(6, result["reference_patch_count"])
        self.assertEqual(6, result["partitioned_patch_count"])
        self.assertEqual(1.0, result["score"])
        self.assertEqual(canonical_graph_signature(reference), canonical_graph_signature(partitioned))
        self.assertEqual(set(range(7)), {face for patch in partitioned["canonical_patches"] for face in patch["face_indices"]})
        self.assertTrue(any(patch["face_count"] == 2 for patch in partitioned["canonical_patches"]))

    def test_canonical_patches_do_not_cross_instance_boundaries_or_sharp_edges(self):
        reference_shape, split_shape = split_top_face_box()
        self.assertEqual(6, len(extract_graph_v2(reference_shape)["canonical_patches"]))
        labels = []
        split_top_counter = 0
        for face in TopologyExplorer(split_shape).faces():
            surface = BRepAdaptor_Surface(face)
            is_top = (
                surface.GetType() == GeomAbs_Plane
                and abs(surface.Plane().Axis().Direction().Z()) > 0.9
                and abs(surface.Plane().Location().Z() - 10.0) < 0.1
            )
            if is_top:
                split_top_counter += 1
                labels.append({
                    "semantic_label": 28,
                    "instance_id": split_top_counter,
                    "macro_feature": "rib",
                    "macro_feature_id": 28,
                })
            else:
                labels.append({"semantic_label": 0, "instance_id": 0, "macro_feature": "stock", "macro_feature_id": 0})
        labelled_graph = extract_graph_v2(split_shape, face_labels=labels)
        self.assertEqual(2, split_top_counter)
        self.assertEqual(7, len(labelled_graph["canonical_patches"]))
        self.assertFalse(any(len(set(patch["instance_ids"])) > 1 for patch in labelled_graph["canonical_patches"]))

    def test_training_export_is_finite_bounded_and_quantiles_fit_only_on_train(self):
        graphs = [extract_graph_v2(box_with_through_hole()), extract_graph_v2(scaled(box_with_through_hole(), 2.3))]
        transform = fit_training_quantile_transform_from_graphs(graphs, split_name="train", lower_quantile=0.05, upper_quantile=0.95)
        self.assertEqual(len(GRAPH_V2_NODE_FEATURE_SCHEMA), len(transform.node_lower))
        self.assertEqual(len(GRAPH_V2_EDGE_FEATURE_SCHEMA), len(transform.edge_lower))
        record = export_graph_v2_training_record(graphs[0], quantile_transform=transform)
        all_values = [value for row in record["node_features"] + record["edge_features"] for value in row]
        self.assertTrue(all(math.isfinite(value) for value in all_values))
        self.assertTrue(all(abs(value) <= MAX_MODEL_FEATURE_ABS for value in all_values))
        with self.assertRaises(ValueError):
            fit_training_quantile_transform_from_graphs(graphs, split_name="val")

        corrupted = copy.deepcopy(graphs[0]["nodes"][0])
        recognition = corrupted["attributes"]["recognition_features_v2"]
        recognition["log1p_face_area_over_part_total_area"] = float("nan")
        recognition["log1p_radius_over_part_scale"] = float("inf")
        recognition["log1p_local_extent_long_middle_ratio"] = 1e12
        vector = build_graph_v2_node_feature_vector(corrupted)
        self.assertTrue(all(math.isfinite(value) and abs(value) <= MAX_MODEL_FEATURE_ABS for value in vector))

    def test_directed_adjacency_is_symmetric_and_identity_safe(self):
        graph = extract_graph_v2(box_with_through_hole())
        edge_pairs = Counter(tuple(edge) for edge in graph["edges"])
        for source, target in edge_pairs:
            self.assertEqual(edge_pairs[(source, target)], edge_pairs[(target, source)])
        self.assertEqual(len(graph["edges"]), len(graph["edge_attributes"]))


if __name__ == "__main__":
    unittest.main()
