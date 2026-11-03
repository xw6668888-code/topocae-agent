from __future__ import annotations
NATIVE_KEYS = {"face_attr", "face_grid", "edge_attr", "edge_grid", "edge_index", "native_to_step"}
import numpy as np

INPUT_KEYS = {"sample_id", "face_count", "step"}


def validate_native(arrays: dict[str, np.ndarray], expected_faces: int) -> np.ndarray:
    if set(arrays) != NATIVE_KEYS:
        raise ValueError("MFT author extractor changed its geometry-only array inventory")
    mapping = np.asarray(arrays["native_to_step"], dtype=np.int64)
    if mapping.shape != (expected_faces,) or sorted(mapping.tolist()) != list(range(expected_faces)):
        raise ValueError("Author native-to-final-STEP face history is not bijective")
    if arrays["face_attr"].shape != (expected_faces, 14) or \
            arrays["face_grid"].shape != (expected_faces, 7, 10, 10) or \
            arrays["edge_attr"].ndim != 2 or arrays["edge_attr"].shape[1] != 15 or \
            arrays["edge_grid"].shape != (len(arrays["edge_attr"]), 12, 10) or \
            arrays["edge_index"].shape != (2, len(arrays["edge_attr"])):
        raise ValueError("Author native geometry arrays changed shape or edge alignment")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Author geometry contains nonfinite arrays")
    return mapping
