class BaseFeatureGenerator:
    def __init__(self, semantic_subtype_index):
        self.semantic_subtype_index = semantic_subtype_index
        self.feature_name = "base"
        self.semantic_map = self.build_semantic_map(["stock_face"])
        self.generation_context = {}

    def build_semantic_map(self, aliases):
        if self.semantic_subtype_index is None:
            raise ValueError("semantic_subtype_index is required for semantic label resolution.")

        alias_to_key = self.semantic_subtype_index["alias_to_key"]
        key_to_id = self.semantic_subtype_index["key_to_id"]
        semantic_map = {}

        for alias in aliases:
            canonical_key = alias_to_key.get(alias)
            if canonical_key is None:
                raise KeyError(f"Semantic subtype alias '{alias}' is not defined in ontology.")
            semantic_map[alias] = key_to_id[canonical_key]

        return semantic_map

    def apply_feature(self, stock, L, W, H, num_instances, start_inst_id, occupied_zones):
        raise NotImplementedError()
        
    def match_face(self, surf, pnt, face_cx, face_cy, H, params_list):
        return None

    def set_generation_context(self, generation_context):
        self.generation_context = generation_context or {}
        return self

    def select_feature_axis(self, feature_family):
        orientation = self.generation_context.get("feature_orientation", {})
        axes = orientation.get(f"{feature_family}_axes") or ["z"]
        axes = self._filter_axes_for_current_stock(feature_family, list(axes))
        return random_choice(axes)

    def _filter_axes_for_current_stock(self, feature_family, axes):
        if feature_family != "hole":
            return axes
        stock_profile = self.generation_context.get("current_stock_profile", {})
        stock_type = stock_profile.get("stock_type")
        if stock_type in {"cylinder", "composite"}:
            return ["z"]
        return axes or ["z"]


def random_choice(values):
    import random

    return random.choice(list(values))
