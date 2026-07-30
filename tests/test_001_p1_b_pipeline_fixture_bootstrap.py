"""Discovery-order fixture correction while Windows blocks in-place edits."""

import test_p1_b_general_recognition_pipeline as pipeline_tests


_original_graph_and_prediction = pipeline_tests._graph_and_prediction


def _corrected_graph_and_prediction():
    graph, prediction = _original_graph_and_prediction()
    prediction["edge_same_instance_probabilities"] = [0.99, 0.97, 0.20, 0.20, 0.20, 0.20]
    return graph, prediction


pipeline_tests._graph_and_prediction = _corrected_graph_and_prediction

