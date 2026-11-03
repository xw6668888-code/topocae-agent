from __future__ import annotations

import argparse
import numpy as np
import torch

from common import *
from ablations import masked_inputs, expanded_relations, forward_dense, CONFIGS, make_model
from stages import recover_stages, without_truth
from p1_e6r.common import MACRO_CLASSES
from utils.training_export_v2_2_1 import export_graph_v2_2_1_training_record


def graph_inputs(graph, metadata, config='full'):
    graph = without_truth(graph)
    exported = export_graph_v2_2_1_training_record(graph)
    q, norm = metadata['quantile_transform'], metadata['normalization']
    node = np.asarray(exported['node_features'], dtype=np.float32).reshape(-1, 40)
    edge = np.asarray(exported['edge_features'], dtype=np.float32).reshape(-1, 20)
    node = np.clip(node, np.asarray(q['node_lower'], dtype=np.float32), np.asarray(q['node_upper'], dtype=np.float32))
    edge = np.clip(edge, np.asarray(q['edge_lower'], dtype=np.float32), np.asarray(q['edge_upper'], dtype=np.float32))
    node = (node - np.asarray(norm['node_mean'], dtype=np.float32)) / np.asarray(norm['node_std'], dtype=np.float32)
    edge = (edge - np.asarray(norm['edge_mean'], dtype=np.float32)) / np.asarray(norm['edge_std'], dtype=np.float32)
    node, edge = masked_inputs(torch.from_numpy(node), torch.from_numpy(edge), config)
    adjacency = torch.tensor(exported['edges'], dtype=torch.long).reshape(-1, 2).t().contiguous()
    primitive = torch.tensor([n.get('primitive_surface_id', 0) for n in graph['nodes']], dtype=torch.long)
    return node, adjacency, primitive, edge


def probabilities_to_prediction(macro, semantic, edge):
    faces = []
    for i in range(len(macro)):
        m, s = int(macro[i].argmax()), int(semantic[i].argmax())
        faces.append({'idx': i, 'macro_feature': MACRO_CLASSES[m], 'macro_class_id': m,
            'macro_confidence': float(macro[i, m]),
            'macro_probabilities': {name: float(macro[i, j]) for j, name in enumerate(MACRO_CLASSES)},
            'semantic_label': s, 'semantic_confidence': float(semantic[i, s]), 'embedding': []})
    return {'faces': faces, 'edge_probabilities': edge.tolist()}


def predict_tensors(model, tensors, config='full', device='cpu'):
    node, edges, primitive, features = [t.to(device) for t in tensors]
    model.eval()
    with torch.inference_mode():
        if CONFIGS[config]['head'] == 'dense':
            pairs, attrs = expanded_relations(edges, features, len(node))
            out = forward_dense(model, node, edges, primitive, features, pairs, attrs)
        else:
            out = model(node, edges, primitive, features)
        prediction = probabilities_to_prediction(torch.softmax(out['macro_logits'], 1).cpu(),
            torch.softmax(out['semantic_logits'], 1).cpu(), torch.sigmoid(out['edge_same_instance_logits']).cpu())
        if CONFIGS[config]['head'] == 'dense':
            prediction.update(relation_edges=pairs.cpu().t().tolist(), relation_probabilities=torch.sigmoid(out['relation_logits']).cpu().tolist())
    return prediction


def load_frozen(config, seed):
    freeze = read_json(REPORT / 'CONFIGURATION_FREEZE.json')
    if config not in freeze['configs'] or seed not in freeze['seeds']:
        raise ValueError('configuration not frozen')
    run = run_dir(seed) if config == 'full' else ARTIFACT / 'training_runs' / config / f'seed_{seed}'
    checkpoint = run / 'best.pt'
    expected_metadata = next(item for item in freeze['frozen_files'] if item['path'] == (DATA / 'metadata.json').resolve().as_posix())
    if sha256(DATA / 'metadata.json') != expected_metadata['sha256']:
        raise ValueError('frozen normalization changed')
    if config == 'full':
        expected = next(item for item in freeze['baseline_runs'] if item['seed'] == seed)
        checkpoint_identity = next(item for item in expected['files'] if Path(item['path']).name == 'best.pt')
    else:
        if read_json(run / 'STATUS.json')['status'] != 'completed':
            raise ValueError('model training not complete')
        checkpoint_identity = read_json(run / 'metrics.json')['checkpoint']
    if sha256(checkpoint) != checkpoint_identity['sha256']:
        raise ValueError('checkpoint bytes changed')
    model = make_model()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    if config != 'full' and saved['configuration_freeze_sha256'] != sha256(REPORT / 'CONFIGURATION_FREEZE.json'):
        raise ValueError('checkpoint freeze mismatch')
    return model.eval(), read_json(run / 'same_instance_threshold_selection.json'), checkpoint


def infer_graph(graph, config, seed, device='cpu'):
    model, same, checkpoint = load_frozen(config, seed)
    model.to(device)
    prediction = predict_tensors(model, graph_inputs(graph, read_json(DATA / 'metadata.json'), config), config, device)
    candidate, decoder = [read_json(p) for p in decoder_assets()]
    stages = recover_stages(graph, prediction, same, candidate, decoder, 'unlabelled_input')
    return {'config': config, 'seed': seed, 'checkpoint': hashed(checkpoint), 'truth_used': False,
        'prediction': prediction, 'feature_objects': stages['objects']['full'], 'stages': stages,
        'numerical_projection_guard': False, 'role_completion': False,
        'scope': 'G03 controlled recognition pipeline; engineering-service numeric guards and role completion are separate.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--config', choices=CONFIGS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.input.suffix.lower() in ('.step', '.stp', '.brep'):
        from core.graph_v2_2_1 import extract_graph_v2_2_1
        if args.input.suffix.lower() == '.brep':
            from p1_e6r2_r3.r3 import _read_brep
            shape = _read_brep(args.input)
        else:
            from utils.step_solid_validation import read_step_shape
            shape = read_step_shape(args.input)
        graph = extract_graph_v2_2_1(shape, build_patches=False)
    else:
        graph = read_json(args.input)
    result = infer_graph(graph, args.config, args.seed)
    write_json(args.output, result)
    print(json.dumps({'output': str(args.output), 'objects': len(result['feature_objects']), 'truth_used': False}))


if __name__ == '__main__':
    main()
