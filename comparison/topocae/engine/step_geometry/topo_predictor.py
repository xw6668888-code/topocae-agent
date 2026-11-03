"""G07 inference adapter: same trained network, explicitly selected decoder."""
from __future__ import annotations
import time
import sys
from g07_common import *
from step_entry import graph_tensors, geometry_only
sys.path.insert(0,str(ROOT/'tools/aei_paper_goals_v1/g03'))
from inference import load_frozen, probabilities_to_prediction, predict_tensors
from ablations import expanded_relations, forward_dense, make_model
import torch
from decoder import decode, legacy_stages, frozen_decoder_assets

class PreparedInputs(list):
    def __init__(self,values,graphs):
        super().__init__(values);self.graphs=graphs
    def __getitem__(self,index):
        value=super().__getitem__(index)
        return PreparedInputs(value,self.graphs[index]) if isinstance(index,slice) else value

def batch_forward(model,tensors,dense,device,configuration=None):
    node=[];adj=[];prim=[];edge=[];pairs=[];pattrs=[];apos=[];sizes=[]
    no=ro=0
    for index,(x,e,p,a) in enumerate(tensors):
        node.append(x);adj.append(e+no);prim.append(p);edge.append(a);r=0
        if dense:
            ps,pa=expanded_relations(e,a,len(x));r=ps.shape[1]
            if configuration=='dense_geometry':
                from relation_geometry import pair_attributes
                pa=pair_attributes(tensors.graphs[index],e,a,ps)
            pairs.append(ps+no);pattrs.append(pa);apos.append(torch.arange(e.shape[1])+ro);ro+=r
        sizes.append((len(x),e.shape[1],r));no+=len(x)
    args=[torch.cat(node).to(device),torch.cat(adj,1).to(device),torch.cat(prim).to(device),torch.cat(edge).to(device)]
    with torch.inference_mode():
        out=forward_dense(model,*args,torch.cat(pairs,1).to(device),torch.cat(pattrs).to(device),torch.cat(apos).to(device)) if dense else model(*args)
        mp=torch.softmax(out['macro_logits'],1).cpu();sp=torch.softmax(out['semantic_logits'],1).cpu()
        ep=torch.sigmoid(out['edge_same_instance_logits']).cpu()
        rp=torch.sigmoid(out['relation_logits']).cpu() if dense else None
    results=[];no=eo=ro=0
    for i,(n,e,r) in enumerate(sizes):
        value=probabilities_to_prediction(mp[no:no+n],sp[no:no+n],ep[eo:eo+e])
        if dense:value.update(relation_edges=(pairs[i]-no).T.tolist(),relation_probabilities=rp[ro:ro+r].tolist())
        results.append(value);no+=n;eo+=e;ro+=r
    return results

class Predictor:
    def __init__(self,dataset,method,seed,device='cuda',configuration=None,run_folder=None,development=False):
        self.dataset,self.method,self.seed,self.device=dataset,method,seed,device
        if configuration is None and method=='new' and (REPORT/'MODEL_SELECTION.json').exists():
            configuration=read(REPORT/'MODEL_SELECTION.json')['selected_configuration']
        self.config=(configuration or 'dense_instance') if method=='new' else 'full'
        if dataset=='D0' and self.config in ('full','dense_instance') and run_folder is None:
            self.model,self.same,self.checkpoint=load_frozen(self.config,seed)
            self.metadata_path=FROZEN/'output_training_p1_e6r2_r3_graph_v2_2_1_v1/metadata.json'
        elif dataset in ('D0','D1'):
            folder=Path(run_folder) if run_folder else ARTIFACT/'training_runs'/dataset/self.config/f'seed_{seed}'
            metrics=read(folder/('development_metrics.json' if development else 'metrics.json'));self.checkpoint=Path(metrics['checkpoint']['path'])
            if sha(self.checkpoint)!=metrics['checkpoint']['sha256']:raise ValueError('Checkpoint changed')
            self.model=make_model()
            self.model.load_state_dict(torch.load(self.checkpoint,map_location='cpu',weights_only=False)['model_state_dict'])
            self.same=read(metrics['threshold']['path'])
            self.metadata_path=ARTIFACT/'training_data/D1/metadata.json' if dataset=='D1' else FROZEN/'output_training_p1_e6r2_r3_graph_v2_2_1_v1/metadata.json'
        else:raise ValueError(dataset)
        self.metadata=read(self.metadata_path);self.model.to(device).eval()
        self.candidate,self.global_decode=frozen_decoder_assets()

    def inputs(self,graphs):return PreparedInputs([graph_tensors(g,self.metadata,False) for g in graphs],graphs)
    def forward(self,tensors):return batch_forward(self.model,tensors,self.config!='full',self.device,self.config)

    def recover(self,graph,prediction):
        graph=geometry_only(graph)
        if self.method=='new':
            tick=time.perf_counter();result=decode(graph,prediction,self.same,'dense_minimal')
            seconds=time.perf_counter()-tick
            objects=result['feature_objects']
            return {'candidate':objects,'full':objects}, {'full_decode_seconds':seconds,'diagnostic_seconds':0.}, result['trace']
        result=legacy_stages().recover_stages(graph,prediction,self.same,self.candidate,self.global_decode,'unlabelled_input')
        costs=result['incremental_seconds']
        costs={**costs,'full_decode_seconds':sum(costs[k] for k in ('candidate','proposals','verified','decoded','full')),
               'diagnostic_seconds':costs.get('candidate_boundary',0.)}
        return {s:result['objects'][s] for s in ('candidate','full')},costs,result.get('trace')

def public_objects(objects,mapping=None):
    return [{'class':x.get('class',x.get('macro_feature')),
             'faces':sorted(mapping[i] if mapping is not None else i for i in x.get('faces',x.get('face_indices',[])))} for x in objects]
