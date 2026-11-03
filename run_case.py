"""Fresh STEP recognition -> archived Qwen response -> object binding -> CAD/CAE.

This deterministic replay makes no provider call. It consumes the original model
response, not an author reference plan. New provider calls are separate experiments.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from language_contract import parse_plan
from runtime_loader import prepare_runtime
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case_id',choices=[r['case_id'] for r in read(ROOT/'data/engineering/cases.json')])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prediction',type=Path,help='Optional fresh output from infer.py; STEP identity is checked.')
    p.add_argument('--abaqus',help='Abaqus command; CAE cases compile only unless this is supplied.')
    a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('Use a new output directory.')
    out.mkdir(parents=True);base=ROOT/'data/engineering'/a.case_id;step=(base/'input.step').resolve()
    prediction=a.prediction.resolve() if a.prediction else out/'prediction.json'
    if not a.prediction:
        subprocess.run([sys.executable,str(ROOT/'infer.py'),str(step),'--output',str(prediction)],check=True)
    pred=read(prediction)
    assert pred['truth_used'] is False and pred['input_step']['sha256']==hashlib.sha256(step.read_bytes()).hexdigest()
    # The response was archived from the actual Qwen provider call.
    response=next(r for r in read(ROOT/'data/language/b1_responses.json') if r['case_id']==a.case_id and r['condition']=='S')
    request=read(ROOT/f'data/language/requests/{response["slot_id"]}.json')
    payload=json.loads(next(m['content'] for m in request['messages'] if m['role']=='user'))
    context=payload['confirmed_context']
    raw=parse_plan(response['response_content'],geometry=context['unlabeled_geometry'],semantic=context['semantic_service'])
    if raw['status']!='ready':raise ValueError('Archived model requested clarification; execution stopped.')
    plan=deepcopy(raw);plan['schema_version']='2'
    for op in plan['operations']:
        if 'target' in op:
            if op['target'].pop('selector_type')!='semantic':raise ValueError('Expected semantic selector')
    write(out/'provider_plan.json',raw);write(out/'request.json',request)
    runtime=prepare_runtime();sys.path.insert(0,str(runtime/'tools/aei_paper_goals_v1/g10'))
    import m6_runner as runner
    from m6_contract import resolve_plan
    from g10_common import descriptor
    runner.CASE_SUITE=runtime/'tools/aei_case_suite';runner.PILOT=runtime/'tools/aei_two_case_pilot'
    def output_text(path,text,overwrite=False):
        path=Path(path).resolve()
        if not path.is_relative_to(out) or (path.exists() and not overwrite):
            raise ValueError('Output must be a new file inside the requested case directory')
        text=text.replace('g10_frozen_e3_user_confirmed_reference_plan','archived_qwen_response_fresh_step')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
        return descriptor(path)
    def output_json(path,value,overwrite=False):
        return output_text(path,json.dumps(value,ensure_ascii=True,indent=2)+'\n',overwrite)
    # The frozen compiler's old workspace writer is replaced by an equally
    # bounded public-output writer. The inference and physics code is unchanged.
    runner.write_text=output_text;runner.write_json=output_json
    part=next(r['part'] for r in read(ROOT/'data/engineering/cases.json') if r['case_id']==a.case_id)
    case={'case_id':a.case_id,'part':part,'domain':plan['domain'],'plan':plan,
          'source_step':descriptor(step),'prediction':descriptor(prediction),'frozen_binding':resolve_plan(plan,pred)}
    write(out/'case_binding.json',case)
    adapter=runner._load_adapter(case)
    receipt={'case_id':a.case_id,'response_slot':response['slot_id'],'new_api_calls':0,'truth_used':False,
             'fresh_recognition':a.prediction is None,'reference_plan_used':False,
             'operation_face_sets':case['frozen_binding']['selections']}
    if plan['domain']=='cad':
        with runner._patched_cad_adapter(adapter) as executor:
            edited=executor.execute(part,plan,out/'cad',execution_mode='archived_qwen_response_fresh_step')
        with runner._case_suite_imports():
            from build_cases import read_step,properties
            from OCC.Extend.TopologyUtils import TopologyExplorer
            shape=read_step(edited);solids=len(list(TopologyExplorer(shape).solids()));volume=properties(shape,True)[0]
        if solids!=1 or volume<=0:raise ValueError('Edited STEP failed geometric readback')
        receipt.update(status='CAD_EXECUTED',output_step=str(edited),solid_count=solids,volume_mm3=volume)
    else:
        run=out/'solver';run.mkdir()
        script=runner._compile_cae(case,adapter,out,run,descriptor(out/'case_binding.json'))
        receipt.update(status='CAE_COMPILED',script=str(script))
        if a.abaqus:
            with (out/'abaqus_console.txt').open('w',encoding='utf-8') as log:
                r=subprocess.run([a.abaqus,'cae','noGUI='+str(script)],cwd=run,stdout=log,stderr=subprocess.STDOUT)
            receipt['abaqus_returncode']=r.returncode
            audits=list(run.glob('*audit*.json'))
            receipt['solver_audits']=[{'file':x.name,'status':read(x).get('status')} for x in audits]
            if r.returncode or not audits or any(read(x).get('status')=='FAIL' for x in audits):
                write(out/'RECEIPT.json',receipt);raise RuntimeError('CAE run did not pass; inspect saved audit and console')
            receipt['status']='CAE_EXECUTED';receipt['odb_files']=[str(x) for x in run.glob('*.odb')]
    write(out/'RECEIPT.json',receipt);print(json.dumps({'case_id':a.case_id,'status':receipt['status'],'output':str(out)}))
if __name__=='__main__':main()
