"""Execute a frozen Table 4 response using direct face selections, without a new API call."""
from pathlib import Path
import argparse, json, subprocess, sys, os
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'direct_execution'),str(ROOT/'src')]
from common import read,write,sha
from contract import parse
from qwen_metrics import decode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('condition',choices=['QP-B','QP-S','QM-B'])
    p.add_argument('case_id');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--abaqus',help='Optional Abaqus executable; CAE defaults to script generation')
    p.add_argument('--cpus',type=int,default=4);p.add_argument('--memory-mb',type=int,default=6144)
    p.add_argument('--matrix-solver',choices=['AUTO','DIRECT','ITERATIVE'],default='DIRECT')
    a=p.parse_args();out=a.output.resolve()
    if out.exists():p.error('Use a new output directory')
    if a.cpus<1 or a.memory_mb<1024:p.error('Invalid resource setting')
    sid=a.condition+'__'+a.case_id;base=ROOT/'data/qwen_unified'
    slot=next((r for r in read(base/'slots.json') if r['slot_id']==sid),None)
    if slot is None:p.error('Unknown task')
    attempt=next(r for r in slot['attempts'] if r['attempt_number']==slot['selected_attempt'])
    response=ROOT/attempt['response_path'];assert sha(response)==attempt['response_sha256']
    mapping=read(base/'inputs/mapping'/f'{a.case_id}.json')['handle_to_final_step_face_number']
    raw,usage,done,finish=decode(response)
    if not done or finish in (None,'length'):raise ValueError('No complete response')
    plan=parse(raw,mapping);out.mkdir(parents=True)
    write(out/'provider_plan.json',plan)
    if plan['status']!='ready':
        write(out/'RECEIPT.json',{'status':plan['status'],'executed':False,'new_API_calls':0});return
    step=ROOT/'data/engineering'/a.case_id/'input.step'
    geo=read(base/'inputs/geometry'/f'{a.case_id}.json')
    # The handle map is tied to the original STEP used for the archived response.
    stored_map=read(base/'inputs/mapping'/f'{a.case_id}.json')
    expected=stored_map.get('geometry_version') or geo.get('geometry_version')
    if expected:assert sha(step)==expected
    receipt={'slot_id':sid,'new_API_calls':0,'source_step_sha256':sha(step),'truth_used':False,
        'reference_plan_used':False,'response_sha256':sha(response)}
    if plan['domain']=='cad':
        from cad_edit import execute
        target=execute(step,plan,mapping,out/'work')
        receipt.update(status='CAD_EXECUTED',output_step=str(target),audit=read(out/'work/edit_audit.json'))
    else:
        from cae_compile import compile_plan
        compile_plan(step,plan,mapping,read(base/'inputs/materials.json'),out/'work',{'cpus':a.cpus,'memory_mb':a.memory_mb})
        script=out/'work/final.py';compile(script.read_text(encoding='utf-8'),str(script),'exec')
        receipt.update(status='CAE_COMPILED',script=str(script),matrix_solver=a.matrix_solver)
        if a.abaqus:
            env={**os.environ,'AEI_MATRIX_SOLVER':a.matrix_solver}
            with (out/'solver.log').open('w',encoding='utf-8') as log:
                run=subprocess.run([a.abaqus,'cae','noGUI='+str(script)],cwd=out/'work',env=env,stdout=log,stderr=subprocess.STDOUT)
            audit_path=out/'work/run/execution_audit.json'
            receipt.update(status='SOLVER_RETURNED',returncode=run.returncode)
            if audit_path.exists():receipt['audit']=read(audit_path)
    write(out/'RECEIPT.json',receipt)
    print(json.dumps({'slot_id':sid,'status':receipt['status'],'new_API_calls':0}))

if __name__=='__main__':main()
