"""Task-level acceptance of the same frozen model plans in all three conditions.

CAD is accepted using the intended objects, parameters and verified edited geometry.
CAE model correctness is kept separate from completion of a time-limited solver run.
The original complete-face-set and solver-completion scores are retained in each row.
"""
from pathlib import Path
import hashlib, json, math

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(x,y,tol=1e-5):return isinstance(x,(int,float)) and math.isclose(x,y,abs_tol=tol,rel_tol=1e-8)

def target_object_cover(actual,expected):
    """One operated face group per intended object; no unrelated selected faces."""
    if len(actual)!=len(expected):return False
    options=[[j for j,e in enumerate(expected) if a and set(a)<=set(e)] for a in actual]
    def visit(i,used):
        return i==len(options) or any(visit(i+1,used|{j}) for j in options[i] if j not in used)
    return visit(0,set())

def evaluate(root,slot,plan,reference,mapping,strict_face,parameters):
    root=Path(root);physical=root/'data/qwen_unified/physical'/slot['slot_id']
    result={'operation_targets_correct':False,'physical_model_or_edit_verified':False,'engineering_task_success':False,
        'solver_completed':False,'acceptance_reason':'No complete executable plan'}
    if not plan or plan.get('status')!='ready':return result
    receipt=read(physical/'receipt.json') if (physical/'receipt.json').exists() else {}
    if plan['domain']=='cad':
        op=plan['operations'][0];action=op['action']
        groups=[[mapping.get(h,'unknown:'+h) for h in g] for g in op.get('face_groups',[])]
        targets=target_object_cover(groups,reference['face_number_groups'].get(action,[]))
        audit_path=physical/'cad_audit.json';back_path=physical/'independent_STEP_readback.json'
        verified=False
        if receipt.get('status')=='EXECUTED' and audit_path.exists() and back_path.exists():
            audit=read(audit_path);back=read(back_path)
            verified=audit.get('status')=='PASS' and back.get('status')=='PASS' and back.get('output_readable') is True
            verified=verified and {tuple(sorted(g)) for g in groups}=={tuple(sorted(g)) for g in audit.get('actual_face_groups',[])}
            output=physical/'output.step'
            verified=verified and output.is_file() and digest(output)==audit.get('output_sha256')
            if action=='set_thickness':
                verified=verified and len(audit.get('changes',[]))==len(groups) and all(
                    near(c.get('new_thickness_mm'),op['new_thickness_mm']) and c.get('anchor')==op['anchor']
                    for c in audit.get('changes',[]))
                verified=verified and len(back.get('checks',[]))==len(groups) and all(c.get('pass_') for c in back.get('checks',[]))
            elif action=='set_hole_radius':
                verified=verified and all(near(c.get('new_radius_mm'),op['new_radius_mm']) for c in audit.get('changes',[]))
        result.update(operation_targets_correct=targets,physical_model_or_edit_verified=bool(verified),
            engineering_task_success=bool(targets and parameters and verified),
            acceptance_reason='Intended object coverage, dimensions and geometric readback; thickness may select the two operating side faces.')
    else:
        targets=bool(strict_face);verified=False
        model_path=physical/'physical_readback.json';saved_path=physical/'saved_cae_readback.json'
        if model_path.exists() and saved_path.exists():
            model=read(model_path);saved=read(saved_path)
            checks={k:v for k,v in model.get('checks',{}).items() if k!='independent_ODB'}
            verified=model.get('status')=='PASS' and bool(checks) and all(v is True for v in checks.values()) and saved.get('status')=='PASS'
        eq=physical/'inp_equivalence.json'
        if not verified and eq.exists():
            equivalence=read(eq);audit=read(physical/'execution_audit.json')
            verified=equivalence.get('status')=='PASS' and audit.get('face_mapping_bijective') is True
            verified=verified and equivalence.get('comparison',{}).get('member_sha256')==read(physical/'solver_input_identity.json')['sha256']
        odb=physical/'independent_odb.json'
        solver=odb.exists() and read(odb).get('status')=='PASS'
        result.update(operation_targets_correct=targets,physical_model_or_edit_verified=bool(verified),
            engineering_task_success=bool(targets and parameters and verified),solver_completed=bool(solver),
            acceptance_reason='Correct actual material, mesh, boundary/load regions and parameters; solver completion reported separately.')
    if not parameters:result['acceptance_reason']+=' Action or physical parameter mismatch.'
    if not result['operation_targets_correct']:result['acceptance_reason']+=' Wrong or incomplete operating targets.'
    return result
