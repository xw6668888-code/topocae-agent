"""Predeclared physical-equivalence metrics; evaluation-only module."""
from __future__ import annotations
import math

def close(a,b,tol=1e-8):
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and abs(a-b)<=tol+1e-8*abs(b)

def norm(v):
    length=math.hypot(*v)
    return [x/length for x in v]

def face_metrics(plan,reference,mapping):
    actual={}
    for o in (plan or {}).get('operations',[]):
        if 'face_groups' in o:
            actual[o['action']]={mapping.get(h,'UNKNOWN:'+str(h)) for g in o['face_groups'] for h in g}
    expected={a:{n for g in gs for n in g} for a,gs in reference['face_number_groups'].items()}
    rows=[]
    for action in sorted(set(actual)|set(expected)):
        a=actual.get(action,set());e=expected.get(action,set())
        rows.append(dict(action=action,exact=action in actual and action in expected and a==e,TP=len(a&e),FP=len(a-e),FN=len(e-a),
                         actual_faces=sorted(a,key=str),expected_faces=sorted(e),extra=sorted(a-e,key=str),missing=sorted(e-a)))
    ready=(plan or {}).get('status')=='ready'
    return dict(correct=ready and all(r['exact'] for r in rows) and set(actual)==set(expected),
                operations=rows,TP=sum(r['TP'] for r in rows),FP=sum(r['FP'] for r in rows),FN=sum(r['FN'] for r in rows),
                exact_operations=sum(r['exact'] for r in rows),required_operations=len(expected))

def parameters(plan,reference,mapping,geometry):
    if not plan or plan.get('status')!='ready': return dict(correct=False,differences=['no_ready_plan'])
    target=reference['reference_plan'];actual={o['action']:o for o in plan['operations']};expected={o['action']:o for o in target['operations']}
    differences=[]
    if plan['domain']!=target['domain']: differences.append('domain')
    if set(actual)!=set(expected): differences.append('actions')
    area={f['handle']:f['area_mm2'] for f in geometry['faces']}
    for action in sorted(set(actual)&set(expected)):
        a,e=actual[action],expected[action]
        for k in set(e)-{'action','face_groups','basis'}:
            if action=='apply_force' and k=='magnitude_N': continue
            av,ev=a.get(k),e[k]
            if k in ('direction','axis_direction'):
                ok=isinstance(av,list) and len(av)==3 and math.hypot(*av)>0 and all(close(x,y) for x,y in zip(norm(av),norm(ev)))
            elif k=='axis_point_mm': ok=isinstance(av,list) and len(av)==3 and all(close(x,y,1e-5) for x,y in zip(av,ev))
            elif isinstance(ev,(int,float)):
                tol=.0001 if k in ('old_thickness_mm','old_spacing_mm') else (1e-5 if k in ('new_radius_mm','new_thickness_mm','new_spacing_mm') else 1e-8)
                ok=close(av,ev,tol)
            else: ok=av==ev
            if not ok: differences.append(action+'.'+k)
        if action=='apply_force':
            def allocation(op):
                groups=op.get('face_groups',[])
                areas=[sum(area.get(h,0) for h in g) for g in groups]
                total=sum(areas)
                if total<=0: return None
                return {tuple(sorted((mapping.get(h,'UNKNOWN:'+str(h)) for h in g),key=str)):
                    op['magnitude_N'] if op.get('basis')=='per_instance' else op['magnitude_N']*ar/total
                    for g,ar in zip(groups,areas)}
            aa,ee=allocation(a),allocation(e)
            if aa is None or ee is None or set(aa)!=set(ee) or any(not close(aa[k],ee[k],1e-6) for k in ee if k in aa):
                differences.append('apply_force.per_object_resultants')
    return dict(correct=not differences,differences=differences)
