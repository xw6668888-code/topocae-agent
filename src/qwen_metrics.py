"""Original independent physical-equivalence scoring functions, relocated without changes."""
from pathlib import Path
import math, json

def decode(path):
    chunks=[];usage=None;done=False;finish=None
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if not line.startswith('data:'):continue
            data=line[len('data:'):].strip()
            if data=='[DONE]':done=True;continue
            if not data:continue
            message=json.loads(data)
            if message.get('usage'):usage=message['usage']
            for c in message.get('choices',[]):
                chunks.append(c.get('delta',{}).get('content') or '')
                if c.get('finish_reason') is not None:finish=c['finish_reason']
    return ''.join(chunks),usage,done,finish


def same_number(a,b,tolerance=1e-8):
    return isinstance(a,(int,float)) and abs(a-b)<=tolerance+abs(b)*1e-8


def normalized(v):
    length=math.sqrt(sum(x*x for x in v));return [x/length for x in v]


def scores(plan,reference,mapping,geometry):
    if not plan or plan.get('status')!='ready':return False,False
    actual={o['action']:o for o in plan['operations']};expected={o['action']:o for o in reference['reference_plan']['operations']}
    masks={name:set(mapping.get(h,'unknown:'+h) for gs in op['face_groups'] for h in gs) for name,op in actual.items() if 'face_groups' in op}
    expected_masks={name:set(n for gs in groups for n in gs) for name,groups in reference['face_number_groups'].items()}
    localization=masks==expected_masks
    parameters=plan['domain']==reference['reference_plan']['domain'] and set(actual)==set(expected)
    area={f['handle']:f['area_mm2'] for f in geometry['faces']}
    for name,e in expected.items():
        if name not in actual:parameters=False;continue
        a=actual[name]
        for key,value in e.items():
            if key in ('action','face_groups','basis') or (name=='apply_force' and key=='magnitude_N'):continue
            got=a.get(key)
            if key in ('direction','axis_direction'):
                equal=isinstance(got,list) and len(got)==3 and sum(x*x for x in got)>0
                if equal:equal=all(same_number(x,y) for x,y in zip(normalized(got),normalized(value)))
            elif key=='axis_point_mm':equal=isinstance(got,list) and len(got)==3 and all(same_number(x,y,1e-5) for x,y in zip(got,value))
            elif isinstance(value,(int,float)):
                tol=.0001 if key in ('old_thickness_mm','old_spacing_mm') else 1e-5 if key in ('new_radius_mm','new_thickness_mm','new_spacing_mm') else 1e-8
                equal=same_number(got,value,tol)
            else:equal=got==value
            parameters=parameters and equal
        if name=='apply_force':
            # Independently integrate uniform face tractions over each requested object.
            def field(op):
                ids=[h for gs in op['face_groups'] for h in gs]
                if any(h not in area for h in ids):return {}
                total=sum(area[h] for h in ids);out={}
                for gs in op['face_groups']:
                    denominator=total if op['basis']=='total' else sum(area[h] for h in gs)
                    for h in gs:out[h]=op['magnitude_N']/denominator
                return out
            af,ef=field(a),field(e)
            parameters=parameters and set(af)==set(ef) and all(same_number(af.get(h),v) for h,v in ef.items())
            for gs in e['face_groups']:
                parameters=parameters and same_number(sum(area[h]*af.get(h,0) for h in gs),sum(area[h]*ef[h] for h in gs),1e-6)
    return localization,parameters

