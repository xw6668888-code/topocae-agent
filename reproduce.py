"""Offline reproduction of paper tables from individual frozen observations."""
from pathlib import Path
from collections import Counter, defaultdict
from copy import deepcopy
import argparse, csv, hashlib, json, math, statistics, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def close(a,b):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-9),(a,b)
def table(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def recognition(out):
    from recount import recount
    result=recount(ROOT/'data/t10')
    write(out/'recognition_recount.json',result)
    source=read(ROOT/'data/t10/recorded_main_results.json')
    # The independent recount above checks every individual channel, metric and tier.
    for name in ('MAIN_RESULTS.csv','ABLATION_RESULTS.csv'):
        rows=list(csv.DictReader((ROOT/'results/expected'/name).open(encoding='utf-8-sig')))
        for row in rows:
            for metric in ('macro_f1','micro_f1','whole_part_exact_rate','face_accuracy'):
                if metric+'_mean' in row:close(float(row[metric+'_mean']),source['three_seed_summary'][row['score_key']][metric]['mean'])
        table(out/name,rows)
    subset_rows=[]
    for method in ('E3','AAGNet18-SI','MFTReNet-SI-D0'):
        for tier in ('overall','T10-I','T10-P','T10-S'):
            values=[]
            for seed in (20260718,20260719,20260720):
                channel=source['channels'][f'{method}|{seed}']
                v=channel['overall'] if tier=='overall' else channel['per_tier'][tier]
                values.append({'part_accuracy':v['whole_part_exact_rate'],'face_accuracy':v['faces']['accuracy'],
                               'macro_f1':v['exact_instances']['macro_f1'],'micro_f1':v['exact_instances']['micro_f1']})
            row={'method':method,'subset':tier,'parts_per_seed':1200 if tier=='overall' else 400}
            for metric in values[0]:
                row[metric+'_mean']=statistics.mean(v[metric] for v in values)
                row[metric+'_sample_sd']=statistics.stdev(v[metric] for v in values)
            subset_rows.append(row)
    table(out/'comparison_by_subset.csv',subset_rows)
    return {'status':result['status'],'parts':1200,'channels':len(source['channels']),
            'predictions':1200*len(source['channels']),'E3':source['three_seed_summary']['E3']}

def engineering(out):
    from paper_evidence import engineering_tables
    result=engineering_tables()
    cases=read(ROOT/'data/engineering/cases.json')
    counts=Counter()
    for c in cases:
        assert c['status']=='PASS' and c['actually_executed'] and c['truth_used'] is False
        base=ROOT/'data/engineering'/c['case_id']
        for name,key in [('input.step','geometry_version'),('prediction.json','prediction_sha256')]:
            assert hashlib.sha256((base/name).read_bytes()).hexdigest()==c[key]
        if c['domain']=='cae':
            audit=read(base/'independent_odb_audit.json');assert audit['status']=='PASS'
            close(audit['max_displacement_mm'],c['max_displacement_mm'])
        else:assert (base/'output.step').is_file()
        counts[c['domain']]+=1
    assert dict(counts)=={'cae':7,'cad':6}
    result['archived_reference_plan_artifacts']=dict(counts)
    write(out/'engineering.json',result)
    return result

def historical(out):
    from language_contract import parse_plan,ground_plan
    from historical_metrics import face_union,same_source_parameters
    base=ROOT/'data/language';expected=read(base/'recorded_equivalence.json')
    erecord={r['slot_id']:r for r in expected['rows']};refs=read(base/'references.json')
    rows=[];totals=defaultdict(Counter)
    for r in read(base/'b1_responses.json'):
        cid,sid,condition=r['case_id'],r['slot_id'],r['condition']
        geo=read(base/f'catalogs/unlabeled_geometry/{cid}.json')
        sem=read(base/f'catalogs/e3_semantic_service/{cid}.json') if condition=='S' else None
        mapping=read(base/f'catalogs/local_handle_maps/{cid}.json')['handle_to_final_step_face_number']
        body=read(base/f'requests/{sid}.json')
        payload=json.loads(next(m['content'] for m in body['messages'] if m['role']=='user'))
        assert payload['confirmed_context']['semantic_service']==sem
        assert payload['confirmed_context']['unlabeled_geometry']==geo
        plan=parse_plan(r['response_content'],geometry=geo,semantic=sem)
        ref=refs[cid];wanted={s['action']:s['role_face_sets'] for s in ref['frozen_binding']['selections']}
        actual={};face=False;param=False
        if plan['status']=='ready':
            bound=ground_plan(plan,geo,sem,mapping)
            actual={s['action']:s['face_number_groups'] for s in bound['resolved_targets']}
            face={a:face_union(g) for a,g in actual.items()}=={a:face_union(g) for a,g in wanted.items()}
            param,_=same_source_parameters(plan,ref['plan'],actual,wanted)
        old=erecord[sid];ex=old['existing_execution'];evidence=False
        for key in ('evidence','independent_odb_audit','independent_cad_audit'):
            d=ex.get(key,{})
            if 'packaged_path' in d:
                assert hashlib.sha256((ROOT/d['packaged_path']).read_bytes()).hexdigest()==d['sha256']
        if ex['kind']=='old_equivalent_artifact':
            receipt=read(ROOT/ex['evidence']['packaged_path'])
            evidence=receipt['status']=='PASS' and receipt['actually_executed']
        elif ex['kind']=='new_CAE_ODB':
            audit=read(ROOT/ex['independent_odb_audit']['packaged_path'])
            evidence=not ex['different_physical_fields'] and audit['status']=='PASS'
        elif ex['kind']=='new_CAD_STEP':
            audit=read(ROOT/ex['independent_cad_audit']['packaged_path'])
            evidence=audit['status']=='PASS' and all(audit['checks'].values())
        passed=bool(face and param and evidence)
        assert [face,param,passed]==[old['operation_faces_correct'],old['engineering_parameters_equivalent'],old['verified_engineering_task']],sid
        rows.append(dict(case_id=cid,condition=condition,face_correct=face,parameters_correct=param,task_verified=passed,artifact_origin=ex['kind']))
        totals[condition].update(tasks=1,face_correct=face,parameters_correct=param,task_verified=passed)
    assert totals['S']['task_verified']==13 and totals['B']['task_verified']==2
    table(out/'historical_object_binding.csv',rows)
    write(out/'historical_object_binding.json',{'counts':{k:dict(v) for k,v in totals.items()},
        'evaluation':'posthoc physical-equivalence evaluation of original responses',
        'artifact_reuse_included':True,'both_execution_conditions_share_object_backed_backend':True})
    return {k:dict(v) for k,v in totals.items()}

def unified(out):
    from qwen_metrics import decode,scores
    import jsonschema
    base=ROOT/'data/qwen_unified';refs=read(base/'references.json');schema=read(base/'PLAN_SCHEMA.json')
    prices=read(base/'pricing.json')['prices_per_million'];slots=read(base/'slots.json')
    expected={r['condition']:r for r in read(base/'recorded_results.json')['summary']}
    groups=defaultdict(list);details=[]
    for cid in refs:
        b=read(base/f'requests/QP-B__{cid}.json')['body'];s=deepcopy(read(base/f'requests/QP-S__{cid}.json')['body'])
        bp=json.loads(b['messages'][1]['content']);sp=json.loads(s['messages'][1]['content'])
        assert bp['semantic_service'] is None and sp['semantic_service'] is not None
        sp['semantic_service']=None;assert bp==sp
        s['messages'][1]['content']=b['messages'][1]['content'];assert b==s
        assert json.loads(read(base/f'requests/QM-B__{cid}.json')['body']['messages'][1]['content'])==bp
    for slot in slots:
        sid,cid,condition=slot['slot_id'],slot['case_id'],slot['condition']
        plan=None;elapsed=0.;cost=0.;usage_total=Counter();missing=0
        for a in slot['attempts']:
            elapsed+=a['elapsed_seconds'];usage=None
            if a.get('response_path'):
                p=ROOT/a['response_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==a['response_sha256']
                content,usage,done,finish=decode(p)
                assert usage==a.get('usage')
                if a['attempt_number']==slot['selected_attempt'] and done and finish not in (None,'length'):
                    try:plan=json.loads(content);jsonschema.validate(plan,schema)
                    except (ValueError,jsonschema.ValidationError):plan=None
            if not usage:missing+=1;continue
            inp=usage['prompt_tokens'];output=usage['completion_tokens']
            cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0)
            price=prices[a['requested_model']]
            cost+=((inp-cached)*price['input']+cached*price.get('cached_input',price['input'])+output*price['output'])/1e6
            usage_total.update(prompt_tokens=inp,completion_tokens=output,total_tokens=usage['total_tokens'],cached_prompt_tokens=cached)
        mapping=read(base/f'inputs/mapping/{cid}.json')['handle_to_final_step_face_number']
        geo=read(base/f'inputs/geometry/{cid}.json')
        face,param=scores(plan,refs[cid],mapping,geo)
        physical=base/'physical'/sid;artifact=False
        if plan and plan['status']=='ready' and (physical/'receipt.json').is_file():
            receipt=read(physical/'receipt.json')
            if receipt.get('status')=='EXECUTED':
                if plan['domain']=='cad':
                    paths=[physical/'independent_STEP_readback.json',physical/'cad_audit.json']
                else:
                    paths=[physical/n for n in ('independent_odb.json','saved_cae_readback.json','physical_readback.json')]
                artifact=all(p.is_file() and read(p).get('status')=='PASS' for p in paths)
        passed=bool(face and param and artifact)
        assert [face,param,artifact,passed]==[slot['recorded_face_correct'],slot['recorded_parameters_correct'],slot['recorded_artifact_pass'],slot['recorded_total_success']],sid
        row=dict(slot_id=sid,condition=condition,case_id=cid,face_correct=face,parameters_correct=param,
                 artifact_pass=artifact,task_success=passed,API_calls=len(slot['attempts']),API_elapsed_s=elapsed,
                 cost_CNY=cost,missing_usage_attempts=missing,**usage_total)
        groups[condition].append(row);details.append(row)
    summary=[]
    for condition,rows in groups.items():
        r=dict(condition=condition,tasks=len(rows),localization_correct=sum(x['face_correct'] for x in rows),
               task_success=sum(x['task_success'] for x in rows),API_calls=sum(x['API_calls'] for x in rows),
               API_elapsed_total_s=sum(x['API_elapsed_s'] for x in rows),
               API_elapsed_mean_s=statistics.mean(x['API_elapsed_s'] for x in rows),
               API_elapsed_median_s=statistics.median(x['API_elapsed_s'] for x in rows),
               prompt_tokens=sum(x.get('prompt_tokens',0) for x in rows),
               output_tokens=sum(x.get('completion_tokens',0) for x in rows),
               total_tokens=sum(x.get('total_tokens',0) for x in rows),
               cost_total_CNY=sum(x['cost_CNY'] for x in rows),cost_mean_CNY=statistics.mean(x['cost_CNY'] for x in rows),
               missing_usage_attempts=sum(x['missing_usage_attempts'] for x in rows))
        e=expected[condition]
        for k,v in [('localization_correct','face_correct'),('task_success','total_success'),('API_calls','actual_API_requests'),
                    ('API_elapsed_total_s','API_elapsed_seconds'),('prompt_tokens','prompt_tokens'),('output_tokens','completion_tokens'),
                    ('cost_total_CNY','estimated_API_cost'),('missing_usage_attempts','cost_usage_missing_attempts')]:close(r[k],e[v])
        summary.append(r)
    table(out/'qwen_unified_per_task.csv',details);table(out/'qwen_unified_summary.csv',summary)
    write(out/'qwen_unified.json',{'summary':summary,'price_basis':'2026-10-23 official list estimate, not invoice',
        'timing':'Sum all attempt wall times per task, including unsuccessful attempts; then arithmetic mean or median across 13 tasks. Excludes recognition, CAD and solver time.',
        'missing_usage':'Failed attempts without reported usage remain unknown; cost is the sum of observed usage, not a claim of zero cost for missing usage.'})
    return summary

def robustness(out):
    records=read(ROOT/'data/robustness/measurements.json')
    assert len(records)==len({(r['sample_id'],r['kind'],r['method'],r['seed']) for r in records})==3600
    groups=defaultdict(list)
    for r in records:
        assert r['truth_used'] is False
        groups[r['method'],r['seed'],r['kind']].append(r['comparison'])
    rows=[]
    for (method,seed,kind),values in sorted(groups.items()):
        c=Counter(v['status'] for v in values)
        rows.append(dict(method=method,seed=seed,variant=kind,fixed_parts=len(values),compared=c['compared'],
                         complete_object_invariant_true=sum(v.get('complete_object_invariant') is True for v in values),
                         complete_object_invariant_false=sum(v.get('complete_object_invariant') is False for v in values),
                         face_semantics_invariant_true=sum(v.get('face_semantics_invariant') is True for v in values),
                         face_semantics_invariant_false=sum(v.get('face_semantics_invariant') is False for v in values)))
    actual={(r['method'],r['seed'],r['variant']):r for r in rows}
    for e in csv.DictReader((ROOT/'results/expected/ROBUSTNESS.csv').open(encoding='utf-8-sig')):
        r=actual[e['method'],int(e['seed']),e['variant']]
        for k in r:
            if k not in ('method','seed','variant'):assert r[k]==int(e[k])
    table(out/'robustness.csv',rows);return {'observations':len(records),'status':'PASS'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/reproduced')
    p.add_argument('--only',choices=['all','recognition','engineering','historical','unified','robustness'],default='all')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);result={}
    for name,fn in [('recognition',recognition),('engineering',engineering),('historical',historical),('unified',unified),('robustness',robustness)]:
        if a.only not in ('all',name):continue
        print('Recomputing '+name+' ...',flush=True);result[name]=fn(a.output)
        print(name+': PASS',flush=True)
    write(a.output/'SUMMARY.json',result)
if __name__=='__main__':main()
