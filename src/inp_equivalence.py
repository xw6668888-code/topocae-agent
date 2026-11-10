"""Compare finite-element models while reporting the matrix-solver setting separately."""
from pathlib import Path
import hashlib, json, zipfile

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def sections(archive,member):
    records=[];whole=hashlib.sha256()
    with zipfile.ZipFile(archive) as z, z.open(member) as f:
        current=None
        for raw in f:
            whole.update(raw)
            line=raw.decode('latin1').strip()
            if not line or line.startswith('**'):continue
            if line.startswith('*'):
                if current:current['data_sha256']=digest.hexdigest();records.append(current)
                # Only the matrix-solver option is outside physical-model equality.
                tokens=[s.strip() for s in line.split(',')]
                normalized=','.join(t for t in tokens if not (tokens[0].lower()=='*step' and t.lower().startswith('solver=')))
                current={'keyword':normalized,'original_keyword':line,'data_lines':0}
                digest=hashlib.sha256()
            else:
                if current is None:raise ValueError('Data before INP keyword')
                digest.update((line+'\n').encode('latin1'));current['data_lines']+=1
        if current:current['data_sha256']=digest.hexdigest();records.append(current)
    return {'member':member,'member_sha256':whole.hexdigest(),'sections':records}

def compare(a,b):
    x=sections(a,'model.inp');y=sections(b,'model.inp')
    def physical(r):return [{k:v for k,v in s.items() if k!='original_keyword'} for s in r['sections']]
    return {'schema':'inp_physical_equivalence_v1','status':'PASS' if physical(x)==physical(y) else 'FAIL',
        'evaluation':'Exact section data and normalized keywords; only *Step solver= option excluded.',
        'reference':x,'comparison':y,
        'different_keyword_lines':[{'reference':s['original_keyword'],'comparison':t['original_keyword']}
            for s,t in zip(x['sections'],y['sections']) if s['original_keyword']!=t['original_keyword']]}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('comparison',type=Path)
    a=p.parse_args();r=compare(a.reference,a.comparison);print(json.dumps(r,indent=2));raise SystemExit(r['status']!='PASS')
