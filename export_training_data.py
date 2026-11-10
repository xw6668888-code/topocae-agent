"""Export the six recorded histories for the author's single Fig. 3 image."""
from pathlib import Path
import argparse,csv,hashlib,json
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'results/reproduced/fig3_data.csv')
    a=p.parse_args();rows=[]
    index=json.loads((ROOT/'data/training_curves/index.json').read_text())
    for branch in ('R','Q'):
        for display,seed in enumerate((20260720,20260719,20260718),1):
            entry=next(r for r in index if r['branch']==branch and r['seed']==seed)
            source=ROOT/entry['history'];assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256']
            history=json.loads(source.read_text());assert len(history)==entry['epochs']
            panel='abcdef'[(0 if branch=='R' else 3)+display-1]
            for r in history:rows.append(dict(panel=panel,branch=branch,display_seed=display,seed=seed,epoch=r['epoch'],
                training_loss=r['train']['loss'],validation_loss=r['val']['loss']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Exported 287 observed epoch rows; no curve interpolation or image replacement.')
if __name__=='__main__':main()
