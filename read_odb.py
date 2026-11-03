"""Run with Abaqus Python 2.7/3: read all seven packaged ODBs; no solver starts."""
from __future__ import print_function
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from odb_verification import verify_odb


def read(path):
    with open(path, 'rb') as f: return json.loads(f.read().decode('utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if os.path.exists(a.output): p.error('Use a new output file.')
    from odbAccess import openOdb
    results = []
    for row in read(os.path.join(ROOT, 'data', 'engineering', 'cases.json')):
        if row['domain'] != 'cae': continue
        base = os.path.join(ROOT, 'data', 'engineering', row['case_id'])
        audit = read(os.path.join(base, 'independent_odb_audit.json'))
        plan = read(os.path.join(base, 'odb_plan.json'))
        temporary = tempfile.mkdtemp(prefix='tc-odb-')
        try:
            with zipfile.ZipFile(os.path.join(base, 'solver_output.zip')) as z:
                z.extract('model.odb', temporary)
            path = os.path.join(temporary, 'model.odb')
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                while True:
                    block = f.read(1048576)
                    if not block: break
                    h.update(block)
            assert h.hexdigest() == audit['odb_sha256']
            odb = openOdb(path, readOnly=True)
            try: result = verify_odb(odb, plan)
            finally: odb.close()
            assert result['status'] == 'PASS', result
            for key in ('max_displacement_mm', 'max_von_mises_mpa', 'equilibrium_relative_error', 'final_frame_time'):
                assert abs(result[key] - audit[key]) <= max(1e-10, abs(audit[key]) * 1e-8), (key, result[key], audit[key])
            result['case_id'] = row['case_id']
            results.append(result)
            print(row['case_id'] + ': PASS')
        finally: shutil.rmtree(temporary)
    with open(a.output, 'w') as f: json.dump(results, f, indent=2)


if __name__ == '__main__': main()
