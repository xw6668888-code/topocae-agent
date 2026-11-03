# -*- coding: ascii -*-
"""Independent frozen-plan ODB audit with Abaqus Python 2 string adaptation."""
from __future__ import print_function

import hashlib
import json
import os
import sys
import traceback


PY2 = sys.version_info[0] < 3
try:
    TEXT_TYPE = unicode
except NameError:
    TEXT_TYPE = str


def native(value):
    """Convert JSON unicode recursively to native Windows byte strings on Py2."""
    if PY2 and isinstance(value, TEXT_TYPE):
        return value.encode('mbcs')
    if isinstance(value, dict):
        return dict((native(key), native(item)) for key, item in value.items())
    if isinstance(value, list):
        return [native(item) for item in value]
    return value


def file_sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with open(path, 'w') as stream:
        json.dump(value, stream, indent=2)


def self_test(output_path):
    loaded = json.loads('{"path":"W:/ascii/final.odb","nested":["value"]}')
    converted = native(loaded)
    before_unicode = PY2 and isinstance(loaded['path'], TEXT_TYPE)
    after_native_str = isinstance(converted['path'], str)
    after_unicode = PY2 and isinstance(converted['path'], TEXT_TYPE)
    nested_native_str = isinstance(converted['nested'][0], str)
    passed = after_native_str and nested_native_str and (not PY2 or (
        before_unicode and not after_unicode
    ))
    result = {
        'schema': 'g10_m6_abaqus_python_string_adapter_smoke_v1',
        'status': 'PASS' if passed else 'FAIL',
        'python_major': int(sys.version_info[0]),
        'python_minor': int(sys.version_info[1]),
        'json_path_type_before': type(loaded['path']).__name__,
        'json_path_type_after': type(converted['path']).__name__,
        'before_unicode': bool(before_unicode),
        'after_native_str': bool(after_native_str),
        'after_unicode': bool(after_unicode),
        'nested_native_str': bool(nested_native_str),
        'truth_used': False,
    }
    write_json(output_path, result)
    print(output_path)
    return 0 if passed else 2


def load_verifier(plan):
    records = []
    for record in plan['binding_files']:
        path = record['path']
        if not os.path.isfile(path) \
                or os.path.getsize(path) != record['bytes'] \
                or file_sha256(path) != record['sha256']:
            raise RuntimeError('Frozen G10 binding file changed: ' + path)
        if os.path.basename(path).lower() == 'abaqus_verification.py':
            records.append(record)
    if len(records) != 1:
        raise RuntimeError('Expected exactly one frozen ODB verifier')
    verifier_path = records[0]['path']
    with open(verifier_path, 'r') as stream:
        source = stream.read()
    namespace = {}
    eval(compile(source, verifier_path, 'exec'), namespace, namespace)
    if 'verify_odb' not in namespace:
        raise RuntimeError('Frozen verifier entry point is absent')
    return namespace['verify_odb'], records[0]


def audit(plan_path, output_path):
    from odbAccess import openOdb

    with open(plan_path, 'r') as stream:
        plan = native(json.load(stream))
    result = {
        'schema': 'g10_independent_odb_audit_v2',
        'status': 'FAIL',
        'truth_used': False,
        'execution_plan_sha256': file_sha256(plan_path),
        'prediction_sha256': plan.get('prediction_sha256'),
        'manifest_sha256': plan.get('manifest_sha256'),
        'geometry_version': plan.get('geometry_version'),
        'string_adapter': 'recursive_json_unicode_to_native_mbcs_on_python2',
        'python_major': int(sys.version_info[0]),
    }
    odb = None
    try:
        if plan.get('schema') != 'g10_e3_cae_execution_plan_v1' \
                or plan.get('truth_used') is not False:
            raise RuntimeError('Invalid or truth-bearing G10 execution plan')
        verify_odb, verifier_record = load_verifier(plan)
        result['verifier'] = verifier_record
        source_path = plan['source_step']
        if file_sha256(source_path) != plan['source_sha256']:
            raise RuntimeError('Source STEP changed after recognition')
        odb_path = os.path.join(plan['run_directory'], 'final.odb')
        result['odb_path'] = odb_path
        if not os.path.isfile(odb_path):
            raise RuntimeError('final.odb is absent')
        odb = openOdb(path=odb_path, readOnly=True)
        result.update(verify_odb(odb, plan))
        result['odb_sha256'] = file_sha256(odb_path)
        result['instance_count'] = len(odb.rootAssembly.instances)
        result['element_count'] = sum(
            len(instance.elements)
            for instance in odb.rootAssembly.instances.values()
        )
        result['node_count'] = sum(
            len(instance.nodes)
            for instance in odb.rootAssembly.instances.values()
        )
        element_types = sorted(set(
            element.type
            for instance in odb.rootAssembly.instances.values()
            for element in instance.elements
        ))
        result['element_types'] = element_types
        if element_types != ['C3D10']:
            result['status'] = 'FAIL'
            result['verification_error'] = 'unexpected_element_types'
    except Exception:
        result['status'] = 'FAIL'
        result['error'] = traceback.format_exc()
    finally:
        if odb is not None:
            odb.close()
    write_json(output_path, result)
    print(output_path)
    return 0 if result.get('status') == 'PASS' else 2


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        return self_test(sys.argv[2])
    if len(sys.argv) != 3:
        print('usage: m6_odb_reaudit_v2.py EXECUTION_PLAN OUTPUT_JSON')
        return 2
    return audit(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    raise SystemExit(main())
