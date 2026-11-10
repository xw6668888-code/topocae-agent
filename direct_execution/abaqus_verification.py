"""Python 2/3 compatible verification shared by generated and saved-ODB runs."""
import math


def verify_odb(odb, plan):
    try:
        import builtins as builtin
    except ImportError:
        import __builtin__ as builtin
    diagnostic=odb.diagnosticData
    result={'diagnostic_status':str(diagnostic.jobStatus),
            'analysis_errors':int(diagnostic.numberOfAnalysisErrors),
            'analysis_warnings':int(diagnostic.numberOfAnalysisWarnings),'status':'FAIL'}
    if result['diagnostic_status']!='JOB_STATUS_COMPLETED_SUCCESSFULLY':
        result['verification_error']='odb_job_not_completed'
        return result
    if 'Static-Load' not in odb.steps.keys() or not odb.steps['Static-Load'].frames:
        result['verification_error']='completed_step_missing'
        return result
    frame=odb.steps['Static-Load'].frames[-1]
    fields=frame.fieldOutputs
    if not all(name in fields.keys() for name in ('U','S','RF')):
        result['verification_error']='output_fields_missing'
        return result
    reactions=fields['RF'].getSubset(region=odb.rootAssembly.nodeSets['FIXED-FEATURES']).values
    reaction=[builtin.sum(float(v.data[k]) for v in reactions) for k in range(3)]
    expected=plan['expected_force_N']
    residual=[reaction[k]+expected[k] for k in range(3)]
    error=math.sqrt(builtin.sum(v*v for v in residual))/builtin.max(1e-8,plan['force_balance_scale_N'])
    result.update(final_frame_time=float(frame.frameValue),expected_applied_force_N=expected,
                  reaction_force_N=reaction,equilibrium_relative_error=error,
                  max_displacement_mm=builtin.max(float(v.magnitude) for v in fields['U'].values),
                  max_von_mises_mpa=builtin.max(float(v.mises) for v in fields['S'].values))
    if result['analysis_errors'] or builtin.abs(frame.frameValue-1)>1e-8 or error>.005:
        result['verification_error']='final_frame_or_force_balance_failed'
        return result
    result['status']='PASS'
    return result
