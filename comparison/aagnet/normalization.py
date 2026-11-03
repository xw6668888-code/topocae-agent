import numpy as np

def normalize(x, statistics):
    result = dict(x)
    for key in ['node', 'edge']:
        section = statistics[key]
        result[key] = ((x[key].astype(np.float64) - np.asarray(section['mean'])) /
                       np.asarray(section['std'])).astype(np.float32)
    return result
