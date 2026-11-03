KAPPA = 20.0
import math
from .priors import source_group

def logit(value):
    if not 0 < value < 1 or not math.isfinite(value):
        raise ValueError("Prior odds must be strictly finite")
    return math.log(value / (1 - value))


def class_source_prior(data, macro, group):
    """Twenty source-specific pseudo-observations shrink rare class cells."""
    source = data["by_source"][group]["jeffreys_mean"]
    cell = data["by_class"][macro][group]
    return (cell["positive"] + KAPPA * source) / (cell["candidates"] + KAPPA)


def calibrated_score(raw_logit, candidate, prior, temperature):
    """Only predicted class/provenance and frozen E1 logit affect inference."""
    if not math.isfinite(raw_logit):
        raise ValueError("Nonfinite frozen E1 object logit")
    group = source_group(candidate)
    value = class_source_prior(prior, candidate["macro_feature"], group)
    return raw_logit / temperature + logit(value) - logit(prior["overall"]["jeffreys_mean"])
