from scipy.stats import beta
import math

def source_group(candidate):
    origins = {(row["branch"], row["stage"]) for row in candidate["provenance"]}
    if ("R", "final") in origins:
        return "R_final"
    if any(branch == "R" for branch, _ in origins):
        return "R_other"
    if origins and all(branch == "Q" for branch, _ in origins):
        return "Q_only"
    raise ValueError("Candidate has no registered R/Q provenance")


def posterior(pos, total):
    """Jeffreys Beta(1/2,1/2) posterior, useful for sparse classes."""
    if not 0 <= pos <= total:
        raise ValueError("Positive count outside candidate denominator")
    a, b = pos + .5, total - pos + .5
    return {"positive": pos, "candidates": total, "negative": total - pos,
            "raw_precision": pos / total if total else None,
            "jeffreys_mean": a / (a + b),
            "credible_95": [float(beta.ppf(.025, a, b)), float(beta.ppf(.975, a, b))]}
