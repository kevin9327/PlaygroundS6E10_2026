"""Summit Composer helpers. Reuse: add this notebook as an input, then `from summit_composer import *`."""
import numpy as np
from scipy.stats import rankdata


def pct(v):
    return (rankdata(np.asarray(v, dtype=np.float64), method='average') - 0.5) / len(v)


def hblend(M, main, corr, mix=(0.3, 0.7)):
    """F.A. Nina's h-blend: sort each row both ways, weight by member + position, mix the two sides (float32 like the original)."""
    n, sides = len(M), []
    for rev in (False, True):
        order = np.argsort(-M if rev else M, axis=1, kind='stable')
        s = np.zeros(n)
        for j in range(M.shape[1]):
            k = order[:, j]
            s += M[np.arange(n), k] * (main[k] + corr[j])
        sides.append(s.astype(np.float32))
    return (np.float32(mix[0]) * sides[0] + np.float32(mix[1]) * sides[1]).astype(np.float32)


def rabari_line(t, RC, V41, V43):
    """KRISH RABARI's line V34 + t*e, rebuilt from his measured points t = 1.25 (V41) and t = 4.5 (V43)."""
    a125, a450 = V41 - RC, V43 - RC
    return pct(RC + a125 + (t - 1.25) / 3.25 * (a450 - a125))


def compose(base3, fourth, teacher, teacher_w, main, corr, gate_amp=0.0):
    """Mix each member with the teacher (share gated by member disagreement), then h-blend."""
    M = np.column_stack(list(base3) + [fourth])
    w = teacher_w + gate_amp * (pct(M.std(axis=1)) - 0.5)
    return pct(hblend(M * (1 - w[:, None]) + w[:, None] * teacher[:, None], np.asarray(main, float), np.asarray(corr, float)))


def peel(public, layer, w, ref):
    """Undo public = pct(w*layer + (1-w)*S) for S (raw scale). The quantiles of the mix come from `ref`, a base close to S."""
    q = np.sort(w * layer + (1 - w) * ref)
    k = np.rint(public * len(q) + 0.5).astype(int) - 1
    return (q[k] - w * layer) / (1 - w)


def layer(base, model_prob, w):
    """Mehran's TabPFN layer: mix the model's probabilities into a rank base, then re-rank."""
    return pct(w * model_prob + (1 - w) * base)
