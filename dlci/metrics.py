# All metrics

from tensorflow.keras import backend as K


def mae(P, T):
    return K.mean(K.abs(P - T))


def _live(index, weight=1):
    def live(P, T):
        return mae(P[..., index], T[..., index]) * weight

    return live


def _dead(index, weight=1):
    def dead(P, T):
        return mae(P[..., index], T[..., index]) * weight

    return dead


# _metrics = [_live, _dead]
_metrics = [_live]


def metrics():
    return [_metrics[index](index) for index in range(len(_metrics))]
