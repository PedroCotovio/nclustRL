import numpy as np
from sklearn.metrics import consensus_score
from nclustgen import BiclusterGenerator as bg


def _index_to_elems(X, bic):

    return [[X[ax][index] for index in axis] for ax, axis in enumerate(bic)]


def match_score(X, fbics, hbics, **kwargs):

    # Enforce Types

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    fbics = [_index_to_elems(X, bic) for bic in fbics]
    hbics = [_index_to_elems(X, bic) for bic in hbics]

    return consensus_score(fbics, hbics, **kwargs)


if __name__ == '__main__':

    # Test
    instance = bg()
    X, y = instance.generate(nclusters=2)
    print(match_score(X, y, y))