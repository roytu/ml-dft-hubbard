
import numpy as np
from numpy import linalg as LA
from mldfthubbard.hubbard import HubbardInstance

def test_linalg():
    """ Testing LA.eigh against results from wolfram alpha
    """

    # Test 1

    M = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ])

    w, v = LA.eigh(M)
    w_true = np.array([-3.20191, -0.911179, 4.11309])

    assert np.allclose(w, w_true)

    # Test 2

    M = np.array([
        [ 0, -1, -2],
        [-1,  0, -3],
        [-2, -3,  0]
    ])

    w, v = LA.eigh(M)
    w_true = np.array([-4.11309, 0.911179, 3.20191])

    assert np.allclose(w, w_true)


