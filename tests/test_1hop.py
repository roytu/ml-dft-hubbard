
import numpy as np
from mldfthubbard.hubbard import HubbardInstance

def test_1hop():
    """ Test 1hop
    """

    L = 8
    N_up = 2
    N_down = 2
    t = 1
    U = 1
    hi = HubbardInstance(L=L, N_up=N_up, N_down=N_down, t=t, U=U)
    hi.initialize()

    assert hi._check_1hop(3, 5) == (True, False)
    assert hi._check_1hop(5, 6) == (True, False)
    assert hi._check_1hop(3, 130) == (True, True)
    assert hi._check_1hop(3, 6) == (False, False)



