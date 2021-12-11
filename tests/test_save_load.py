
import numpy as np
from mldfthubbard.ml import run_experiment
from mldfthubbard.results import Results

def test_save_load():
    """ Test saving and loading
    """

    # Save
    res = run_experiment(10, W=0.005)
    name = res.save()

    # Load
    nres = Results()
    nres.load(name)
    nres.show()
