
from itertools import combinations
def combinations_binary(L, n):
    """ Return a list of integers whose binary representations are the 
        occupation vectors for 'L choose n', sorted.
    """
    xs = list(combinations(range(L), n))
    bin_xs = sorted([sum([1 << c for c in cs]) for cs in xs])
    return bin_xs

def _check_1hop(self, up, down):
    """ Check if two basis vectors are 1hop away.

        Returns:
            result: True or False
            periodic: if True, whether the hop was over the periodic boundary

        TODO I am so sorry
    """
    def findall(p, s):
        i = s.find(p)
        while i != -1:
            yield i
            i = s.find(p, i+1)

    res = bin(up ^ down)[2:]
    if res.count("1") == 2:
        # Find the indices of the hopping electrons and check they are neighbors
        idxs = list(findall("1", res))
        a, b = idxs[0], idxs[1]
        if abs(a - b) == 1:
            return True, False  # Hop not over periodic boundary
        elif abs(a - b) == self.L - 1:
            return True, True  # Hop over periodic boundary
    return False, False  # Not 1hop
