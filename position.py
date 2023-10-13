from numba import jit
import numpy as np
# import logging
# numba_logger = logging.getLogger('numba')
# numba_logger.setLevel(logging.WARNING)

@jit(nopython=True)
def entity2key(head: int, entity: int):
    ts = 0
    key = '-'.join([str(head), str(entity)])
    return key