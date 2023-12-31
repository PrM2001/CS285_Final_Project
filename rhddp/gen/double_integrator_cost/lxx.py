# -----------------------------------------------------------------------------
# This file was autogenerated by symforce from template:
#     function/FUNCTION.py.jinja
# Do NOT modify by hand.
# -----------------------------------------------------------------------------

# pylint: disable=too-many-locals,too-many-lines,too-many-statements,unused-argument,unused-import

import math
import typing as T

import numpy

import sym


def lxx(x, u):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    This function was autogenerated. Do not modify by hand.

    Args:
        x: Matrix21
        u: Matrix11

    Outputs:
        lx: Matrix22
    """

    # Total ops: 0

    # Input arrays
    if x.shape == (2,):
        x = x.reshape((2, 1))
    elif x.shape != (2, 1):
        raise IndexError(
            "x is expected to have shape (2, 1) or (2,); instead had shape {}".format(x.shape)
        )

    if u.shape == (1,):
        u = u.reshape((1, 1))
    elif u.shape != (1, 1):
        raise IndexError(
            "u is expected to have shape (1, 1) or (1,); instead had shape {}".format(u.shape)
        )

    # Intermediate terms (0)

    # Output terms
    _lx = numpy.zeros((2, 2))
    _lx[0, 0] = 0.0200000000000000
    _lx[1, 0] = 0
    _lx[0, 1] = 0
    _lx[1, 1] = 0.0200000000000000
    return _lx
