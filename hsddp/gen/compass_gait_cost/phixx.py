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


def phixx(x_t):
    # type: (numpy.ndarray) -> numpy.ndarray
    """
    This function was autogenerated. Do not modify by hand.

    Args:
        x_t: Matrix81

    Outputs:
        phixx: Matrix88
    """

    # Total ops: 37

    # Input arrays
    if x_t.shape == (8,):
        x_t = x_t.reshape((8, 1))
    elif x_t.shape != (8, 1):
        raise IndexError(
            "x_t is expected to have shape (8, 1) or (8,); instead had shape {}".format(x_t.shape)
        )

    # Intermediate terms (8)
    _tmp0 = x_t[2, 0] + 0.05
    _tmp1 = math.exp(3 * _tmp0 - 2)
    _tmp2 = _tmp1 + 1
    _tmp3 = math.exp(2 * x_t[6, 0] + x_t[7, 0] - 2)
    _tmp4 = _tmp3 + 1
    _tmp5 = _tmp3 / _tmp4
    _tmp6 = math.exp(4 * x_t[6, 0] + 2 * x_t[7, 0] - 4) / _tmp4**2
    _tmp7 = 20.0 * _tmp5 - 20.0 * _tmp6

    # Output terms
    _phixx = numpy.zeros((8, 8))
    _phixx[0, 0] = 0
    _phixx[1, 0] = 0
    _phixx[2, 0] = 0
    _phixx[3, 0] = 0
    _phixx[4, 0] = 0
    _phixx[5, 0] = 0
    _phixx[6, 0] = 0
    _phixx[7, 0] = 0
    _phixx[0, 1] = 0
    _phixx[1, 1] = 0
    _phixx[2, 1] = 0
    _phixx[3, 1] = 0
    _phixx[4, 1] = 0
    _phixx[5, 1] = 0
    _phixx[6, 1] = 0
    _phixx[7, 1] = 0
    _phixx[0, 2] = 0
    _phixx[1, 2] = 0
    _phixx[2, 2] = 1575.0 * _tmp1 / _tmp2 + 7200.0 - 1575.0 * math.exp(6 * _tmp0 - 4) / _tmp2**2
    _phixx[3, 2] = 3600.00000000000
    _phixx[4, 2] = 0
    _phixx[5, 2] = 0
    _phixx[6, 2] = 0
    _phixx[7, 2] = 0
    _phixx[0, 3] = 0
    _phixx[1, 3] = 0
    _phixx[2, 3] = 3600.00000000000
    _phixx[3, 3] = 1800.00000000000
    _phixx[4, 3] = 0
    _phixx[5, 3] = 0
    _phixx[6, 3] = 0
    _phixx[7, 3] = 0
    _phixx[0, 4] = 0
    _phixx[1, 4] = 0
    _phixx[2, 4] = 0
    _phixx[3, 4] = 0
    _phixx[4, 4] = 600.000000000000
    _phixx[5, 4] = 0
    _phixx[6, 4] = 0
    _phixx[7, 4] = 0
    _phixx[0, 5] = 0
    _phixx[1, 5] = 0
    _phixx[2, 5] = 0
    _phixx[3, 5] = 0
    _phixx[4, 5] = 0
    _phixx[5, 5] = 0
    _phixx[6, 5] = 0
    _phixx[7, 5] = 0
    _phixx[0, 6] = 0
    _phixx[1, 6] = 0
    _phixx[2, 6] = 0
    _phixx[3, 6] = 0
    _phixx[4, 6] = 0
    _phixx[5, 6] = 0
    _phixx[6, 6] = 40.0 * _tmp5 - 40.0 * _tmp6
    _phixx[7, 6] = _tmp7
    _phixx[0, 7] = 0
    _phixx[1, 7] = 0
    _phixx[2, 7] = 0
    _phixx[3, 7] = 0
    _phixx[4, 7] = 0
    _phixx[5, 7] = 0
    _phixx[6, 7] = _tmp7
    _phixx[7, 7] = 10.0 * _tmp5 - 10.0 * _tmp6
    return _phixx
