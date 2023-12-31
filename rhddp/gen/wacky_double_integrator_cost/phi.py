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


def phi(x_t):
    # type: (numpy.ndarray) -> numpy.ndarray
    """
    This function was autogenerated. Do not modify by hand.

    Args:
        x_t: Matrix21

    Outputs:
        phi: Matrix11
    """

    # Total ops: 12

    # Input arrays
    if x_t.shape == (2,):
        x_t = x_t.reshape((2, 1))
    elif x_t.shape != (2, 1):
        raise IndexError(
            "x_t is expected to have shape (2, 1) or (2,); instead had shape {}".format(x_t.shape)
        )

    # Intermediate terms (1)
    _tmp0 = x_t[0, 0] - 5

    # Output terms
    _phi = numpy.zeros(1)
    _phi[0] = (
        100 * _tmp0**2 + x_t[1, 0] * (500 * _tmp0 + 100 * x_t[1, 0]) - 1 / abs(x_t[0, 0] - 5)
    )
    return _phi
