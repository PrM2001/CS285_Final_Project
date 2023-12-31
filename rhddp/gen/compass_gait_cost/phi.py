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
        x_t: Matrix81

    Outputs:
        phi: Matrix11
    """

    # Total ops: 22

    # Input arrays
    if x_t.shape == (8,):
        x_t = x_t.reshape((8, 1))
    elif x_t.shape != (8, 1):
        raise IndexError(
            "x_t is expected to have shape (8, 1) or (8,); instead had shape {}".format(x_t.shape)
        )

    # Intermediate terms (0)

    # Output terms
    _phi = numpy.zeros(1)
    _phi[0] = (
        300.0 * x_t[4, 0] ** 2
        + 900.0 * (2 * x_t[2, 0] + x_t[3, 0]) ** 2
        + 175.0 * math.log(0.157237166313628 * math.exp(3 * x_t[2, 0]) + 1)
        + 10.0 * math.log(math.exp(2 * x_t[6, 0] + x_t[7, 0] - 2) + 1)
    )
    return _phi
