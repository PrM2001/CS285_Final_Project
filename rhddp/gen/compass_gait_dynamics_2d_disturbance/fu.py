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


def fu(x, u):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    This function was autogenerated. Do not modify by hand.

    Args:
        x: Matrix81
        u: Matrix11

    Outputs:
        fu: Matrix81
    """

    # Total ops: 209

    # Input arrays
    if x.shape == (8,):
        x = x.reshape((8, 1))
    elif x.shape != (8, 1):
        raise IndexError(
            "x is expected to have shape (8, 1) or (8,); instead had shape {}".format(x.shape)
        )

    if u.shape == (1,):
        u = u.reshape((1, 1))
    elif u.shape != (1, 1):
        raise IndexError(
            "u is expected to have shape (1, 1) or (1,); instead had shape {}".format(u.shape)
        )

    # Intermediate terms (60)
    _tmp0 = x[2, 0] + x[3, 0]
    _tmp1 = math.cos(_tmp0)
    _tmp2 = math.sin(_tmp0)
    _tmp3 = math.sin(x[2, 0])
    _tmp4 = _tmp2 + _tmp3
    _tmp5 = math.cos(x[2, 0])
    _tmp6 = _tmp1 + _tmp5
    _tmp7 = -0.0833333333333333 * _tmp1 * _tmp6 - 0.0833333333333333 * _tmp2 * _tmp4 + 0.25
    _tmp8 = 1 / (-0.0833333333333333 * _tmp4**2 - 0.0833333333333333 * _tmp6**2 + 0.5)
    _tmp9 = _tmp7**2 * _tmp8
    _tmp10 = 1 / (-0.0833333333333333 * _tmp1**2 - 0.0833333333333333 * _tmp2**2 - _tmp9 + 0.25)
    _tmp11 = 0.5 * _tmp10
    _tmp12 = _tmp10 * _tmp7
    _tmp13 = _tmp12 * _tmp8
    _tmp14 = 0.5 * _tmp13
    _tmp15 = -0.333333333333333 * _tmp1 * _tmp11 + 0.333333333333333 * _tmp14 * _tmp6
    _tmp16 = 1.0 * _tmp10
    _tmp17 = _tmp8 * (_tmp16 * _tmp9 + 1.0)
    _tmp18 = 1.0 * _tmp5
    _tmp19 = 0.5 * _tmp1
    _tmp20 = 0.5 * _tmp17
    _tmp21 = 0.333333333333333 * _tmp13 * _tmp19 - 0.333333333333333 * _tmp20 * _tmp6
    _tmp22 = _tmp17 * _tmp18 + _tmp21
    _tmp23 = 0.166666666666667 * _tmp6
    _tmp24 = _tmp7 * _tmp8
    _tmp25 = -0.166666666666667 * _tmp1 + _tmp23 * _tmp24
    _tmp26 = _tmp8 * (-_tmp12 * _tmp25 - _tmp23)
    _tmp27 = _tmp10 * _tmp25
    _tmp28 = 0.5 * _tmp26
    _tmp29 = (
        -0.333333333333333 * _tmp19 * _tmp27
        - 0.333333333333333 * _tmp28 * _tmp6
        + 0.333333333333333
    )
    _tmp30 = 1 / (_tmp18 * _tmp22 + _tmp18 * _tmp26 + _tmp29)
    _tmp31 = 0.166666666666667 * _tmp4
    _tmp32 = -0.166666666666667 * _tmp2 + _tmp24 * _tmp31
    _tmp33 = _tmp8 * (-_tmp12 * _tmp32 - _tmp31)
    _tmp34 = 0.5 * _tmp33
    _tmp35 = _tmp10 * _tmp32
    _tmp36 = 0.5 * _tmp2
    _tmp37 = (
        -0.333333333333333 * _tmp34 * _tmp4
        - 0.333333333333333 * _tmp35 * _tmp36
        + 0.333333333333333
    )
    _tmp38 = 1.0 * _tmp3
    _tmp39 = 0.333333333333333 * _tmp13 * _tmp36 - 0.333333333333333 * _tmp20 * _tmp4
    _tmp40 = _tmp17 * _tmp38 + _tmp39
    _tmp41 = -0.333333333333333 * _tmp27 * _tmp36 - 0.333333333333333 * _tmp28 * _tmp4
    _tmp42 = _tmp18 * _tmp40 + _tmp26 * _tmp38 + _tmp41
    _tmp43 = -0.333333333333333 * _tmp19 * _tmp35 - 0.333333333333333 * _tmp34 * _tmp6
    _tmp44 = _tmp30 * (_tmp18 * _tmp33 + _tmp22 * _tmp38 + _tmp43)
    _tmp45 = _tmp42 * _tmp44
    _tmp46 = 1 / (_tmp33 * _tmp38 + _tmp37 + _tmp38 * _tmp40 - _tmp45)
    _tmp47 = _tmp30 * (_tmp45 * _tmp46 + 1)
    _tmp48 = _tmp18 * _tmp47
    _tmp49 = _tmp38 * _tmp46
    _tmp50 = _tmp16 * _tmp24
    _tmp51 = -0.333333333333333 * _tmp11 * _tmp2 + 0.333333333333333 * _tmp14 * _tmp4
    _tmp52 = _tmp46 * _tmp51
    _tmp53 = _tmp15 * _tmp47 - _tmp44 * _tmp52 - _tmp50 * (-_tmp44 * _tmp49 + _tmp48)
    _tmp54 = _tmp30 * _tmp42
    _tmp55 = _tmp48 - _tmp49 * _tmp54
    _tmp56 = _tmp18 * _tmp46
    _tmp57 = -_tmp44 * _tmp56 + _tmp49
    _tmp58 = _tmp15 * _tmp55 - _tmp50 * (_tmp18 * _tmp55 + _tmp38 * _tmp57) + _tmp51 * _tmp57
    _tmp59 = -_tmp15 * _tmp46 * _tmp54 - _tmp50 * (_tmp49 - _tmp54 * _tmp56) + _tmp52

    # Output terms
    _fu = numpy.zeros(8)
    _fu[0] = 0.0
    _fu[1] = 0.0
    _fu[2] = 0.0
    _fu[3] = 0.0
    _fu[4] = (
        0.003 * _tmp15 - 0.003 * _tmp21 * _tmp58 - 0.003 * _tmp29 * _tmp53 - 0.003 * _tmp43 * _tmp59
    )
    _fu[5] = (
        -0.003 * _tmp37 * _tmp59
        - 0.003 * _tmp39 * _tmp58
        - 0.003 * _tmp41 * _tmp53
        + 0.003 * _tmp51
    )
    _fu[6] = (
        -0.003 * _tmp17 * _tmp58
        - 0.003 * _tmp26 * _tmp53
        - 0.003 * _tmp33 * _tmp59
        - 0.003 * _tmp50
    )
    _fu[7] = (
        0.003 * _tmp16 - 0.003 * _tmp27 * _tmp53 - 0.003 * _tmp35 * _tmp59 + 0.003 * _tmp50 * _tmp58
    )
    return _fu