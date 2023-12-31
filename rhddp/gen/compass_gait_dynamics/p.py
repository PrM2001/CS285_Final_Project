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


def p(x, d):
    # type: (numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    """
    This function was autogenerated. Do not modify by hand.

    Args:
        x: Matrix81
        d: Matrix11

    Outputs:
        p: Matrix81
    """

    # Total ops: 392

    # Input arrays
    if x.shape == (8,):
        x = x.reshape((8, 1))
    elif x.shape != (8, 1):
        raise IndexError(
            "x is expected to have shape (8, 1) or (8,); instead had shape {}".format(x.shape)
        )

    if d.shape == (1,):
        d = d.reshape((1, 1))
    elif d.shape != (1, 1):
        raise IndexError(
            "d is expected to have shape (1, 1) or (1,); instead had shape {}".format(d.shape)
        )

    # Intermediate terms (105)
    _tmp0 = x[2, 0] + x[3, 0]
    _tmp1 = d[0, 0] + 0.9
    _tmp2 = math.cos(_tmp0)
    _tmp3 = _tmp2 + math.cos(x[2, 0])
    _tmp4 = math.sin(_tmp0)
    _tmp5 = _tmp4 + math.sin(x[2, 0])
    _tmp6 = 1 / (-0.0833333333333333 * _tmp3**2 - 0.0833333333333333 * _tmp5**2 + 0.5)
    _tmp7 = 0.166666666666667 * _tmp5
    _tmp8 = -0.0833333333333333 * _tmp2 * _tmp3 - 0.0833333333333333 * _tmp4 * _tmp5 + 0.25
    _tmp9 = _tmp6 * _tmp8**2
    _tmp10 = 1 / (-0.0833333333333333 * _tmp2**2 - 0.0833333333333333 * _tmp4**2 - _tmp9 + 0.25)
    _tmp11 = _tmp6 * _tmp8
    _tmp12 = _tmp11 * _tmp7 - 0.166666666666667 * _tmp4
    _tmp13 = _tmp10 * _tmp12
    _tmp14 = _tmp6 * (-_tmp13 * _tmp8 - _tmp7)
    _tmp15 = 0.5 * _tmp5
    _tmp16 = 0.5 * _tmp4
    _tmp17 = (
        -0.333333333333333 * _tmp13 * _tmp16
        - 0.333333333333333 * _tmp14 * _tmp15
        + 0.333333333333333
    )
    _tmp18 = 1.0 * _tmp10
    _tmp19 = _tmp18 * _tmp4
    _tmp20 = -_tmp11 * _tmp19
    _tmp21 = 1.0 * _tmp4
    _tmp22 = _tmp6 * (_tmp18 * _tmp9 + 1.0)
    _tmp23 = _tmp10 * _tmp16
    _tmp24 = 0.333333333333333 * _tmp11 * _tmp23 - 0.333333333333333 * _tmp15 * _tmp22
    _tmp25 = _tmp20 + _tmp21 * _tmp22 + _tmp24
    _tmp26 = 1.0 * _tmp2
    _tmp27 = 0.166666666666667 * _tmp3
    _tmp28 = _tmp11 * _tmp27 - 0.166666666666667 * _tmp2
    _tmp29 = _tmp10 * _tmp28
    _tmp30 = _tmp6 * (-_tmp27 - _tmp29 * _tmp8)
    _tmp31 = _tmp10 * _tmp11
    _tmp32 = 0.333333333333333 * _tmp15 * _tmp31 - 0.333333333333333 * _tmp23
    _tmp33 = _tmp19 + _tmp20 + _tmp32
    _tmp34 = -0.333333333333333 * _tmp15 * _tmp30 - 0.333333333333333 * _tmp16 * _tmp29
    _tmp35 = _tmp19 * _tmp28 + _tmp21 * _tmp30 + _tmp25 * _tmp26 + _tmp26 * _tmp33 + _tmp34
    _tmp36 = 0.5 * _tmp3
    _tmp37 = 0.5 * _tmp2
    _tmp38 = -0.333333333333333 * _tmp13 * _tmp37 - 0.333333333333333 * _tmp14 * _tmp36
    _tmp39 = _tmp18 * _tmp2
    _tmp40 = -_tmp11 * _tmp39
    _tmp41 = _tmp10 * _tmp37
    _tmp42 = 0.333333333333333 * _tmp11 * _tmp41 - 0.333333333333333 * _tmp22 * _tmp36
    _tmp43 = _tmp22 * _tmp26 + _tmp40 + _tmp42
    _tmp44 = 0.333333333333333 * _tmp31 * _tmp36 - 0.333333333333333 * _tmp41
    _tmp45 = _tmp39 + _tmp40 + _tmp44
    _tmp46 = (
        -0.333333333333333 * _tmp29 * _tmp37
        - 0.333333333333333 * _tmp30 * _tmp36
        + 0.333333333333333
    )
    _tmp47 = 1 / (_tmp26 * _tmp30 + _tmp26 * _tmp43 + _tmp26 * _tmp45 + _tmp28 * _tmp39 + _tmp46)
    _tmp48 = _tmp47 * (
        _tmp12 * _tmp39 + _tmp14 * _tmp26 + _tmp21 * _tmp43 + _tmp21 * _tmp45 + _tmp38
    )
    _tmp49 = _tmp35 * _tmp48
    _tmp50 = 1 / (
        _tmp12 * _tmp19 + _tmp14 * _tmp21 + _tmp17 + _tmp21 * _tmp25 + _tmp21 * _tmp33 - _tmp49
    )
    _tmp51 = _tmp21 * _tmp50
    _tmp52 = _tmp47 * (_tmp49 * _tmp50 + 1)
    _tmp53 = _tmp26 * _tmp52
    _tmp54 = -_tmp48 * _tmp51 + _tmp53
    _tmp55 = _tmp18 * _tmp54
    _tmp56 = -_tmp11 * _tmp55
    _tmp57 = _tmp24 * _tmp50
    _tmp58 = _tmp22 * _tmp54 + _tmp42 * _tmp52 - _tmp48 * _tmp57 + _tmp56
    _tmp59 = _tmp35 * _tmp47
    _tmp60 = -_tmp51 * _tmp59 + _tmp53
    _tmp61 = _tmp26 * _tmp50
    _tmp62 = -_tmp48 * _tmp61 + _tmp51
    _tmp63 = _tmp21 * _tmp62 + _tmp26 * _tmp60
    _tmp64 = _tmp18 * _tmp63
    _tmp65 = -_tmp11 * _tmp64
    _tmp66 = _tmp22 * _tmp63 + _tmp24 * _tmp62 + _tmp42 * _tmp60 + _tmp65
    _tmp67 = 1.0 - _tmp66
    _tmp68 = _tmp51 - _tmp59 * _tmp61
    _tmp69 = _tmp18 * _tmp68
    _tmp70 = -_tmp11 * _tmp69
    _tmp71 = _tmp50 * _tmp59
    _tmp72 = _tmp22 * _tmp68 - _tmp42 * _tmp71 + _tmp57 + _tmp70
    _tmp73 = -_tmp38 * _tmp72 + _tmp42 * _tmp67 - _tmp44 * _tmp66 - _tmp46 * _tmp58
    _tmp74 = 0.5 * _tmp73
    _tmp75 = _tmp32 * _tmp50
    _tmp76 = -_tmp44 * _tmp71 + _tmp69 + _tmp70 + _tmp75
    _tmp77 = _tmp32 * _tmp62 + _tmp44 * _tmp60 + _tmp64 + _tmp65
    _tmp78 = 1.0 - _tmp77
    _tmp79 = _tmp44 * _tmp52 - _tmp48 * _tmp75 + _tmp55 + _tmp56
    _tmp80 = -_tmp38 * _tmp76 - _tmp42 * _tmp77 + _tmp44 * _tmp78 - _tmp46 * _tmp79
    _tmp81 = _tmp34 * _tmp50
    _tmp82 = _tmp29 * _tmp68 + _tmp30 * _tmp68 - _tmp46 * _tmp71 + _tmp81
    _tmp83 = _tmp29 * _tmp63 + _tmp30 * _tmp63 + _tmp34 * _tmp62 + _tmp46 * _tmp60
    _tmp84 = -_tmp29 * _tmp54 - _tmp30 * _tmp54 - _tmp46 * _tmp52 + _tmp48 * _tmp81 + 1.0
    _tmp85 = -_tmp38 * _tmp82 - _tmp42 * _tmp83 - _tmp44 * _tmp83 + _tmp46 * _tmp84
    _tmp86 = 0.25 * _tmp80
    _tmp87 = _tmp17 * _tmp50
    _tmp88 = _tmp13 * _tmp54 + _tmp14 * _tmp54 + _tmp38 * _tmp52 - _tmp48 * _tmp87
    _tmp89 = _tmp13 * _tmp63 + _tmp14 * _tmp63 + _tmp17 * _tmp62 + _tmp38 * _tmp60
    _tmp90 = -_tmp13 * _tmp68 - _tmp14 * _tmp68 + _tmp38 * _tmp71 - _tmp87 + 1.0
    _tmp91 = _tmp38 * _tmp90 - _tmp42 * _tmp89 - _tmp44 * _tmp89 - _tmp46 * _tmp88
    _tmp92 = 0.5 * _tmp91
    _tmp93 = _tmp1 * (
        x[4, 0] * (_tmp3 * _tmp74 + _tmp37 * _tmp80 + 3.0 * _tmp85)
        + x[5, 0] * (_tmp16 * _tmp80 + _tmp5 * _tmp74 + 3.0 * _tmp91)
        + x[6, 0] * (_tmp36 * _tmp85 + _tmp5 * _tmp92 + _tmp74 + _tmp86)
        + x[7, 0] * (_tmp37 * _tmp85 + _tmp4 * _tmp92 + 0.25 * _tmp73 + _tmp86)
    )
    _tmp94 = _tmp93 / _tmp2
    _tmp95 = _tmp11 * _tmp18
    _tmp96 = -_tmp13 * _tmp76 + _tmp18 * _tmp78 - _tmp29 * _tmp79 + _tmp77 * _tmp95
    _tmp97 = 0.5 * _tmp96
    _tmp98 = -_tmp13 * _tmp72 - _tmp18 * _tmp66 - _tmp29 * _tmp58 - _tmp67 * _tmp95
    _tmp99 = 0.5 * _tmp98
    _tmp100 = _tmp18 * _tmp89
    _tmp101 = _tmp100 * _tmp11 - _tmp100 + _tmp13 * _tmp90 - _tmp29 * _tmp88
    _tmp102 = _tmp18 * _tmp83
    _tmp103 = _tmp102 * _tmp11 - _tmp102 - _tmp13 * _tmp82 + _tmp29 * _tmp84
    _tmp104 = 0.25 * _tmp96

    # Output terms
    _p = numpy.zeros(8)
    _p[0] = x[0, 0]
    _p[1] = x[1, 0]
    _p[2] = _tmp0
    _p[3] = -x[3, 0]
    _p[4] = _tmp93
    _p[5] = _tmp4 * _tmp94
    _p[6] = -_tmp94
    _p[7] = -_tmp1 * (
        x[4, 0] * (3.0 * _tmp103 + _tmp2 * _tmp97 + _tmp3 * _tmp99)
        + x[5, 0] * (3.0 * _tmp101 + _tmp4 * _tmp97 + _tmp5 * _tmp99)
        + x[6, 0] * (_tmp101 * _tmp15 + _tmp103 * _tmp36 + _tmp104 + _tmp99)
        + x[7, 0] * (_tmp101 * _tmp16 + _tmp103 * _tmp37 + _tmp104 + 0.25 * _tmp98)
    )
    return _p
