import triton
from triton import language as tl

# TODO: Some ops are already in tl.math,
# when kernel implementations (e.g. FlagGems) switch to tl.math ops,
# remove their extern stubs.


@triton.jit
def div_rn(arg0, arg1):
    return tl.math.div_rn(arg0, arg1)


@triton.jit
def rsqrt(arg0):
    return tl.math.rsqrt(arg0)


@triton.jit
def exp(arg0):
    return tl.math.exp(arg0)


@triton.jit
def erf(arg0):
    return tl.math.erf(arg0)


@tl.core.extern
def div_rz(arg0, arg1, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0, arg1], {
            (tl.float32, tl.float32): ("__vsi_div_rz", tl.float32),
            (tl.float64, tl.float64): ("__vsi_div_rz", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def fmod(arg0, arg1, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0, arg1], {
            (tl.float32, tl.float32): ("__vsi_fmod", tl.float32),
            (tl.float64, tl.float64): ("__vsi_fmod", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def round(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_round", tl.float32),
            (tl.float64, ): ("__vsi_round", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def trunc(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float64, ): ("__vsi_trunc", tl.float64),
            (tl.float32, ): ("__vsi_trunc", tl.float32),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def pow(arg0, arg1, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0, arg1], {
            (tl.int32, tl.int32): ("__vsi_ipowi", tl.int32),
            (tl.float32, tl.int32): ("__vsi_fpowi", tl.float32),
            (tl.float64, tl.int32): ("__vsi_fpowi", tl.float64),
            (tl.float32, tl.float32): ("__vsi_pow", tl.float32),
            (tl.float64, tl.float64): ("__vsi_pow", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def exp2(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_exp2", tl.float32),
            (tl.float64, ): ("__vsi_exp2", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def log2(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_log2", tl.float32),
            (tl.float64, ): ("__vsi_log2", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def log10(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_log10", tl.float32),
            (tl.float64, ): ("__vsi_log10", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def expm1(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_expm1", tl.float32),
            (tl.float64, ): ("__vsi_expm1", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def log1p(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_log1p", tl.float32),
            (tl.float64, ): ("__vsi_log1p", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def tanh(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_tanh", tl.float32),
            (tl.float64, ): ("__vsi_tanh", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def atan2(arg0, arg1, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0, arg1], {
            (tl.float32, tl.float32): ("__vsi_atan2", tl.float32),
            (tl.float64, tl.float64): ("__vsi_atan2", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def atan(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_atan", tl.float32),
            (tl.float64, ): ("__vsi_atan", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def asin(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_asin", tl.float32),
            (tl.float64, ): ("__vsi_asin", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def acos(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_acos", tl.float32),
            (tl.float64, ): ("__vsi_acos", tl.float64),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def isinf(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_isinf", tl.int1),
            (tl.float64, ): ("__vsi_isinf", tl.int1),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def isnan(arg0, _builder=None):
    return tl.core.extern_elementwise(
        "", "", [arg0], {
            (tl.float32, ): ("__vsi_isnan", tl.int1),
            (tl.float64, ): ("__vsi_isnan", tl.int1),
        }, is_pure=True, _builder=_builder)


@tl.core.extern
def finitef(arg0, _builder=None):
    return tl.core.extern_elementwise("", "", [arg0], {
        (tl.float32, ): ("__vsi_isfinite", tl.int1),
    }, is_pure=True, _builder=_builder)


@tl.core.extern
def isfinited(arg0, _builder=None):
    return tl.core.extern_elementwise("", "", [arg0], {
        (tl.float64, ): ("__vsi_isfinite", tl.int1),
    }, is_pure=True, _builder=_builder)
