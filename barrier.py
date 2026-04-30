import warp as wp
from scalar_types import *

dhat = 0.0475 * 2
d2hat = scalar(dhat * dhat)
# @wp.func
# def barrier_derivative(d: scalar) -> scalar:
#     ret = scalar(0.0)
#     if d < d2hat:
#         ret = (d2hat - d) * (scalar(2.0) * wp.log(d / d2hat) + (d - d2hat) / d) / (d2hat * d2hat)

#     return ret


# @wp.func
# def barrier_derivative2(d: scalar) -> scalar:
#     ret = scalar(0.0)
#     if d < d2hat:
#         ret = -(scalar(2.0) * wp.log(d / d2hat) + (d - d2hat) / d + (d - d2hat) * (scalar(2.0) / d + d2hat / (d * d))) / (d2hat * d2hat)
#     return ret


@wp.func
def barrier_derivative(d: scalar) -> scalar:
    ret = scalar(0.0)
    if d < d2hat:
        ret = scalar(2.0) * (d - d2hat)

    return ret

@wp.func 
def barrier_derivative2(d: scalar) -> scalar:
    ret = scalar(0.0)
    if d < d2hat:
        ret = scalar(2.0)
    return ret