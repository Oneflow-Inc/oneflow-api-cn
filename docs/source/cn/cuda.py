import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.cuda.BoolTensor,
    r"""
    创造一个 bool 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)

reset_docstr(
    oneflow.cuda.ByteTensor,
    r"""
    创造一个 uint8 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)

reset_docstr(
    oneflow.cuda.CharTensor,
    r"""
    创造一个 int8 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)

reset_docstr(
    oneflow.cuda.DoubleTensor,
    r"""
    创造一个 float64 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)

reset_docstr(
    oneflow.cuda.FloatTensor,
    r"""
    创造一个 float32 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)

reset_docstr(
    oneflow.cuda.HalfTensor,
    r"""
    创造一个 float16 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)
reset_docstr(
    oneflow.cuda.IntTensor,
    r"""
    创造一个 int32 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)

reset_docstr(
    oneflow.cuda.LongTensor,
    r"""
    创造一个 int64 类型张量，它的其余参数与 :func:`oneflow.Tensor` 一致。
    """
)

reset_docstr(
    oneflow.cuda.device_count,
    r"""
    返回可用的 GPU 数量。
    """
)

reset_docstr(
    oneflow.cuda.is_available,
    r"""
    返回一个 bool 用于指代 CUDA 是否可用。
    """
)
