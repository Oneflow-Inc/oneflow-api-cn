import oneflow
from docreset import reset_docstr


reset_docstr(
    oneflow.BoolTensor,
    r"""
    创建一个张量，其数据类型为布尔型，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

reset_docstr(
    oneflow.ByteTensor,
    r"""
    创建一个张量，其数据类型为 uint8，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

reset_docstr(
    oneflow.CharTensor,
    r"""
    创建一个张量，其数据类型为 int8，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

reset_docstr(
    oneflow.DoubleTensor,
    r"""
    创建一个张量，其数据类型为 float64，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

reset_docstr(
    oneflow.FloatTensor,
    r"""
    创建一个张量，其数据类型为 float32，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

reset_docstr(
    oneflow.HalfTensor,
    r"""
    创建一个张量，其数据类型为 float16，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

reset_docstr(
    oneflow.LongTensor,
    r"""
    创建一个张量，其数据类型为 int64，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

reset_docstr(
    oneflow.IntTensor,
    r"""
    创建一个张量，其数据类型为 int32，它的参数与 :func:`oneflow.Tensor` 相同。
    """,
)

