import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.tensor,
    r"""
    用数据构造一个张量，如果设置了 :attr:`placement` 和 :attr:`sbp` ，则返回consistent tensor，
        否则返回一个 local tensor 。
       
    参数：
        - **data**: 张量的初始数据。可以是列表、元组、NumPy ndarray、标量或张量。

    关键词参数：
        - **dtype** (oneflow.dtype, 可选)：返回张量的所需数据类型。默认值：如果没有，则从数据推断数据类型。
        - **device** (oneflow.device, 可选)：返回张量的所需设备。如果 placement 和 sbp 为 None，则使用当前 cpu 作为默认设备。
        - **placement** (oneflow.placement, 可选)：设置返回张量的 placement 属性。
        - **sbp** (oneflow.sbp 或 oneflow.sbp 中的元组, 可选)：返回张量的所需 sbp。
        - **requires_grad** (bool, 可选)：如果已经自动求导则记录对返回张量的操作。默认值：False。

    注意：
        关键词参数 device 与 placement 、 sbp 是互斥的。
        consistent tensor只能由张量构造。


    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1,2,3])
        >>> x
        tensor([1, 2, 3], dtype=oneflow.int64)

    """,
)

reset_docstr(
    oneflow.Tensor.atan2,
    r"""
    参考 :func:`oneflow.atan2`
    """
)

reset_docstr(
    oneflow.Tensor.expand,
    r"""
    Tensor.expand() -> Tensor

    参考 :func:`oneflow.expand`
    """
)

reset_docstr(
    oneflow.Tensor.expand_as,
    r"""
    expand_as(other) -> Tensor

    将输入张量扩展到与 :attr:`other` 相同的大小。
    ``self.expand_as(other)`` 等价于 ``self.expand(other.size())`` 。

    更多有关 ``expand`` 的细节请参考 :meth:`~Tensor.expand`。

    参数：
        - **other** (:class:`oneflow.Tensor`): 返回张量与 :attr:`other` 大小相同。
    """,
)

reset_docstr(
    oneflow.Tensor.flatten,
    r"""
    参考 :func:`oneflow.flatten`
    """
)

reset_docstr(
    oneflow.Tensor.floor,
    r"""
    参考 :func:`oneflow.floor`
    """
)

reset_docstr(
    oneflow.Tensor.flip,
    r"""
    参考 :func:`oneflow.flip`
    """,
)

reset_docstr(
    oneflow.Tensor.in_top_k,
    r"""
    Tensor.in_top_k(targets, predictions, k) -> Tensor

    参考 :func:`oneflow.in_top_k`
    """,
)

reset_docstr(
    oneflow.Tensor.index_select,
    r"""
    Tensor.index_select(dim, index) -> Tensor

    参考 :func:`oneflow.index_select`
    """,
)

reset_docstr(
    oneflow.Tensor.numel,
    r"""
    参考 :func:`oneflow.numel`
    """,
)

reset_docstr(
    oneflow.Tensor.new_ones,
    r"""
    Tensor.new_ones() -> Tensor

    参考 :func:`oneflow.new_ones`
    """,
)

reset_docstr(
    oneflow.Tensor.to_global,
    r"""
    Tensor.to_global() -> Tensor

    参考 :func:`oneflow.to_global`
    """,
)

reset_docstr(
    oneflow.Tensor.transpose,
    r"""
    参考 :func:`oneflow.transpose`
    """,
)

reset_docstr(
    oneflow.Tensor.logical_not,
    r"""
    logical_not() -> Tensor
    参考 :func:`oneflow.logical_not`
    """,
)

reset_docstr(
    oneflow.Tensor.std,
    r"""
    参考 :func:`oneflow.std`
    """,
)

reset_docstr(
    oneflow.Tensor.var,
    r"""
    参考 :func:`oneflow.var`
    """,
)

reset_docstr(
    oneflow.Tensor.squeeze,
    r"""
    参考 :func:`oneflow.squeeze`
    """,
)