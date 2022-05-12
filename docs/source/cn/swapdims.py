import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow._C.swapdims,
    r"""
    swapdims(input, dim0, dim1) -> Tensor

    这个函数与 torch 的 swapdims 函数等价

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> x
        tensor([[[0, 1],
                 [2, 3]],
        <BLANKLINE>
                [[4, 5],
                 [6, 7]]], dtype=oneflow.int64)
        >>> flow.swapdims(x, 0, 1)
        tensor([[[0, 1],
                 [4, 5]],
        <BLANKLINE>
                [[2, 3],
                 [6, 7]]], dtype=oneflow.int64)
        >>> flow.swapdims(x, 0, 2)
        tensor([[[0, 4],
                 [2, 6]],
        <BLANKLINE>
                [[1, 5],
                 [3, 7]]], dtype=oneflow.int64)

    """
)
