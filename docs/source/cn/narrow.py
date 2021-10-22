import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.narrow,
    r"""narrow(x, dim, start, length) -> Tensor

    返回一个 :attr:`x` 的缩小版本张量。其形状为在 :attr:`dim` 维度上从 :attr:`start` 开始到 `start + length` 。

    参数：
        - **x** : 要缩小的张量
        - **dim** : 要去进行缩小的维度
        - **start** : 起始维度
        - **length** : 到结束维度的距离

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> flow.narrow(x, 0, 0, 2)
        tensor([[1, 2, 3],
                [4, 5, 6]], dtype=oneflow.int64)
        >>> flow.narrow(x, 1, 1, 2)
        tensor([[2, 3],
                [5, 6],
                [8, 9]], dtype=oneflow.int64)
    """
)
