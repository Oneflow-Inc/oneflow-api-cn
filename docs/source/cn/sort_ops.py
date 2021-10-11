import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.argsort,
    r"""argsort(input, dim=-1, descending=False) -> Tensor

    在指定维度对 :attr:`input` 进行排序并返回排序后的张量的 `index` 。

    参数：
        - **input** (oneflow.Tensor): 输入张量
        - **dim** (int, 可选): 要进行排序的维度，默认为最大维度(-1)。
        - **descending** (bool, 可选): 排序顺序（升序或降序，`False` 为降序）。

    返回类型：
        oneflow.Tensor: 排序后的张量的 `index`

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([[10, 2, 9, 3, 7], [1, 9, 4, 3, 2]], dtype=flow.int32)
        >>> output = flow.argsort(input)
        >>> output
        tensor([[1, 3, 4, 2, 0],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, descending=True)
        >>> output
        tensor([[0, 2, 4, 3, 1],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, dim=0)
        >>> output
        tensor([[1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]], dtype=oneflow.int32)

    """
)
