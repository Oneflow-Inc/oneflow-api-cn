import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.argsort,
    r"""
    argsort() -> Tensor

    此运算符以指定的 dim 对输入张量进行排序，并返回排序张量的索引。

    参数:
        - **input** (oneflow.Tensor) - 输入张量。
        - **dim** (int, optional) - 要排序的维度。默认为最后一个维度(-1)。
        - **descending** (bool, optional) -  控制排序顺序（升序或降序）。

    返回值类型:
        oneflow.Tensor: 排序张量的索引。

    示例:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([[10, 2, 9, 3, 7],
        ...               [1, 9, 4, 3, 2]]).astype("float32")
        >>> input = flow.Tensor(x)
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