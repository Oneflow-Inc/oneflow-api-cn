import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.repeat,
    r"""repeat(input, *sizes) -> Tensor
    
    沿指定维度通过重复使 :attr:`input` 尺寸变大，并返回。

    参数：
        - **x** (oneflow.Tensor): 输入张量
        - ***size** (flow.Size 或 int): 沿每个维度重复的次数

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([[[[0, 1]],
        ...                       [[2, 3]],
        ...                       [[4, 5]]]], dtype=flow.int32)
        >>> out = input.repeat(1, 1, 2, 2)
        >>> out
        tensor([[[[0, 1, 0, 1],
                  [0, 1, 0, 1]],
        <BLANKLINE>
                 [[2, 3, 2, 3],
                  [2, 3, 2, 3]],
        <BLANKLINE>
                 [[4, 5, 4, 5],
                  [4, 5, 4, 5]]]], dtype=oneflow.int32)
    """
)
