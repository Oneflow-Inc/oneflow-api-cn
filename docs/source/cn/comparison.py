import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.greater,
    r"""gt(input, other)
    
    返回 :math:`input > other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.Tensor): 输入张量
        - **other** (oneflow.Tensor): 输入张量

    返回类型：
        oneflow.Tensor: 数据类型为 `int8` 的张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.randn(2, 6, 5, 3, dtype=flow.float32)
        >>> input2 = flow.randn(2, 6, 5, 3, dtype=flow.float32)

        >>> out = flow.gt(input1, input2).shape
        >>> out
        oneflow.Size([2, 6, 5, 3])

    """,
)

reset_docstr(
    oneflow.greater_equal,
    r"""ge(input, other)
    
    
    返回 :math:`input >= other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.Tensor): 输入张量
        - **other** (oneflow.Tensor): 输入张量

    返回类型：
        oneflow.Tensor: 数据类型为 `int8` 的张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 1, 4], dtype=flow.float32)

        >>> out = flow.ge(input1, input2)
        >>> out
        tensor([1, 1, 0], dtype=oneflow.int8)

    """,
)
