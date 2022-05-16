import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.greater,
    r"""gt(input, other)
    
    返回 :math:`input > other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor: 数据类型为 `int8` 的张量

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
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor: 数据类型为 `int8` 的张量 

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 1, 4], dtype=flow.float32)

        >>> out = flow.ge(input1, input2)
        >>> out
        tensor([ True,  True, False], dtype=oneflow.bool)

    """,
)

reset_docstr(
    oneflow.eq,
    r"""oneflow.eq(input, other) -> Tensor
    
    返回 :math:`input == other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.tensor): 要去对比的张量
        - **other** (oneflow.tensor, float or int): 对比的目标

    返回类型：
        - oneflow.tensor，元素为 boolean, 若 :attr:`input` 等于 :attr:`other` 则为 True。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([2, 3, 4, 5], dtype=flow.float32)
        >>> other = flow.tensor([2, 3, 4, 1], dtype=flow.float32)

        >>> y = flow.eq(input, other)
        >>> y
        tensor([ True,  True,  True, False], dtype=oneflow.bool)


    """
)

reset_docstr(
    oneflow.lt,
    r"""lt(input, other) -> Tensor

    返回 :math:`input < other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor: 数据类型为 int8 的张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 2, 4], dtype=flow.float32)

        >>> out = flow.lt(input1, input2)
        >>> out
        tensor([False, False,  True], dtype=oneflow.bool)
    
    """
)

reset_docstr(
    oneflow.le,
    r"""le(input, other) -> Tensor

    返回 :math:`input <=other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor: 数据类型为 bool 的张量

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.le(input1, input2)
        >>> out
        tensor([ True, False,  True], dtype=oneflow.bool)
    
    """
)

reset_docstr(
    oneflow.Tensor.ne,
    r"""ne(inout, other) -> Tensor

    计算 element-wise 元素不相等性。
    
    第二个参数 :attr:`other` 可以是一个数字或张量，其形状可以用第一个参数 :attr:`input` 广播。

    参数：
        - **input** (oneflow.Tensor): 要比较的张量
        - **other** (oneflow.Tensor, float 或 int): 要做比较的目标

    返回类型：
        - **oneflow.Tensor** : 一个包含 bool 的张量，如果 :attr:`input` 的元素不等于 :attr:`other` 的元素则为 True 。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([2, 3, 4, 5], dtype=flow.float32)
        >>> other = flow.tensor([2, 3, 4, 1], dtype=flow.float32)

        >>> y = flow.ne(input, other)
        >>> y
        tensor([False, False, False,  True], dtype=oneflow.bool)

    """
)