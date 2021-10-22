import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.logical_and,
    r"""logical_and(input, other) -> Tensor
    
    计算给定的输入 tensor 的逐元素逻辑 AND 。
    值为 0 的元素被视作 `False` ，非 0 元素被视作 `True` 。


    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 计算 `and` 逻辑的另一个张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 0, 1], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 1, 0], dtype=flow.float32)

        >>> out = flow.logical_and(input1, input2)
        >>> out
        tensor([1, 0, 0], dtype=oneflow.int8)

    """
)

reset_docstr(
    oneflow.logical_or,
    r"""logical_or(input, other) -> Tensor
    
    计算给定的输入 tensor 的逐元素逻辑 OR 。
    值为 0 的元素被视作 `False` ，非 0 元素被视作 `True` 。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 计算 `or` 逻辑的另一个张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 0, 1], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 0, 0], dtype=flow.float32)

        >>> out = flow.logical_or(input1, input2)
        >>> out
        tensor([1, 0, 1], dtype=oneflow.int8)

    """
)

reset_docstr(
    oneflow.logical_xor, 
    r"""logical_xor(input, other) -> Tensor
    
    计算给定的输入 tensor 的逐元素逻辑 XOR 。
    值为 0 的元素被视作 `False` ，非 0 元素被视作 `True` 。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 计算 `xor` 逻辑的另一个张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python
    
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 0, 1], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 0, 0], dtype=flow.float32)
        >>> out = flow.logical_xor(input1, input2)
        >>> out
        tensor([0, 0, 1], dtype=oneflow.int8)

    """
)
