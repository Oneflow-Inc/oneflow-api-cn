import oneflow
from docreset import reset_docstr


reset_docstr(
    oneflow.selu,
    r"""
    selu(x) -> Tensor
   
    应用以下 element-wise 公式： 

    :math:`\text{SELU}(x) = scale * (\max(0,x) + \min(0, \alpha * (\exp(x) - 1)))` ,
    :math:`\alpha=1.6732632423543772848170429916717` , 
    :math:`scale=1.0507009873554804934193349852946` 

    更多信息请参考 :class:`~oneflow.nn.SELU` 。

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> out = flow.nn.functional.selu(input)
        >>> out
        tensor([1.0507, 2.1014, 3.1521], dtype=oneflow.float32)
    
    """,
)

reset_docstr(
    oneflow.sigmoid,
    r"""
    sigmoid(input) -> Tensor

    应用以下 element-wise 公式： 
    :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    更多信息请参考 :class:`~oneflow.nn.Sigmoid` 。

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([0.81733328, 0.43621480, 0.10351428], dtype=flow.float32)
        >>> out = flow.nn.functional.sigmoid(input)
        >>> out
        tensor([0.6937, 0.6074, 0.5259], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.nn.Sigmoid,
    r"""
    sigmoid(input) -> Tensor

    应用以下 element-wise 公式： 
    :math:`\text{Sigmoid}(x) = \frac{1}{1 + \exp(-x)}`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([0.81733328, 0.43621480, 0.10351428], dtype=flow.float32)
        >>> m = flow.nn.Sigmoid()
        >>> out = m(x)
        >>> out
        tensor([0.6937, 0.6074, 0.5259], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.silu,
    r"""
    silu(x) -> Tensor

    公式为：

    .. math::

        \text{silu}(x) = x * sigmoid(x)
        
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1, 2, 3], dtype=flow.float32)       
        >>> out = flow.silu(input)
        >>> out
        tensor([0.7311, 1.7616, 2.8577], dtype=oneflow.float32)

    更多信息请参考 :class:`~oneflow.nn.SiLU` 。

    """,
)

reset_docstr(
    oneflow.softmax,
    r"""softmax(input, dim=None) -> Tensor
    
    将 Softmax 函数应用于 n 维 :attr:`input` tensor 。并且重新缩放，
    使输出 tensor 的元素于 [0,1] 范围内并且总和为 1 。

    Softmax 的公式为：

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    当输入张量是稀疏张量时，未指定的值将被视为 ``-inf`` 。

    形状:
        - Input: :math:`(*)` ，其中 `*` 表示任意数量的附加维度
        - Output: :math:`(*)` ，与 :attr:`input` 相同的形状

    返回类型：
        与 :attr:`input` 具有相同维度和形状的张量，其值在 [0, 1] 范围内

    参数：
        - **dim** (int): 要计算 Softmax 的维度（沿 `dim` 的每个切片的总和为 1）

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> m = flow.nn.Softmax(dim = 2)
        >>> x = flow.tensor([[[-0.46716809,  0.40112534,  0.61984003], [-1.31244969, -0.42528763,  1.47953856]]], dtype=flow.float32)
        >>> out = m(x)
        >>> out
        tensor([[[0.1575, 0.3754, 0.4671],
                 [0.0507, 0.1230, 0.8263]]], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.softsign,
    r"""
    softsign(x) -> Tensor 

    公式为：
    
    .. math::  
    
        softsign(x) = \frac{x}{1 + |x|}
    
    示例：
    
    .. code-block:: python
    
        >>> import oneflow as flow

        >>> input = flow.tensor([1, 2, 3], dtype=flow.float32) 
        >>> out = flow.nn.functional.softsign(input)
        >>> out
        tensor([0.5000, 0.6667, 0.7500], dtype=oneflow.float32)
 
    更多细节请参考 :class:`~oneflow.nn.Softsign` 。
    
    """,
)

reset_docstr(
    oneflow.nn.Sigmoid,
    r"""应用逐元素函数：
    
    .. math::
        
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}

    图型：
        - Input: :math:`(N, *)` 其中 `*` 表示任意数量的附加维度
        - Output: :math:`(N, *)` 与输入相同的形状

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([0.81733328, 0.43621480, 0.10351428], dtype=flow.float32)
        >>> m = flow.nn.Sigmoid()
        >>> out = m(x)
        >>> out
        tensor([0.6937, 0.6074, 0.5259], dtype=oneflow.float32)
        """
)
