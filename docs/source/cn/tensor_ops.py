import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.is_floating_point,
    r"""is_floating_point(input) -> boolean

    如果 :attr:`input` 的数据类型是浮点数据类型，则返回 True 。浮点数据类型为 flow.float64、flow.float32 或者 flow.float16 。

    参数：
        **input** (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1, 2, 3, 4, 5], dtype=flow.int)
        >>> output = flow.is_floating_point(input)
        >>> output
        False
    
    """
)

reset_docstr(
    oneflow.is_nonzero,
    r"""is_nonzero(input) -> Tensor
    
    如果 :attr:`input` 是转换类型后不等于值为 0 的单元素张量的张量
    （ :attr:`flow.tensor([0.])` 或者 :attr:`flow.tensor([0])` ）返回 True 。

    报告 :attr:`RuntimeError` 如果 :attr:`input.shape.numel()!=1`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> flow.is_nonzero(flow.tensor([0.]))
        False
        >>> flow.is_nonzero(flow.tensor([1.5]))
        True
        >>> flow.is_nonzero(flow.tensor([3]))
        True

    """
)

reset_docstr(
    oneflow.negative,
    r"""negative(input) -> Tensor

    返回 :attr:`input` 的负值。

    参数：
        - **input** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1.0, -1.0, 2.3], dtype=flow.float32)
        >>> out = flow.negative(input)
        >>> out
        tensor([-1.0000,  1.0000, -2.3000], dtype=oneflow.float32)

    """
)


reset_docstr(
    oneflow.Tensor.add,
    r"""add(input, other) -> Tensor
    
    计算 `input` 和 `other` 的和。支持 element-wise、标量和广播形式的加法。

    公式为：

    .. math::
        out = input + other

    参数：
        - **input** (Tensor): 输入张量
        - **other** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        # element-wise 加法
        >>> x = flow.randn(2, 3, dtype=flow.float32)
        >>> y = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.add(x, y)
        >>> out.shape
        oneflow.Size([2, 3])

        # 标量加法
        >>> x = 5
        >>> y = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.add(x, y)
        >>> out.shape
        oneflow.Size([2, 3])

        # 广播加法
        >>> x = flow.randn(1, 1, dtype=flow.float32)
        >>> y = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.add(x, y)
        >>> out.shape
        oneflow.Size([2, 3])

    """,
)

reset_docstr(
    oneflow.Tensor.cosh,
    r"""cosh(x) -> Tensor

    返回一个包含 :attr:`x` 中元素的双曲余弦值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \cosh(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([ 0.1632,  1.1835, -0.6979, -0.7325], dtype=flow.float32)
        >>> output = flow.cosh(input)
        >>> output
        tensor([1.0133, 1.7860, 1.2536, 1.2805], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.Tensor.cos,
    r"""cos(x) -> Tensor
    返回一个包含 :attr:`x` 中元素的余弦值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1.4309,  1.2706, -0.8562,  0.9796], dtype=flow.float32)
        >>> output = flow.cos(input)

    """,
)

reset_docstr(
    oneflow.Tensor.div,
    r"""div(input, other) -> Tensor
    
    计算 `input` 除以 `other`，支持 element-wise、标量和广播形式的除法。

    公式为：

    .. math::
        out = \frac{input}{other}
    
    参数：
        - **input** (Union[int, float, oneflow.tensor]): input.
        - **other** (Union[int, float, oneflow.tensor]): other.

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        # element-wise 除法
        >>> input = flow.randn(2, 3, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.div(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 标量除法
        >>> input = 5
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.div(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 广播除法
        >>> input = flow.randn(1, 1, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.div(input,other)
        >>> out.shape 
        oneflow.Size([2, 3])

    """,
)

reset_docstr(
    oneflow.Tensor.le,
    r"""
    返回 :math:`input <= other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor: 数据类型为 int8 的张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 1, 4], dtype=flow.float32)

        >>> out = flow.le(input1, input2)
        >>> out
        tensor([1, 0, 1], dtype=oneflow.int8)

    
    """
)

reset_docstr(
    oneflow.Tensor.log,
    r"""log(x) -> Tensor

    返回一个新 tensor 包含 :attr:`x` 中元素的自然对数。
    公式为：

    .. math::
        y_{i} = \log_{e} (x_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow  

        >>> input = flow.randn(2, 3, 4, 5, dtype=flow.float32)
        >>> output = flow.log(input)

    """,
)

reset_docstr(
    oneflow.Tensor.long,
    r"""long(input) -> Tensor

    `Tensor.long()` 等价于 `Tensor.to(flow.int64)` 。 参考 :func:`oneflow.to` 。

    参数：
        - **input**  (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.randn(1, 2, 3, dtype=flow.float32)
        >>> input = input.long()
        >>> input.dtype
        oneflow.int64
    """
)
