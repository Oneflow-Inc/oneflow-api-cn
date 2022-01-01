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
    oneflow.Tensor.cpu,
    r"""
    返回此对象在 CPU 内存中的副本。
    
    如果此对象已在 CPU 内存中且位于正确的设备上，则不会执行复制，而是返回原始对象。

    示例:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1, 2, 3, 4, 5], device=flow.device("cuda"))
        >>> output = input.cpu()
        >>> output.device
        device(type='cpu', index=0)
    """
)

reset_docstr(
    oneflow.Tensor.cuda,
    r"""cuda(device=None)

    返回此对象在 CUDA 内存中的副本。

    如果此对象已在 CUDA 内存中且位于正确的设备上，则不会执行复制，而是返回原始对象。

    参数：
        - **device**  (flow.device): 目标 GPU 设备。默认为当前 CUDA 设备。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.Tensor([1, 2, 3, 4, 5])
        >>> output = input.cuda()
        >>> output.device
        device(type='cuda', index=0)

    """
)

reset_docstr(
    oneflow.Tensor.double,
    r"""
    
    `Tensor.double()` 等价于 `Tensor.to(flow.float64)` 。 参见 :mod:`oneflow.to` 。
    
    参数：
        - **input**  (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.randint(0, 5, (3,3))
        >>> input = input.double()
        >>> input.dtype
        oneflow.float64
    """
)

reset_docstr(
    oneflow.Tensor.float,
    r"""float(input)
    
    `Tensor.float()` 等价于 `Tensor.to(flow.float32)` 。 参见 :mod:`oneflow.to` 。

    参数：
        - **input** (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.randint(0, 5, (3,3))
        >>> input = input.float()
        >>> input.dtype
        oneflow.float32
    """
)

reset_docstr(
    oneflow.gather,
    r"""gather(dim, index, sparse_grad=False)

    沿 :attr:`dim` 指定的维度收集值。

    对 3-D tensor ，输出被定义为::

        out[i][j][k] = input[index[i][j][k]][j][k]  # 如果 dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # 如果 dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # 如果 dim == 2

    :attr:`input` 和 :attr:`index` 的维度数必须相同。对于所有 ``d != dim`` 的维度 d ，
    必须有 ``index.size(d) <= input.size(d)`` ， :attr:`out` 的形状和 :attr:`index` 的形状相同。
    请注意， ``input`` 和 ``index`` 不会相互广播。

    参数：
        - **input** (Tensor): 源张量
        - **dim** (int): 索引的维度
        - **index** (LongTensor): 要收集的元素的索引

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.randn(3, 4, 3, 5, dtype=flow.float32)
        >>> index = flow.randint(0, 3, (3, 4, 3, 5))
        >>> output = flow.gather(input, 1, index)
        >>> output.shape
        oneflow.Size([3, 4, 3, 5])
    """
)

reset_docstr(
    oneflow.Tensor.int, 
    r"""
    `Tensor.int()` 等价于 `Tensor.to(flow.int32)` 。 参见 :mod:`oneflow.to` 。

    参数：
        - **input** (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.randn(1, 2, 3, dtype=flow.float32)
        >>> input = input.int()
        >>> input.dtype
        oneflow.int32
    """
)

reset_docstr(
    oneflow.Tensor.item,
    r"""
    将 tensor 的值作为标准 Python 数字返回。仅适用于只有一个元素的 tensor 。
    
    其他情况请参考 :mod:`oneflow.tolist` 。

    这个操作不可导。

    参数：
        - **input** (Tensor): 输入张量

    示例：

    .. code-block:: python


        >>> import oneflow as flow
        >>> x = flow.tensor([1.0])
        >>> x.item()
        1.0
    """
)

reset_docstr(
    oneflow.Tensor.reciprocal,
    r"""reciprocal(x) -> Tensor
    计算 :attr:`x` 的倒数，如果 :attr:`x` 为0，倒数将被设置为 0。

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([[1, 2, 3], [4, 5, 6]], dtype=flow.float32)
        >>> out = flow.reciprocal(x)
        >>> out
        tensor([[1.0000, 0.5000, 0.3333],
                [0.2500, 0.2000, 0.1667]], dtype=oneflow.float32)
    """,
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

