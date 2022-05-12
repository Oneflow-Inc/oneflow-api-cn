import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.add,
    r"""add(input, other) -> oneflow.Tensor
    
    计算 `input` 和 `other` 的和。支持 element-wise 、标量和广播形式的加法。

    公式为：

    .. math::
        out = input + other

    参数：
        - **input** (Tensor) - 输入张量
        - **other** (Tensor) - 其余输入张量

    返回类型：
        oneflow.Tensor

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
    oneflow.sub,
    r"""sub(input, other) -> Tensor
    
    计算 `input` 和 `other` 的差，支持 element-wise、标量和广播形式的减法。

    公式为：

    .. math::
        out = input - other

    参数：
        - **input** (Tensor): 输入张量
        - **other** (Tensor): 输入张量

    返回类型：   
        oneflow.tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        # element-wise 减法
        >>> input = flow.randn(2, 3, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.sub(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 标量减法
        >>> input = 5
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.sub(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 广播减法
        >>> input = flow.randn(1, 1, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.sub(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

    """,
)

reset_docstr(
    oneflow.abs,
    r"""abs(x) -> Tensor
    
    返回一个包含 `x` 中每个元素的绝对值的tensor:`y = |x|`。
    
    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([-1, 2, -3, 4], dtype=flow.float32)
        >>> flow.abs(x)
        tensor([1., 2., 3., 4.], dtype=oneflow.float32)
    
    """,
)

reset_docstr(
    oneflow.div,
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
    oneflow.mul,
    r"""mul(input, other) -> Tensor
    
    计算 `input` 与 `other` 相乘，支持 element-wise、标量和广播形式的乘法。
    
    公式为：

    .. math::
        out = input \times other

    参数：
        - **input** (Tensor): 输入张量。
        - **other** (Tensor): 输入张量。

    返回类型：
        oneflow.tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        # element-wise 乘法
        >>> input = flow.randn(2, 3, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.mul(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 标量乘法
        >>> input = 5
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.mul(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 广播乘法
        >>> input = flow.randn(1, 1, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.mul(input,other)
        >>> out.shape 
        oneflow.Size([2, 3])

    """,
)

reset_docstr(
    oneflow.reciprocal,
    r"""reciprocal(x) -> Tensor
    计算 :attr:`x` 的倒数，如果 :attr:`x` 为0，倒数将被设置为0。

    参数：
        **x** (Tensor): 输入张量。

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
    oneflow.asin,
    r"""arcsin(x) -> Tensor

    返回一个新的 tensor 包含 :attr:`x` 中每个元素的反正弦。

    公式为：

    .. math::
        \text{out}_{i} = \sin^{-1}(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([-0.5,  0.8, 1.0,  -0.8], dtype=flow.float32)
        >>> output = flow.asin(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output
        tensor([-0.5236,  0.9273,  1.5708, -0.9273], dtype=oneflow.float32)
        >>> input1 = flow.tensor([[0.8, 1.0], [-0.6, -1.0]], dtype=flow.float32)
        >>> output1 = input1.asin()
        >>> output1.shape
        oneflow.Size([2, 2])
        >>> output1
        tensor([[ 0.9273,  1.5708],
                [-0.6435, -1.5708]], dtype=oneflow.float32)
    """,
)

reset_docstr(
    oneflow.asinh,
    r"""arcsinh(x) -> Tensor
    
    返回一个包含 :attr:`x` 中每个元素的反双曲正弦的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \sinh^{-1}(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    关键词参数：
        **out** (Tensor, optional): 输出张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([2, 3, 4], dtype=flow.float32)
        >>> output = flow.asinh(input)
        >>> output.shape
        oneflow.Size([3])
        >>> output
        tensor([1.4436, 1.8184, 2.0947], dtype=oneflow.float32)

        >>> input1 = flow.tensor([[-1, 0, -0.4], [5, 7, 0.8]], dtype=flow.float32)
        >>> output1 = input1.asinh()
        >>> output1.shape
        oneflow.Size([2, 3])
        >>> output1
        tensor([[-0.8814,  0.0000, -0.3900],
                [ 2.3124,  2.6441,  0.7327]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.atan,
    r"""arctan(x) -> Tensor

    返回一个包含 :attr:`x` 中所有元素的反正切的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \tan^{-1}(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python
    
        >>> import oneflow as flow

        >>> input = flow.tensor([0.5, 0.6, 0.7], dtype=flow.float32)
        >>> output = flow.atan(input)
        >>> output.shape
        oneflow.Size([3])
        
    """,
)

reset_docstr(
    oneflow.ceil,
    r"""ceil(x) -> Tensor
    
    返回一个新的 tensor，tensor 中元素为大于或等于 :attr:`x` 中元素的最小整数。

    公式为： 

    .. math::
        \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1

    参数：
        **x** (oneflow.tensor): 张量

    返回类型：
        oneflow.tensor

    示例： 


    .. code-block:: python 
        
        >>> import oneflow as flow

        >>> x = flow.tensor([0.1, -2, 3.4], dtype=flow.float32)
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1., -2.,  4.], dtype=oneflow.float32)
        >>> x = flow.tensor([[2.5, 4.6, 0.6],[7.8, 8.3, 9.2]], dtype=flow.float32)
        >>> y = x.ceil()
        >>> y.shape
        oneflow.Size([2, 3])
        >>> y
        tensor([[ 3.,  5.,  1.],
                [ 8.,  9., 10.]], dtype=oneflow.float32)
        >>> x = flow.tensor([[[2.2, 4.4, 6.5],[7.1, 8.2, 9.3]],[[10.6,11.2,12.2],[13.5,14.8,15.9]]], dtype=flow.float32)
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([2, 2, 3])
        >>> y
        tensor([[[ 3.,  5.,  7.],
                 [ 8.,  9., 10.]],
        <BLANKLINE>
                [[11., 12., 13.],
                 [14., 15., 16.]]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.log1p,
    r"""log1p(x) -> Tensor
    
    返回一个新的 tensor，其自然对数的公式为 (1 + x)。

    .. math::
        \text{out}_{i}=\log_e(1+\text{input}_{i})

    参数：
        **x** (Tensor): 张量
    
    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor([1.3, 1.5, 2.7], dtype=flow.float32)
        >>> out = flow.log1p(x)
        >>> out
        tensor([0.8329, 0.9163, 1.3083], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.exp,
    r"""exp(x) -> Tensor

    此运算符计算 :attr:`x` 的指数。

    公式为：

    .. math::

        out = e^x

    参数：
        **x** (Tensor): 张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> y = flow.exp(x)
        >>> y
        tensor([ 2.7183,  7.3891, 20.0855], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.acos,
    r"""acos(x) -> Tensor

    返回一个包含 :attr:`x` 中元素的反余弦值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \arccos(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([0.5, 0.6, 0.7], dtype=flow.float32)
        >>> output = flow.acos(input)
        >>> output
        tensor([1.0472, 0.9273, 0.7954], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.acosh,
    r"""acosh(x) -> Tensor

    返回具有 :attr:`x` 中元素的反双曲余弦的新 tensor。

    公式为：

    .. math::

        \text{out}_{i} = \cosh^{-1}(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：
    
    .. code-block:: python

        >>> import oneflow as flow

        >>> x1 = flow.tensor([2, 3, 4], dtype=flow.float32)
        >>> out1 = flow.acosh(x1)
        >>> out1
        tensor([1.3170, 1.7627, 2.0634], dtype=oneflow.float32)
        >>> x2 = flow.tensor([1.5, 2.6, 3.7], dtype=flow.float32, device=flow.device('cuda'))
        >>> out2 = flow.acosh(x2)
        >>> out2
        tensor([0.9624, 1.6094, 1.9827], device='cuda:0', dtype=oneflow.float32)

    """,
)


reset_docstr(
    oneflow.atanh,
    r"""arctanh(x) -> Tensor
    
    返回一个包含 :attr:`x` 中元素的反双曲正切值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \tanh^{-1}(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([0.5, 0.6, 0.7], dtype=flow.float32)
        >>> output = flow.atanh(input)
        >>> output
        tensor([0.5493, 0.6931, 0.8673], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.sign,
    r"""sign(x) -> Tensor
    
    求 `x` 中元素的正负。

    公式为：

    .. math::

        \text{out}_{i}  = \text{sgn}(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x1 = flow.tensor([-2, 0, 2], dtype=flow.float32)
        >>> out1 = flow.sign(x1)
        >>> out1
        tensor([-1.,  0.,  1.], dtype=oneflow.float32)
        >>> x2 = flow.tensor([-3.2, -4.5, 5.8], dtype=flow.float32, device=flow.device('cuda'))
        >>> out2 = flow.sign(x2)
        >>> out2
        tensor([-1., -1.,  1.], device='cuda:0', dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.sinh,
    r"""sinh(x) -> Tensor

    返回一个包含 :attr:`x` 中元素的双曲正弦值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \sinh(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x1 = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> x2 = flow.tensor([1.53123589,0.54242598,0.15117185], dtype=flow.float32)
        >>> x3 = flow.tensor([1,0,-1], dtype=flow.float32)

        >>> flow.sinh(x1)
        tensor([ 1.1752,  3.6269, 10.0179], dtype=oneflow.float32)
        >>> flow.sinh(x2)
        tensor([2.2038, 0.5694, 0.1517], dtype=oneflow.float32)
        >>> flow.sinh(x3)
        tensor([ 1.1752,  0.0000, -1.1752], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.tan,
    r"""tan(x) -> Tensor
    
    返回一个包含 :attr:`x` 中元素的正切值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \tan(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import math as m
        >>> import oneflow as flow

        >>> input = flow.tensor([-1/4*m.pi, 0, 1/4*m.pi], dtype=flow.float32)
        >>> output = flow.tan(input)
        >>> output
        tensor([-1.,  0.,  1.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.clamp,
    r"""clamp(input, min = None, max = None) -> Tensor

    返回新的结果 tensor，结果 tensor 中将 :attr:`input` 中元素限制在范围 `[` :attr:`min`, :attr:`max` `]` 中。

    公式为：

    .. math::
        y_i = \begin{cases}
            \text{min} & \text{if } x_i < \text{min} \\
            x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
            \text{max} & \text{if } x_i > \text{max}
        \end{cases}

    如果 :attr:`input` 的类型是 `FloatTensor` 或 `FloatTensor`，参数 :attr:`min` 
    和 :attr:`max` 必须为实数， 如果 :attr:`input` 为其它类型的 tensor，参数 
    :attr:`min` 和 :attr:`max` 必须为 `integer`。

    参数：
        - **input** (Tensor): 输入张量
        - **min** (Number): 要限制到的范围的下限，默认为None
        - **max** (Number): 要限制到的范围的上限，默认为None

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([0.2, 0.6, -1.5, -0.3], dtype=flow.float32)
        >>> output = flow.clamp(input, min=-0.5, max=0.5)
        >>> output
        tensor([ 0.2000,  0.5000, -0.5000, -0.3000], dtype=oneflow.float32)

        >>> input = flow.tensor([0.2, 0.6, -1.5, -0.3], dtype=flow.float32)
        >>> output = flow.clamp(input, min=None, max=0.5)
        >>> output
        tensor([ 0.2000,  0.5000, -1.5000, -0.3000], dtype=oneflow.float32)

        >>> input = flow.tensor([0.2, 0.6, -1.5, -0.3], dtype=flow.float32)
        >>> output = flow.clamp(input, min=-0.5, max=None)
        >>> output
        tensor([ 0.2000,  0.6000, -0.5000, -0.3000], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.clip,
    r"""
    函数 :func:`oneflow.clamp` 的别名. 
    """,
)


reset_docstr(
    oneflow.cos,
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
    oneflow.cosh,
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
    oneflow.erf,
    r"""erf(x) -> Tensor
    
    计算每个元素的误差函数。误差函数定义如下：

    .. math::
            \operatorname{erf}(x)=\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t

    参数：
        **x** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor
               
    示例

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([0, -1., 10.], dtype=flow.float32)
        >>> out = flow.erf(x)
        >>> out.shape
        oneflow.Size([3])
        >>> out
        tensor([ 0.0000, -0.8427,  1.0000], dtype=oneflow.float32)
        >>> x = flow.tensor([[0, -1., 10.], [5, 7, 0.8]], dtype=flow.float32)
        >>> out = flow.erf(x)
        >>> out.shape
        oneflow.Size([2, 3])
        >>> out
        tensor([[ 0.0000, -0.8427,  1.0000],
                [ 1.0000,  1.0000,  0.7421]], dtype=oneflow.float32)
        >>> x = flow.tensor([[0, -1., 10.], [5, 7, 0.8], [2, 3, 4]], dtype=flow.float32)
        >>> out = x.erf()
        >>> out.shape
        oneflow.Size([3, 3])
        >>> out
        tensor([[ 0.0000, -0.8427,  1.0000],
                [ 1.0000,  1.0000,  0.7421],
                [ 0.9953,  1.0000,  1.0000]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.erfc,
    r"""erfc(x) -> Tensor
    
    计算 :attr:`x` 的每个元素的互补误差函数。互补误差函数定义如下：
    
    .. math::
            \operatorname{erfc}(x)=1-\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t

    参数：
        **x** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([0, -1., 10.], dtype=flow.float32)
        >>> out = flow.erfc(x)
        >>> out
        tensor([1.0000e+00, 1.8427e+00, 2.8026e-45], dtype=oneflow.float32)

        >>> x = flow.tensor([[0, -1., 10.], [5, 7, 0.8]], dtype=flow.float32)
        >>> out = flow.erfc(x)
        >>> out
        tensor([[1.0000e+00, 1.8427e+00, 2.8026e-45],
                [1.5375e-12, 4.1838e-23, 2.5790e-01]], dtype=oneflow.float32)
        
    """,
)

reset_docstr(
    oneflow.expm1,
    r"""expm1(x) -> Tensor
    
    返回一个新的张量，其元素为 :attr:`x` 的元素指数减去 1。 

    公式为：

    .. math::
        y_{i} = e^{x_{i}} - 1

    参数：
        **x** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python 
        
        >>> import oneflow as flow
        
        >>> x = flow.tensor([0, -1., 10.], dtype=flow.float32)
        >>> out = flow.erf(x)
        >>> out.shape
        oneflow.Size([3])
        >>> out
        tensor([ 0.0000, -0.8427,  1.0000], dtype=oneflow.float32)
        >>> x = flow.tensor([[0, -1., 10.], [5, 7, 0.8]], dtype=flow.float32)
        >>> out = flow.erf(x)
        >>> out.shape
        oneflow.Size([2, 3])
        >>> out
        tensor([[ 0.0000, -0.8427,  1.0000],
                [ 1.0000,  1.0000,  0.7421]], dtype=oneflow.float32)
        >>> x = flow.tensor([[0, -1., 10.], [5, 7, 0.8], [2, 3, 4]], dtype=flow.float32)
        >>> out = x.erf()
        >>> out.shape
        oneflow.Size([3, 3])
        >>> out
        tensor([[ 0.0000, -0.8427,  1.0000],
                [ 1.0000,  1.0000,  0.7421],
                [ 0.9953,  1.0000,  1.0000]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.fmod,
    r"""fmod(input, other) -> Tensor

    计算逐元素余数。

    被除数和除数可能同时包含整数和浮点数。余数与被除数 :attr:`input` 同号。

    支持广播到通用形状、整数和浮点输入。

    参数：
        - **input** (Tensor): 被除数
        - **other** (Tensor or Scalar): 除数
    
    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python 

        >>> import oneflow as flow

        >>> flow.fmod(flow.tensor([-3., -2, -1, 1, 2, 3], dtype=flow.float32), 2.)
        tensor([-1., -0., -1.,  1.,  0.,  1.], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4, 5.], dtype=flow.float32), 1.5)
        tensor([1.0000, 0.5000, 0.0000, 1.0000, 0.5000], dtype=oneflow.float32)
        >>> flow.fmod(flow.tensor([1, 2, 3, 4., -5]), flow.tensor([4, 2, 1, 3., 1]))
        tensor([1., 0., 0., 1., -0.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.log,
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
    oneflow.minimum,
    r"""minimum(x, y) -> Tensor
    
    计算 `x` 和 `y` 的 element-wise 最小值。

    参数：
        - **x** (Tensor): 输入张量
        - **y** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor((1, 2, -1), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.minimum(x, y)
        tensor([ 1.,  0., -1.], dtype=oneflow.float32)

        >>> x = flow.tensor((1,), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.minimum(x, y)
        tensor([1., 0., 1.], dtype=oneflow.float32)
    """,
)

reset_docstr(
    oneflow.maximum,
    r"""maximum(x, y) -> Tensor

    计算 `x` 和 `y` 的 element-wise 最大值。

    参数：
        - **x** (Tensor): 输入张量
        - **y** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.tensor((1, 2, -1), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.maximum(x, y)
        tensor([3., 2., 4.], dtype=oneflow.float32)

        >>> x = flow.tensor((1,), dtype=flow.float32)
        >>> y = flow.tensor((3, 0, 4), dtype=flow.float32)
        >>> flow.maximum(x, y)
        tensor([3., 1., 4.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.pow,
    r"""pow(input, exponent) -> Tensor
    
    返回一个Tensor, 其元素为用 `exponent` 计算 `input` 中
    每​​个元素的幂。`exponent` 可以是单个浮点数，整数或者与 
    `input` 具有相同形状的 tensor。

    当指数是标量时，操作为：

    .. math::
        \text{out}_i = x_i ^ \text{exponent}

    当指数是张量时：

    .. math::
        \text{out}_i = x_i ^ {\text{exponent}_i}

    参数：
        - **input** (Tensor): 输入张量
        - **exponent** (int, float, Tensor): 指数

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=flow.float32)
        >>> out = flow.pow(x, 2)
        >>> out
        tensor([ 1.,  4.,  9., 16., 25., 36.], dtype=oneflow.float32)

        >>> x = flow.tensor([1.0, 2.0, 3.0, 4.0], dtype=flow.float32)
        >>> y = flow.tensor([1.0, 2.0, 3.0, 4.0], dtype=flow.float32)
        >>> out = flow.pow(x, y)
        >>> out
        tensor([  1.,   4.,  27., 256.], dtype=oneflow.float32)
        
    """,
)

reset_docstr(
    oneflow.rsqrt,
    r""" rsqrt(x) -> Tensor
    
    返回一个新的 Tensor, 其元素为 :attr:`x` 中元素的平方根的倒数

    公式为：

    .. math::
        \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

    参数：
        **x** (Tensor): 输入张量
    
    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python
        >>> import oneflow as flow
            
        >>> a = flow.tensor([1.0, 2.0, 3.0], dtype=flow.float32)
        >>> out = flow.rsqrt(a)
        >>> out
        tensor([1.0000, 0.7071, 0.5774], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.square,
    r"""square(x)  -> Tensor

    返回一个新的张量，其元素为 :attr:`x` 中元素的的平方。

    .. math::
        \text{out}_{i} = \sqrt{\text{input}_{i}}

    参数：
        **x** (Tensor): 输入张量
    
    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
            
        >>> input = flow.tensor([1.0, 2.0, 3.0], dtype=flow.float32)
        >>> output = flow.square(input)
        >>> output
        tensor([1., 4., 9.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.matmul,
    r"""matmul(a, b) -> Tensor

    此运算符将矩阵乘法应用于两个 Tensor :attr:`a` 和 :attr:`b`。

    参数：
        - **a** (oneflow.tensor): 张量
        - **b** (oneflow.tensor): 张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input1 = flow.randn(2, 6, dtype=flow.float32)
        >>> input2 = flow.randn(6, 5, dtype=flow.float32)
        >>> of_out = flow.matmul(input1, input2)
        >>> of_out.shape
        oneflow.Size([2, 5])

    """,
)

reset_docstr(
    oneflow.round,
    r"""round(x)
    
    返回一个新 tensor，其元素为 :attr:`x` 中元素四舍五入到整数。

    参数：
        **x** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x1 = flow.tensor([1.49999, 1.500001, 2.7], dtype=flow.float32)
        >>> out1 = flow.round(x1)
        >>> out1
        tensor([1., 2., 3.], dtype=oneflow.float32)
        >>> x2 = flow.tensor([2.499999, 7.5000001, 5.3, 6.8], dtype=flow.float32)
        >>> out2 = flow.round(x2)
        >>> out2
        tensor([2., 8., 5., 7.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.addmm,
    r"""addmm(beta=1, input, alpha=1, mat1, mat2, out=None) -> Tensor
    
    对 :attr:`mat1` 和 :attr:`mat2` 进行矩阵乘法，并且将结果与 :attr:`input` 相加求和后，返回计算结果。
    
    如果 :attr:`mat1` 是一个 :math:`(n \times m)` 张量，同时 :attr:`mat2` 是一个 :math:`(m \times p)` 张量，
    则 :attr:`input` 必须是可广播为 `(n \times p)` 的张量，:attr:`out` 也必须为 :math:`(n \times p)` 的张量。

    公式为：

    :attr:`alpha` 是 :attr:`mat1` 和 :attr:`mat2` 的矩阵向量乘积的缩放比例因数，
    :attr:`beta` 是 :attr:`input` 的因数

    .. math::
        \text{out} = \beta\ \text{input} + \alpha\ (\text{mat1}_i \mathbin{@} \text{mat2}_i)

    如果 :attr:`input` 的类型为 `float` 或 `double`，
    参数 :attr:`beta` 和 :attr:`alpha` 应为实数，否则只能是整数(integers)。

    参数：
        - **beta** (Number, 可选): :attr:`input` 的因数 (:math:`\beta`)
        - **input** (Tensor): 作为加数的矩阵
        - **alpha** (Number, 可选): :math:`mat1 \mathbin{@} mat2` 的因数 (:math:`\alpha`)
        - **mat1** (Tensor): 作为第一个乘数的矩阵
        - **mat2** (Tensor): 作为第二个乘数的矩阵
        - **out** (Tensor, 可选): 输出张量

    返回类型：
        oneflow.tensor

    示例：
    
        >>> import oneflow as flow
        >>> input = flow.tensor([[1,2,4],[5,11,9.1]], dtype=flow.float64)
        >>> mat1 = flow.tensor([[7.3,1.9,7.3],[10.2,1,5.5]], dtype=flow.float64)
        >>> mat2 = flow.tensor([[7.3,1.9,7.3],[10.2,1,5.5],[3.7,2.2,8.1]], dtype=flow.float64)
        >>> output = flow.addmm(input, mat1, mat2)
        >>> output
        tensor([[100.6800,  33.8300, 126.8700],
                [110.0100,  43.4800, 133.6100]], dtype=oneflow.float64)
        >>> output.shape
        oneflow.Size([2, 3])

        >>> input2 = flow.tensor([1.7], dtype=flow.float64)
        >>> mat1 = flow.tensor([[1,2],[5,9.1],[7.7,1.4]], dtype=flow.float64)
        >>> mat2 = flow.tensor([[1,2,3.7],[5,9.1,6.8]], dtype=flow.float64)
        >>> output2 = flow.addmm(input2, mat1, mat2, alpha=1, beta=2)
        >>> output2
        tensor([[14.4000, 23.6000, 20.7000],
                [53.9000, 96.2100, 83.7800],
                [18.1000, 31.5400, 41.4100]], dtype=oneflow.float64)
        >>> output2.shape
        oneflow.Size([3, 3])
    
    """
)

reset_docstr(
    oneflow.argmax,
    r"""argmax(input, dim=-1, keepdim=False) -> Tensor

    返回 :attr:`input` 在指定维度上的最大值的 `index` 。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **dim** (int, 可选): 要计算的维度，默认为最大维度(-1)。
        - **keepdim** (bool，可选的): 返回值是否保留 input 的原有维数。默认为 False 。

    返回类型：
        oneflow.tensor: 包含 :attr:`input` 特定维度最大值的 index 的新张量(dtype=int64)。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[1, 3, 8, 7, 2],
        ...            [1, 9, 4, 3, 2]], dtype=flow.float32)
        >>> output = flow.argmax(input)
        >>> output
        tensor(6, dtype=oneflow.int64)
        >>> output = flow.argmax(input, dim=1)
        >>> output
        tensor([2, 1], dtype=oneflow.int64)
    
    """

)

reset_docstr(
    oneflow.floor,
    r"""floor(input) -> Tensor

    返回一个新 tensor ，其元素为对 :attr:`input` 向下取整的结果。

    .. math::
        \text{out}_{i} = \lfloor \text{input}_{i} \rfloor

    参数:
        **input** (Tensor): 输入张量

    返回类型：
        oneflow.tensor
        
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([-0.5,  1.5, 0,  0.8], dtype=flow.float32)
        >>> output = flow.floor(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output
        tensor([-1.,  1.,  0.,  0.], dtype=oneflow.float32)
        
        >>> input1 = flow.tensor([[0.8, 1.0], [-0.6, 2.5]], dtype=flow.float32)
        >>> output1 = input1.floor()
        >>> output1.shape
        oneflow.Size([2, 2])
        >>> output1
        tensor([[ 0.,  1.],
                [-1.,  2.]], dtype=oneflow.float32)
    
    """
)

reset_docstr(
    oneflow.full,
    r"""full(size, value, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor
    
    创建并返回一个大小为 :attr:`size` ，其元素全部为 :attr:`value` 的 tensor。此 tensor 的数据类型与 `value` 相同。

    参数：
        - **size** (int...): 列表，元组或者描述输出张量的整数 torch.Size
        - **fill_value** (Number): 用于填充输出张量的值
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型
        - **device** (flow.device, 可选): 返回张量的所需设备。如果为 None ，使用当前设备
        - **placement** (flow.placement, 可选): 设置返回张量的 placement 属性。如果为 None ，则返回的张量是使用参数 `device` 的本地张量。
        - **sbp** (flow.sbp.sbp 或 flow.sbp.sbp 的元组, 可选): 返回的consistent tensor的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量。
        - **requires_grad** (bool, 可选): 使用 autograd 记录对返回张量的操作。默认值： `False` 。

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.full((5,),5) 
        >>> y
        tensor([5, 5, 5, 5, 5], dtype=oneflow.int64)
        >>> y = flow.full((2,3),5.0) # 构造 local tensor
        >>> y
        tensor([[5., 5., 5.],
                [5., 5., 5.]], dtype=oneflow.float32)
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> y = flow.full((2,3),5.0, placement=placement, sbp=flow.sbp.broadcast)  # 构造 global tensor
        >>> y.is_global # doctest: +NORMALIZE_WHITESPACE
        True

    """
)

reset_docstr(
    oneflow.gelu,
    r"""gelu(x) -> Tensor

    Gelu 激活算子.

    公式为：

    .. math::
        out = 0.5 * x * (1 + tanh(\sqrt{\frac{2}{\pi}} * (x + 0.044715x^{3})))

    参数：
        **x** (oneflow.tensor): 输入张量

    返回类型：
         oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([-0.5, 0, 0.5], dtype=flow.float32)
        >>> gelu = flow.nn.GELU()

        >>> out = gelu(input)
        >>> out
        tensor([-0.1543,  0.0000,  0.3457], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.in_top_k,
    r"""in_top_k(targets, predictions, k) -> Tensor
    
    目标是否在前 :attr:`k` 个预测中。

    参数：
        - **targets** (Tensor): 数据类型为 int32 或 int64 的目标张量
        - **predictions** (Tensor): float32 类型的预测张量
        - **k** (int): 要查看计算精度的最大元素的数量

    返回类型：
        oneflow.tensor: 元素为 bool 的张量。k 处的计算精度作 bool 张量值。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> targets1 = flow.tensor([3, 1], dtype=flow.int32)
        >>> predictions1 = flow.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0],], dtype=flow.float32)
        >>> out1 = flow.in_top_k(targets1, predictions1, k=1)
        >>> out1
        tensor([ True, False], dtype=oneflow.bool)
        >>> out2 = flow.in_top_k(targets1, predictions1, k=2)
        >>> out2
        tensor([True, True], dtype=oneflow.bool)
        >>> targets2 = flow.tensor([3, 1], dtype=flow.int32, device=flow.device('cuda'))
        >>> predictions2 = flow.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0],], dtype=flow.float32, device=flow.device('cuda'))
        >>> out3 = flow.in_top_k(targets2, predictions2, k=1)
        >>> out3
        tensor([ True, False], device='cuda:0', dtype=oneflow.bool)

    """
)
 
reset_docstr(
    oneflow.sqrt,
    r"""返回一个元素为 :attr:`input` 元素平方根的新 tensor 。
        公式为：

        .. math::
            \text{out}_{i} = \sqrt{\text{input}_{i}}

        参数：
            - **input** (Tensor): 输入张量

        示例：

        .. code-block:: python

            >>> import oneflow as flow
            
            >>> input = flow.tensor([1.0, 2.0, 3.0], dtype=flow.float32)
            >>> output = flow.sqrt(input)
            >>> output
            tensor([1.0000, 1.4142, 1.7321], dtype=oneflow.float32)
    """,
)

reset_docstr(
    oneflow.var,
    r"""var(input, dim=None, unbiased=True,  keepdim=False) -> Tensor

    返回给定维度 :attr:`dim` 中 :attr:`input` 张量的每一行的方差。

    如果 :attr:`keepdim` 为 `True` ，输出张量与 :attr:`input` 的大小相同。除非维度 :attr:`dim`  的大小为 1 ，
    否则输出的维度将被压缩 (参见 `flow.squeeze()` ) 导致输出张量的维度少 1 （或 `len(dim)` ）。

    参数：
        - **input** (Tensor): 输入张量
        - **dim** (int 或者 tuple of python:ints): 要减少的一个或多个维度。默认为None
        - **unbiased** (bool, 可选): 是否使用贝塞尔校正 (:math:`\delta N = 1`) 。 默认为 True
        - **keepdim** (bool, 可选): 输出张量是否保留了 :attr:`input` 的维度。 默认为 False

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[-0.8166, -1.3802, -0.3560]], dtype=flow.float32)
        >>> output = flow.var(input, 1, True)
        >>> output
        tensor([0.2631], dtype=oneflow.float32)
        
    """,
)

reset_docstr(
    oneflow.std,
    r"""
    返回输入张量在 :attr:`dim` 维度上每行的标准差。如果 :attr:`dim` 是一个维度列表，则对所有维度进行规约。

    如果 keepdim 为真，输出张量与输入张量大小相同，除了在 :attr:`dim` 维度的大小变为1。否则， :attr:`dim` 将被压缩，导致输出张量拥有 1 (或者 len(dim)) 个更少的维度。

    如果 :attr:`unbiased` 为 ``False`` ，则标准差将通过有差估算器计算。否则，贝塞尔校正将被使用。

    参数：
        - **input** (Tensor): 输入张量
        - **dim** (int or tuple of python:ints): 维度或者被减少的多个维度
        - **unbiased** (bool): 是否使用无差估计
        - **keepdim** (bool): 输出张量是否保留 `dim` 

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1.0,2.0,3.0])
        >>> output = flow.std(input, dim=0)
        >>> output
        tensor(1., dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.addcmul,
    r"""
    oneflow.addcmul(input, tensor1, tensor2, *, value=1) -> Tensor

    执行 tensor1 与 tensor2 的逐元乘法，将结果与标量值相乘并加到输入中。

    该文档引用自：
    https://pytorch.org/docs/stable/generated/torch.addcmul.html
    
    .. math::
        \text{out}_i = \text{input}_i + value \times\  \text{tensor1}_i \times\ \text{tensor2}_i
        
    参数:
        - **input** (Tensor) - 被添加的张量
        - **tensor1** (Tensor) - 被乘的张量
        - **tensor2** (Tensor) - 被乘的张量
    
    关键字参数:
        value (Number, optional): 乘法器 :math:`tensor1 * tensor2`.

    返回类型:
        oneflow.Tensor: 输出张量

    示例:

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.rand(2, 3, 4)
        >>> tensor1 = flow.rand(2, 3, 4)
        >>> tensor2 = flow.rand(2, 3, 4)
        >>> out = flow.addcmul(input, tensor1, tensor2, value=2)
        >>> out.size()
        oneflow.Size([2, 3, 4])
    """,
)

reset_docstr(
    oneflow.as_strided,
    r"""
    创建具有指定大小、步幅和存储偏移的现有 oneflow.Tensor 输入的视图。

    该文档引用自：
    https://pytorch.org/docs/1.10/generated/torch.as_strided.html
        
    参数:
        - **input** (Tensor) - 被添加的张量。
        - **size** (tuple or ints) - 输出张量的形状。
        - **stride** (tuple or ints) - 输出张量的步长。
        - **storage_offset** (int) - 输出张量的底层存储中的偏移量。

    返回类型:
        oneflow.Tensor: 输出张量

    示例:

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(2,3,5)
        >>> output = flow.as_strided(input, (2,3,3), (1,2,3), 1)
        >>> output.size()
        oneflow.Size([2, 3, 3])
    """,
)

reset_docstr(
    oneflow.cumprod,
    r"""oneflow.cumprod(input, dim) -> Tensor

    此运算符计算给定维度中输入元素的累积乘积。

    方程是：

    .. math::
        y_{i}=x_{0}*x_{1}*...*x_{i}
        
    参数:
        - **input** (Tensor) - 输入张量。
        - **dim** (int) - 进行 cumsum 的维度，其有效范围为 [-N, N-1)，N 是张量的维度。

    返回类型:
        oneflow.Tensor: 带有 cumprod 结果的张量

    示例:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input=flow.tensor([1, 2, 3])
        >>> flow.cumprod(input, dim=0)
        tensor([1, 2, 6], dtype=oneflow.int64)
    """,
)

reset_docstr(
    oneflow.cumsum,
    r"""oneflow.cumsum(input, dim) -> Tensor

    此运算符计算给定维度中输入元素的累积和。

    方程是：

    .. math::
        y_{i}=x_{0}+x_{1}+...+x_{i}
        
    参数:
        - **input** (Tensor) - 输入张量。
        - **dim** (int) - 进行 cumsum 的维度，其有效范围为 [-N, N-1)，N 是张量的维度。

    返回类型:
        oneflow.Tensor: 带有 cumsum 结果的张量

    示例:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.ones(3, 3)
        >>> dim = 1
        >>> flow.cumsum(input, dim)
        tensor([[1., 2., 3.],
                [1., 2., 3.],
                [1., 2., 3.]], dtype=oneflow.float32)
    """,
)

reset_docstr(
    oneflow.dot,
    r"""该运算符计算张量输入和其他的点积。

    方程是：

    .. math::
        sum_{i=1}^{n}(x[i] * y[i])
        
        
    参数:
        - **input** (Tensor) - 点积中的第一个张量。
        - **other** (Tensor) - 点积中的第二个张量。

    形状：
        - input：输入必须是一维的。
        - other：其他必须是一维的。

    示例:

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.dot(flow.Tensor([2, 3]), flow.Tensor([2, 1]))
        tensor(7., dtype=oneflow.float32)
    """,
)

reset_docstr(
    oneflow.floor_,
    r"""
    函数 :func:`oneflow.floor` 的 In-place 版本。
    """,
)

reset_docstr(
    oneflow.hsplit,
    r"""
    将输入（一个或多个维度的张量）按照 indices_or_sections 水平地分割成多个张量。
    每个分割都是输入的一个视图。
    如果输入是一维的，就相当于调用 oneflow.tensor_split(input, indices_or_sections, dim=0) 
    (分割的维度为0)，如果输入有两个或更多维度，则相当于调用 oneflow.tensor_split(input, indices_or_sections, dim=1)（分割维度为1），但如果 indices_or_sections
    是一个整数，它必须均匀地除以分割维度，否则会产生一个运行时错误。
    
    该文档参考自:
    https://pytorch.org/docs/1.10/generated/torch.hsplit.html.

    参数：
        - **input** (Tensor) - 输入张量。
        - **indices_or_sections** (int or a list) - 如果 indices_or_sections 是一个整数 n ，那么输入将沿着 dim 维度被分成 n 个部分。如果输入沿维度 dim 可被n整除，则每个部分的大小都相同。
                                                    如果输入不能被 n 整除，那么第一个 int(input.size(dim)%n) 的大小。
                                                    部分的大小为 int(input.size(dim) / n)+1，其余部分的大小为 int(input.size(dim) / n)。
                                                    如果 indices_or_sections 是一个 ints 的列表或元组，那么输入将沿着 dim 的维度在列表、元组或元组中的每个索引处进行分割。
                                                    列表、元组或张量中的每一个索引进行分割。例如，indices_or_sections=[2, 3]，dim=0，将导致张量的出现 
                                                    input[:2], input[2:3], and input[3:]。如果 indices_or_sections 是一个张量，它必须是一个零维或
                                                    如果indices_or_sections是一个张量，它在 CPU 上必须是一个零维或一维的长张量。

    返回类型：
        oneflow.TensorTuple: 输出的 TensorTuple
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(3,4,5,6)
        >>> output = flow.hsplit(input,(1,3))
        >>> output[0].size()
        oneflow.Size([3, 1, 5, 6])
        >>> output[1].size()
        oneflow.Size([3, 2, 5, 6])
        >>> output[2].size()
        oneflow.Size([3, 1, 5, 6])
    """,
)

reset_docstr(
    oneflow.log2,
    r"""
    oneflow.log2(input) -> Tensor

    返回一个新的张量，张量中的元素是以2为底的自然对数 :attr:`input`。

    .. math::
        y_{i} = log2_{e} (x_{i})
    
    参数：
        - **input** (Tensor) - 输入向量。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> arr = np.random.randn(2, 3, 4, 5)
        >>> input = flow.tensor(arr, dtype=flow.float32)
        >>> output = flow.log2(input)
    """,
)

reset_docstr(
    oneflow.logical_not,
    r"""
    计算给定输入张量的元素的逻辑非。零被视为假，非零被视为真。
    
    参数：
        - **input** (oneflow.Tensor): 输入张量。
        - **other** (oneflow.Tensor): 与之计算逻辑非的张量。

    返回类型：
        oneflow.Tensor: 输出张量
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1, 0, -1], dtype=flow.float32)
        >>> out = flow.logical_not(input)
        >>> out
        tensor([False,  True, False], dtype=oneflow.bool)
    """,
)

reset_docstr(
    oneflow.movedim,
    r"""
    将源中的输入尺寸移动到目标中的位置。
    其他没有明确移动的输入维度保持原来的顺序，并出现在目的地未指定的位置上。

    该文档引用自：
    https://pytorch.org/docs/1.10/generated/torch.movedim.html。

    计算给定输入张量的元素的逻辑非。零被视为假，非零被视为真。
    
    参数：
        - **input** (Tensor) -输入张量。
        - **source** (int or a list) - 要移动的数据的原始位置。这必须是唯一的。
        - **destination** (int or a list) - 要移动的数据的目标位置。这必须是唯一的。

    返回类型：
        oneflow.Tensor: 输出张量
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(2, 3, 4, 5), dtype=flow.float32)
        >>> output = flow.movedim(input, 1, 0)
        >>> output.shape
        oneflow.Size([3, 2, 4, 5])
        >>> output = flow.movedim(input, (1, 2), (0, 1))
        >>> output.shape
        oneflow.Size([3, 4, 2, 5])
    """,
)

reset_docstr(
    oneflow.sin_,
    r"""
    函数 :func:`oneflow.sin` 的本地版本。
    """,
)

reset_docstr(
    oneflow.tensor_split,
    r"""
    将一个张量分割成多个子张量，这些子张量都是输入的展开，按照 :attr:`indices_or_sections` 指定的索引或节数沿维度 dim 分割。
    
    该文档参考了:
    https://pytorch.org/docs/1.10/generated/torch.tensor_split.html。
    
    参数：
        - **input** (Tensor) -输入张量。
        - **indices_or_sections** (int or a list) - 如果 :attr:`indices_or_sections` 是一个整数 n，那么输入将沿着 dim 维度被分割成 n 个部分。
            如果输入沿维度 dim 可被 n 整除，则每个部分的大小相等，即 input.size(dim) / n。

            如果输入不被 n 整除，第一个 int(input.size(dim)% n).部分的大小将是int(input.size(dim) / n)+1，其余部分的大小将是 int(input.size(dim) / n)。

            如果 :attr:`indices_or_sections` 是一个 ints 的列表或元组，那么输入将在列表、元组或张量中的每个索引处沿 dim 维度进行分割。
            
            例如，indices_or_sections=[2, 3]，dim=0，将产生张量 input[:2]，input[2:3] 和 input[3:]。
            
            如果 indices_or_sections 是一个张量，它在 CPU 上必须是一个零维或一维的长张量。
        - **dim** (int) - 用来分割张量的维度。

    返回类型：
        oneflow.TensorTuple: 输出TensorTuple.
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(3,4,5)
        >>> output = flow.tensor_split(input,(2,3),2)
        >>> output[0].size()
        oneflow.Size([3, 4, 2])
        >>> output[1].size()
        oneflow.Size([3, 4, 1])
        >>> output[2].size()
        oneflow.Size([3, 4, 2])
    """,
)