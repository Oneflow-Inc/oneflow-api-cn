import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.add,
    r"""add(input, other) -> Tensor
    
    计算 `input` 和 `other` 的和。支持 element-wise、标量和广播形式的加法。

    公式为：

    .. math::
        out = input + other

    参数：
        - **input** (Tensor): 输入张量
        - **other** (Tensor): 输入张量

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量
    
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

    返回值：
        oneflow.Tensor: 结果张量

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
        - **input** (Union[int, float, oneflow.Tensor]): input.
        - **other** (Union[int, float, oneflow.Tensor]): other.

    返回值：
        oneflow.Tensor: 结果张量
    
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

    返回值：
        oneflow.Tensor: 结果张量
    
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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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
        **x** (oneflow.Tensor): 张量

    返回值：
        oneflow.Tensor: 结果张量

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
    
    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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
    oneflow.cos,
    r"""cos(x) -> Tensor
    返回一个包含 :attr:`x` 中元素的余弦值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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
        **x** (oneflow.Tensor): 输入张量

    返回值：
        oneflow.Tensor: 结果张量   
               
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
        **x** (oneflow.Tensor): 输入张量

    返回至：
        oneflow.Tensor: 结果张量

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
        **x** (oneflow.Tensor): 输入张量

    返回值：
        oneflow.Tensor: 结果张量

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
    
    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量
    
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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        oneflow.Tensor: 结果张量

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

    返回值：
        Tensor: 张量 `input` 数轴上的方差结果

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
    
    返回值：
        oneflow.Tensor: 结果张量

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
    
    返回值：
        oneflow.Tensor: 结果张量

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
        - **a** (oneflow.Tensor): 张量
        - **b** (oneflow.Tensor): 张量

    返回值：
        oneflow.Tensor: 结果张量

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
        **x** (oneflow.Tensor): 输入张量

    返回值：
        oneflow.Tensor: 结果张量
    
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
