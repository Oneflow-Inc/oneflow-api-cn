import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.add,
    r"""add(input, other)
    
    计算 `input` 和 `other` 的和。支持 element-wise、标量和广播形式的加法。
    公式为：

    .. math::
        out = input + other

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise 加法
        >>> x = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # 标量加法
        >>> x = 5
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

        # 广播加法
        >>> x = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> y = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.add(x, y).numpy()
        >>> out.shape
        (2, 3)

    """,
)

reset_docstr(
    oneflow.sub,
    r"""sub(input, other)
    
    计算 `input` 和 `other` 的差，支持 element-wise、标量和广播形式的减法。
    公式为：

    .. math::
        out = input - other
    
    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise 减法
        >>> input = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # 标量减法
        >>> input = 5
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

        # 广播减法
        >>> input = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.sub(input,other).numpy()
        >>> out.shape
        (2, 3)

    """,
)

reset_docstr(
    oneflow.abs,
    r"""abs(input)
    
    返回一个包含 `input` 中每个元素的绝对值的tensor:`y = |x|`。
    
    参数：
        input (Tensor): 输入张量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> x = flow.tensor(np.array([-1, 2, -3, 4]).astype(np.float32))
        >>> flow.abs(x)
        tensor([1., 2., 3., 4.], dtype=oneflow.float32)
    
    """,
)

reset_docstr(
    oneflow.div,
    r"""div(input, other)
    
    计算 `input` 除以 `other`，支持 element-wise、标量和广播形式的除法。
    公式为：

    .. math::
        out = \frac{input}{other}
    
    参数：
        input (Union[int, float, oneflow.Tensor]): input.
        other (Union[int, float, oneflow.Tensor]): other.
    
    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise 除法
        >>> input = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # 标量除法
        >>> input = 5
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.div(input,other).numpy()
        >>> out.shape
        (2, 3)

        # 广播除法
        >>> input = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.div(input,other).numpy()
        >>> out.shape 
        (2, 3)

    """,
)

reset_docstr(
    oneflow.mul,
    r"""mul(input, other)
    
    计算 `input` 与 `other` 相乘，支持 element-wise、标量和广播形式的乘法。
    
    公式为：

    .. math::
        out = input \times other
    
    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        # element-wise 乘法
        >>> input = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape
        (2, 3)

        # 标量乘法
        >>> input = 5
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape
        (2, 3)

        # 广播乘法
        >>> input = flow.tensor(np.random.randn(1,1), dtype=flow.float32)
        >>> other = flow.tensor(np.random.randn(2,3), dtype=flow.float32)
        >>> out = flow.mul(input,other).numpy()
        >>> out.shape 
        (2, 3)

    """,
)

reset_docstr(
    oneflow.reciprocal,
    r"""reciprocal(x)
    计算x的倒数，如果x为0，倒数将被设置为0。

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=flow.float32)
        >>> out = flow.reciprocal(x)
        >>> out.numpy()
        array([[1.        , 0.5       , 0.33333334],
               [0.25      , 0.2       , 0.16666667]], dtype=float32)
    """,
)

reset_docstr(
    oneflow.asin,
    r"""asin(input)

    返回一个新的 tensor 包含 :attr:`input` 中每个元素的反正弦。

    .. math::
        \text{out}_{i} = \sin^{-1}(\text{input}_{i})

    参数：
        input (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([-0.5,  0.8, 1.0,  -0.8]), dtype=flow.float32)
        >>> output = flow.asin(input)
        >>> output.shape
        oneflow.Size([4])
        >>> output
        tensor([-0.5236,  0.9273,  1.5708, -0.9273], dtype=oneflow.float32)
        >>> input1 = flow.tensor(np.array([[0.8, 1.0], [-0.6, -1.0]]), dtype=flow.float32)
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
    r"""asinh()
    
    返回一个包含 :attr:`input` 中每个元素的反双曲正弦的新 tensor。

    .. math::
        \text{out}_{i} = \sinh^{-1}(\text{input}_{i})

    参数：
        input (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([2, 3, 4]), dtype=flow.float32)
        >>> output = flow.asinh(input)
        >>> output.shape
        oneflow.Size([3])
        >>> output
        tensor([1.4436, 1.8184, 2.0947], dtype=oneflow.float32)

        >>> input1 = flow.tensor(np.array([[-1, 0, -0.4], [5, 7, 0.8]]), dtype=flow.float32)
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
    r"""atan(input)

    返回一个包含 :attr:`input` 中所有元素的反正切的新 tensor。

    .. math::
        \text{out}_{i} = \tan^{-1}(\text{input}_{i})

    参数：
        input (Tensor): 输入张量

    示例：

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
        >>> input = flow.tensor(np.array([0.5, 0.6, 0.7]), dtype=flow.float32)
        >>> output = flow.atan(input)
        >>> output.shape
        oneflow.Size([3])
        
    """,
)

reset_docstr(
    oneflow.ceil,
    r"""ceil(input)
    
    返回一个新的 tensor，tensor 中元素为大于或等于 :attr:`input` 中元素的最小整数。
    公式为： 

    .. math::
        \text{out}_{i} = \left\lceil \text{input}_{i} \right\rceil = \left\lfloor \text{input}_{i} \right\rfloor + 1

    参数：
        input (oneflow.Tensor): 张量
    
    返回值：
        oneflow.Tensor: 结果张量

    示例： 


    .. code-block:: python 
        
        >>> import oneflow as flow
        >>> import numpy as np   
        >>> x = flow.tensor(np.array([0.1, -2, 3.4]).astype(np.float32))
        >>> y = flow.ceil(x)
        >>> y.shape
        oneflow.Size([3])
        >>> y
        tensor([ 1., -2.,  4.], dtype=oneflow.float32)
        >>> x = flow.tensor(np.array([[2.5, 4.6, 0.6],[7.8, 8.3, 9.2]]).astype(np.float32))
        >>> y = x.ceil()
        >>> y.shape
        oneflow.Size([2, 3])
        >>> y
        tensor([[ 3.,  5.,  1.],
                [ 8.,  9., 10.]], dtype=oneflow.float32)
        >>> x = flow.tensor(np.array([[[2.2, 4.4, 6.5],[7.1, 8.2, 9.3]],[[10.6,11.2,12.2],[13.5,14.8,15.9]]]).astype(np.float32))
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
    r"""log1p(input)
    
    返回一个新的 tensor，其自然对数为 (1 + input)。

    .. math::
        \text{out}_{i}=\log_e(1+\text{input}_{i})

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = flow.tensor(np.array([1.3, 1.5, 2.7]), dtype=flow.float32)
        >>> out = flow.log1p(x)
        >>> out
        tensor([0.8329, 0.9163, 1.3083], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.exp,
    r"""exp(x)

    此运算符计算 `input` 的指数。

    公式为：

    .. math::

        out = e^x

    参数：
        x (oneflow.Tensor): 张量

    返回值：
        oneflow.Tensor: 结果张量

    示例：

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> x = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> y = flow.exp(x)
        >>> y
        tensor([ 2.7183,  7.3891, 20.0855], dtype=oneflow.float32)

    """,
)
