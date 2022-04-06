import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.adaptive_avg_pool1d,
    r"""adaptive_avg_pool1d(input, output_size)
    
    在由多个平面组成的信号 `input` 上应用 1D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool1d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** - 目标输出大小（单个整数）

    """
)

reset_docstr(
    oneflow.adaptive_avg_pool2d,
    r"""adaptive_avg_pool2d(input, output_size)
    
    在由多个平面组成的信号 `input` 上应用 2D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool2d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** - 目标输出大小（单个整数或包含两个整数的元组）

    """
)

reset_docstr(
    oneflow.adaptive_avg_pool3d,
    r"""adaptive_avg_pool3d(input, output_size)

    在由多个平面组成的信号 `input` 上应用 3D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool3d`

    参数：
        - **input** (Tensor) - 输入张量
        - **output_size** - 目标输出大小（单个整数或包含三个整数的元组）
    """,
)

reset_docstr(
    oneflow.nn.AdaptiveAvgPool1d,
    r"""AdaptiveAvgPool1d(output_size)

    在由多个平面组成的输入信号 `input` 上应用 1D 自适应平均池化。

    对于任何大小的输入，输出大小都是 H。
    
    输出的数量等于输入平面的数量。

    参数：
        - **output_size** - 目标输出大小 H（单个整数）
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = flow.randn(1, 64, 8)
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 5])

    """
)

reset_docstr(
    oneflow.nn.AdaptiveAvgPool2d,
    r"""AdaptiveAvgPool2d(output_size)
    
    在由多个平面组成的的信号 `input` 上应用 2D 自适应平均池化。

    对于任何大小的输入，输出大小都是 H x W 。
    
    输出的数量等于输入平面的数量。

    参数：
        - **output_size** - 目标输出大小（单个整数 H 或包含两个整数的元组 ``(H, W)`` ）。 H 和 W 可以是 ``int`` 也可以是 ``None`` ，如果为 ``None`` 则大小将和输入大小一致。


    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = flow.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 5, 7])

        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = flow.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 7, 7])

        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = flow.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 10, 7])

    """
    )

reset_docstr(
    oneflow.nn.AdaptiveAvgPool3d,
    r"""AdaptiveAvgPool3d(output_size)

    在由多个平面组成的信号 `input` 上应用 3D 自适应平均池化。

    对于任何大小的输入，输出大小都是 D x H x W 。
    
    输出的数量等于输入平面的数量。

    参数：
        - **output_size** - 目标输出大小（单个整数 D 则为 D x D x D 或包含三个整数的元组 (D, H, W) ）。 H 、 W 和 D 可以是 ``int`` 也可以是 ``None`` ，如果为 ``None`` 则大小将和输入大小一致。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> m = nn.AdaptiveAvgPool3d((5,7,9))
        >>> input = flow.randn(1, 64, 8, 9, 10)
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 5, 7, 9])

        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = flow.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 7, 7, 7])

        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = flow.randn(1, 64, 10, 9, 8)
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 7, 9, 8])

    """
    )   

reset_docstr(
    oneflow.nn.AvgPool1d,
    r"""AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

    在由多个平面组成的信号 `input` 上执行 1D 平均池化。
    在最简单的情况下，输出值是输入大小为 :math:`(N, C, H, W)` 的层。
    输出 :math:`(N, C, H_{out}, W_{out})` 和 `kernel_size` ， :math:`k` 可以被精确地描述为：
    
    .. math::
        out(N_i, C_j, l)  = \frac{1}{k} \sum_{m=0}^{k-1}
                            input(N_i, C_j, stride[0] \times h + m, stride*l + m)
    
    如果 :attr:`padding` 非零，则输入在两侧隐式填充 0 以填充点数。
    
    参数 :attr:`kernel_size` 、 :attr:`stride` 、 :attr:`padding` 可以为 int 或者单元素元组。
    
    Note:
        当 :attr:`ceil_mode` 为 True 时，如果滑动窗口在 left padding 或输入内开始，则允许滑动窗口出界。忽略在右侧填充区域开始的滑动窗口。
    
    参数：
        - **kernel_size** (Union[int, Tuple[int, int]]): 窗口的大小
        - **strides** (Union[int, Tuple[int, int]], 可选): 窗口的 stride 。默认值为 None 
        - **padding** (Union[int, Tuple[int, int]]): 如果非 0 ，在两侧添加隐式填充 0 。默认为 0 
        - **ceil_mode** (bool): 如果为 True ，将使用 ceil 而不是 floor 来计算输出形状。默认为 False 
        - **count_include_pad** (bool): 如果为 True ，将在平均计算中填充 0 ，默认为 True 
    
    示例：

    .. code-block:: python 
        
        import oneflow as flow 

        m = flow.nn.AvgPool1d(kernel_size=3, padding=1, stride=1)
        x = flow.randn(1, 4, 4)
        y = m(x)
        y.shape 
        oneflow.Size([1, 4, 4])

    """
)

reset_docstr(
    oneflow.nn.AvgPool2d,
    r"""AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0)
    
    在由多个平面组成的信号 `input` 上执行 2D 平均池化。

    在最简单的情况下，输出值是输入大小为 :math:`(N, C, H, W)` 的层。
    
    输出 :math:`(N, C, H_{out}, W_{out})` 和 `kernel_size` ， :math:`(kH, kW)` 可以被精确地描述为：

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    参数：
        - **kernel_size** (Union[int, Tuple[int, int]]): 整数或长度为 1 或 2 的整数列表。输入张量的每个维度的窗口大小
        - **strides** (Union[int, Tuple[int, int]]): 整数或长度为 1 或 2 的整数列表。输入张量的每个维度的滑动窗口的 stride 。默认为 None
        - **padding** (Tuple[int, int]): 整数或长度为 1 或 2 的整数列表。在两侧添加隐式填充 0 。默认为 0 
        - **ceil_mode** (bool, default to False): 如果为 True 。将使用 ceil 而不是 floor 来计算输出形状。默认为 False 

    示例：

    .. code-block:: python

        import oneflow as flow 

        m = flow.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        x = flow.randn(1, 4, 4, 4)
        y = m(x)   
        y.shape
        oneflow.Size([1, 4, 4, 4])

    """
)

reset_docstr(
    oneflow.nn.AvgPool3d,
    r"""AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=0)

    在由多个平面组成的信号 `input` 上执行 3D 平均池化。在最简单的情况下，输出值是输入大小为 :math:`(N, C, D, H, W)` 的层。
    
    输出 :math:`(N, C, D_{out}, H_{out}, W_{out})` 和 `kernel_size` ， :math:`(kD, kH, kW)` 可以被精确地描述为：
    
    .. math::
        out(N_i, C_j, d, h, w)  = \frac{1}{kD * kH * kW } \sum_{k=0}^{kD-1} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times d + k, stride[1] \times h + m, stride[2] \times w + n)
    
    如果 :attr:`padding` 非零，则输入在三侧隐式填充 0 以填充点数。
    
    Note:

        当 :attr:`ceil_mode` 为 True 时，如果滑动窗口在 left padding 或输入内开始，则允许滑动窗口出界。忽略在右侧填充区域开始的滑动窗口。
    
    参数：
        - **kernel_size** (Union[int, Tuple[int, int, int]]): 窗口的大小
        - **strides** (Union[int, Tuple[int, int, int]], 可选): 窗口的 stride 。默认值为 None 
        - **padding** (Union[int, Tuple[int, int, int]]):  如果非 0 ，在三侧添加隐式填充 0 。默认为 0 
        - **ceil_mode** (bool): 如果为 True ，将使用 ceil 而不是 floor 来计算输出形状。默认为 False 
        - **count_include_pad** (bool): 如果为 True ，将在平均计算中填充 0 ，默认为 True 
        - **divisor_override** (int): 如果设定了 attr:`divisor_override` ，它将用作除数，否则 attr:`kernel_size` 将作为除数。默认为 0 
    
    形状：
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`

        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` 
        
          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{kernel_size}[0]}{\text{stride}[0]} + 1\right\rfloor
        
          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{kernel_size}[1]}{\text{stride}[1]} + 1\right\rfloor
        
          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{kernel_size}[2]}{\text{stride}[2]} + 1\right\rfloor

    示例：
    
    .. code-block:: python
    
        import oneflow as flow
        
        m = flow.nn.AvgPool3d(kernel_size=(2,2,2),padding=(0,0,0),stride=(1,1,1))
        x = flow.randn(9, 7, 11, 32, 20)
        y = m(x)
        y.shape
        oneflow.Size([9, 7, 10, 31, 19])

    """
)
