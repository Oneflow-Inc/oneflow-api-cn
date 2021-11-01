import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.adaptive_avg_pool1d,
    r"""adaptive_avg_pool1d(input, output_size)
    
    在由多个平面组成的信号 `input` 上应用 1D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool1d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** (Int64List[1]): 目标输出大小（单个整数）

    """
)

reset_docstr(
    oneflow.adaptive_avg_pool2d,
    r"""adaptive_avg_pool2d(input, output_size)
    
    在由多个平面组成的信号 `input` 上应用 2D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool2d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** (Int64List[2]): 目标输出大小（单个整数或包含两个整数的元组）

    """
)

reset_docstr(
    oneflow.adaptive_avg_pool3d,
    r"""adaptive_avg_pool3d(input, output_size)

    在由多个平面组成的信号 `input` 上应用 3D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool3d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** (Int64List[3]): 目标输出大小（单个整数或包含三个整数的元组）
    """,
)

reset_docstr(
    oneflow.nn.AdaptiveAvgPool1d,
    r"""AdaptiveAvgPool1d(output_size)

    在由多个平面组成的输入信号 `input` 上应用 1D 自适应平均池化。

    对于任何大小的输入，输出大小都是 H 。
    输出的数量等于输入平面的数量。

    参数：
        - **output_size** (Int64List[1]): 目标输出大小（单个整数）
    
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
    
    在由多个平面组成的的输入信号上应用 2D 自适应平均池化。

    对于任何大小的输入，输出大小都是 H x W 。
    输出的数量等于输入平面的数量。

    参数：
        - **output_size** (Int64List[2]): 目标输出大小（单个整数 H 或包含两个整数的元组 (H, W) ）。 H 和 W 可以是 ``int`` 也可以是 ``None`` ，如果为 ``None`` 则大小将和输入大小一致。


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
        - **output_size** (Int64List[3]): 目标输出大小（单个整数 D 则为 D x D x D 或包含三个整数的元组 (D, H, W) ）。 H 、 W 和 D 可以是 ``int`` 也可以是 ``None`` ，如果为 ``None`` 则大小将和输入大小一致。

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
