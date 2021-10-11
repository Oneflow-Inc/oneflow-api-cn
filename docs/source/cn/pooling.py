import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.adaptive_avg_pool1d,
    r"""adaptive_avg_pool1d(input, output_size)
    
    在由多个平面组成的的信号 `input` 上应用 1D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool1d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** (Int64List[1]): 目标输出大小（单个整数）

    """
)

reset_docstr(
    oneflow.adaptive_avg_pool2d,
    r"""adaptive_avg_pool2d(input, output_size)
    
    在由多个平面组成的的信号 `input` 上应用 2D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool2d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** (Int64List[2]): 目标输出大小（单个整数或包含两个整数的元组）

    """
)

reset_docstr(
    oneflow.adaptive_avg_pool3d,
    r"""adaptive_avg_pool3d(input, output_size)

    在由多个平面组成的的信号 `input` 上应用 3D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool3d`

    参数：
        - **input** (Tensor): 输入张量
        - **output_size** (Int64List[3]): 目标输出大小（单个整数或包含三个整数的元组）
    """,
)
