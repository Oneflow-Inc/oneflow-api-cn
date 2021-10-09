import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.adaptive_avg_pool3d,
    r"""adaptive_avg_pool3d(input, output_size)

    输入由多个平面组成的的信号，并对其应用 3D 自适应平均池化。

    参考： :mod:`oneflow.nn.AdaptiveAvgPool3d`

    参数：
        - **input**: 输入张量
        - **output_size**: 目标输出大小（单个整数或包含三个整数的元组）
    """,
)
