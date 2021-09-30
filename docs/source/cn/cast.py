import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.cast,
    r"""cast(x, dtype) -> Tensor
    
    将输入张量 `x` 转化为数据类型 `dtype`

    参数：
        - **x** (oneflow.Tensor): 输入张量
        - **dtype** (flow.dtype): 输出张量的数据类型

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.randn(2, 3, 4, 5, dtype=flow.float32) 
        >>> output = flow.cast(input, flow.int8)
        >>> output.shape
        oneflow.Size([2, 3, 4, 5])

    """,
)    
