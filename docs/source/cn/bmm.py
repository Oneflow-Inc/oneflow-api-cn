import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.bmm,
    r"""
    对存储在 input 和 mat2 中的矩阵进行批量矩阵-矩阵乘法。

    `input` 和 `mat2` 必须是 3-D 张量，每个张量都包含相同数量的矩阵。

    如果 input 是 (b x n x m) 张量，mat2 是 (b x m x p) 张量，out 将是 (b x n x p) 张量。

    参数:
        - **input** (oneflow.Tensor) - 要相乘的第一批矩阵。
        - **mat2** (oneflow.Tensor) - 要相乘的第二批矩阵。

    示例:

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> import numpy as np
        >>> input1 = flow.randn(10, 3, 4)
        >>> input2 = flow.randn(10, 4, 5)
        >>> of_out = flow.bmm(input1, input2)
        >>> of_out.shape
        oneflow.Size([10, 3, 5])
    """
)