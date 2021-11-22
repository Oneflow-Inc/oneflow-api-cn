import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.cast,
    r"""cast(x, dtype) -> Tensor
    
    将输入张量 `x` 转化为数据类型 `dtype`

    参数：
        - **x** (oneflow.tensor): 输入张量
        - **dtype** (flow.dtype): 输出张量的数据类型

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.randn(2, 3, 4, 5, dtype=flow.float32)
        >>> output = flow.cast(input, flow.int8)
        >>> output.shape
        oneflow.Size([2, 3, 4, 5])

    """,
)    

reset_docstr(
    oneflow.transpose,
    r"""transpose(input, dim0, dim1) -> Tensor
    
    返回一个 tensor ，它是 :attr:`input` 的转置版本。交换指定维度 :attr:`dim0` 和 :attr:`dim1` 。

    输出 tensor 与输入 tensor 共享内存，
    所以改变其中一个的元素会改变另一个的元素。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **dim0** (int): 要转置的第一个维度。
        - **dim1** (int): 要转置的第二个维度。
    
    返回类型：
        oneflow.tensor: 转置张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.randn(2, 6, 5, 3, dtype=flow.float32)
        >>> out = flow.transpose(input, 0, 1).shape
        >>> out
        oneflow.Size([6, 2, 5, 3])

    """,
)
