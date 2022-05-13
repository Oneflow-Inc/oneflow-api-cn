import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.topk,
    r"""
    查找指定轴上的 k 个最大条目的值和索引。

    参数：
        - **input** (oneflow.Tensor) - 输入张量。
        - **k** (int) - “top-k” 的 k。
        - **dim** (int,可选) - 要排序用的维度。默认为最后一个维度（-1）。
        - **largest** (bool,可选) - 控制是否返回最大或最小的元素
        - **sorted** (bool,可选) - 控制是否按排序顺序返回元素（现在只支持 True！）。

    返回类型：
        - **Tuple** (oneflow.Tensor, oneflow.Tensor(dtype=int32)) - 一个(values, indices)的元组，其中 indices 是原始输入张量中的元素的索引。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=np.float32)
        >>> (values, indices) = flow.topk(flow.Tensor(x), k=3, dim=1)
        >>> values
        tensor([[8., 7., 3.],
                [9., 4., 3.]], dtype=oneflow.float32)
        >>> indices
        tensor([[2, 3, 1],
                [1, 2, 3]], dtype=oneflow.int64)
        >>> values.shape
        oneflow.Size([2, 3])
        >>> indices.shape
        oneflow.Size([2, 3])
        >>> (values, indices) = flow.topk(flow.Tensor(x), k=2, dim=1, largest=False)
        >>> values
        tensor([[1., 2.],
                [1., 2.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 4],
                [0, 4]], dtype=oneflow.int64)
        >>> values.shape
        oneflow.Size([2, 2])
        >>> indices.shape
        oneflow.Size([2, 2])
    """,
)
