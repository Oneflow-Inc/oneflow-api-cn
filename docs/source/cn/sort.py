import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.sort,
    r"""sort(input, dim=-1, descending=False) -> tuple(values, indices)
    
    按值升序沿给定维度 :attr:`dim` 对张量 :attr:`input` 的元素进行排序。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **dim** (int, 可选): 要排序的维度，默认为（dim = -1）
        - **descending** (bool, 可选): 控制排序方式（升序或降序）

    返回类型：
        Tuple(oneflow.tensor, oneflow.tensor(dtype=int32)): 一个元素为 (values, indices) 的元组
        元素为排序后的 :attr:`input` 的元素，索引是原始输入张量 :attr:`input` 中元素的索引。

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=flow.float32)
        >>> (values, indices) = flow.sort(input)
        >>> values
        tensor([[1., 2., 3., 7., 8.],
                [1., 2., 3., 4., 9.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 4, 1, 3, 2],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> (values, indices) = flow.sort(input, descending=True)
        >>> values
        tensor([[8., 7., 3., 2., 1.],
                [9., 4., 3., 2., 1.]], dtype=oneflow.float32)
        >>> indices
        tensor([[2, 3, 1, 4, 0],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> (values, indices) = flow.sort(input, dim=0)
        >>> values
        tensor([[1., 3., 4., 3., 2.],
                [1., 9., 8., 7., 2.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 0, 1, 1, 0],
                [1, 1, 0, 0, 1]], dtype=oneflow.int32)
 
    """
)

