import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.unbind,
    r"""
    这个函数与 PyTorch 的 unbind 函数等价，可移除一个张量维度。

    返回沿给定维度的移除的张量的所有切片的一个元组，即已经被移除的部分。
    
    参数：
        - **x** (Tensor) - 需要 unbind 张量。
        - **dim** (int) - 移除的维度。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor(range(12)).reshape([3,4])
        >>> flow.unbind(x)
        (tensor([0, 1, 2, 3], dtype=oneflow.int64), tensor([4, 5, 6, 7], dtype=oneflow.int64), tensor([ 8,  9, 10, 11], dtype=oneflow.int64))
        >>> flow.unbind(x, 1)
        (tensor([0, 4, 8], dtype=oneflow.int64), tensor([1, 5, 9], dtype=oneflow.int64), tensor([ 2,  6, 10], dtype=oneflow.int64), tensor([ 3,  7, 11], dtype=oneflow.int64))
    """,
)
