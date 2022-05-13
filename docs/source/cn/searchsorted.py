import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.searchsorted,
    r"""
    searchsorted() -> oneflow.Tensor

    该文档引用自：https://pytorch.org/docs/1.10/generated/torch.searchsorted.html?highlight=searchsorted
    
    从 sorted_sequence 的最内层维度中找到索引，如果在这些索引之前插入 values 中的相应值，那么 sorted_sequence 中相应的最内层维度的顺序将被保留下来。返回一个大小与 value 相同的新张量。如果 right 是 False（默认），那么 sorted_sequence 的左边边界将被关闭。即，返回的索引满足以下规则：

    =================  =========  ==========================================================================
    sorted_sequence     right      returned index satisfies
    =================  =========  ==========================================================================
    1-D                 False      sorted_sequence[i-1] < values[m][n]...[l][x] <= sorted_sequence[i]
    1-D                 True       sorted_sequence[i-1] <= values[m][n]...[l][x] < sorted_sequence[i]
    N-D                 False      sorted_sequence[m][n]...[l][i-1] < values[m][n]...[l][x] 
                                                    <= sorted_sequence[m][n]...[l][i]
    N-D                 True       sorted_sequence[m][n]...[l][i-1] <= values[m][n]...[l][x] 
                                                    sorted_sequence[m][n]...[l][i]
    =================  =========  ==========================================================================

    参数：
        - **sorted_sequence** (Tensor) - N-D 或1-D 张量，包含最内侧维度上的单调增加序列。
        - **values** (Tensor or Scalar) - N-D 张量或包含搜索值的标量。
        - **out_int32** (bool 可选) - 标明输出数据类型。如果为 True，则为 torch.int32，否则为 torch.int64。默认值为 False，即默认输出数据类型为 torch.int64。
        - **right** (bool 可选) - 如果是 False，返回找到的第一个合适的位置。如果为 True，则返回最后一个这样的索引。如果没有找到合适的索引，返回0的非数字值 (例如：Nan, inf)，或者在 sorted_sequence 中的最内层维度的大小（一个传递最内层维度的最后一个索引）。换句话说，如果是 False，则得到获得每个值的下限索引，这些值是在排序_序列的相应最内层维度上的维度上的每个值的下限索引。如果是 True，则获得上界的索引。默认值是 False。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> sorted_sequence = flow.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
        >>> sorted_sequence
        tensor([[ 1,  3,  5,  7,  9],
                [ 2,  4,  6,  8, 10]], dtype=oneflow.int64)
        >>> values = flow.tensor([[3, 6, 9], [3, 6, 9]])
        >>> values
        tensor([[3, 6, 9],
                [3, 6, 9]], dtype=oneflow.int64)
        >>> flow.searchsorted(sorted_sequence, values)
        tensor([[1, 3, 4],
                [1, 2, 4]], dtype=oneflow.int64)
        >>> flow.searchsorted(sorted_sequence, values, right=True)
        tensor([[2, 3, 5],
                [1, 3, 4]], dtype=oneflow.int64)
        >>> sorted_sequence_1d = flow.tensor([1, 3, 5, 7, 9])
        >>> sorted_sequence_1d
        tensor([1, 3, 5, 7, 9], dtype=oneflow.int64)
        >>> flow.searchsorted(sorted_sequence_1d, values)
        tensor([[1, 3, 4],
                [1, 3, 4]], dtype=oneflow.int64)

    """
)
