import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.slice,
    r"""slice(input, slice_tup_list) -> Tensor

    从张量中提取切片。
    以格式为 (start, stop, step) 的 :attr:`slice_tup_list` 为切片索引，在每个维度进行切片。

    参数：
        - **input** (Tensor)
        - **slice_tup_list** (Sequence[Tuple[int, int, int]]): 指定每个维度的切片 (start, stop, step) 

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.randn(3, 6, 9, dtype=flow.float32)
        >>> tup_list = [[None, None, None], [0, 5, 2], [0, 6, 3]]
        >>> y = flow.slice(input, slice_tup_list=tup_list)
        >>> y.shape
        oneflow.Size([3, 3, 2])
    """
)

reset_docstr(
    oneflow.logical_slice,
    r"""
    从 global tensor 中提取切片。 `slice_tup_list` 指定了每个维度的分片索引，格式是（start, stop, step）。
    该算子将根据 `slice_tup_list` 对张量进行分割。

    参数：
        - **input** (Tensor) - 输入张量
        - **slice_tup_list**  - 分片元组列表，表示每个维度分片 (start, stop, step)。

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> placement = flow.placement("cpu", ranks=[0])
        >>> x = flow.Tensor([[1, 2], [3, 4]], placement=placement, sbp=flow.sbp.broadcast)
        >>> y = flow.logical_slice(x, slice_tup_list=[[0, 1, 1]])
        >>> y.numpy()
        array([[1., 2.]], dtype=float32)
    """
)
