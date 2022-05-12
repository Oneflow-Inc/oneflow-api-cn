import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.amin,
    r"""
    oneflow.amin(input, dim=None, keepdim=False) -> Tensor

    这个函数与 PyTorch 的 amin 函数等价。

    该文档引用自: https://pytorch.org/docs/stable/generated/torch.amin.html 
    
    返回给定维度 dim 中输入张量的每个切片的最小值。

    如果 :attr: `keepdim` 为 `True`，则输出张量的大小与输入的大小相同，但尺寸为1的维度 dim 除外。否则，dim 被压缩（参见 :func:`oneflow.squeeze`），导致输出张量具有1（或len(dim) ) 更少的维度。

    参数:
        - **input** (oneflow.Tensor) - 输入张量。
        - **dim** (int, Tuple[int]) - 减少的维度。
        - **keepdim** (bool) - 输出张量是否保留了`dim`

    示例:

    .. code-block:: python

        >>> import oneflow as flow
               
        >>> x = flow.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> flow.amin(x, 1)
        tensor([[0, 1],
                [4, 5]], dtype=oneflow.int64)
        >>> flow.amin(x, 0)
        tensor([[0, 1],
                [2, 3]], dtype=oneflow.int64)
        >>> flow.amin(x)
        tensor(0, dtype=oneflow.int64)
        >>> flow.amin(x, 0, True)
        tensor([[[0, 1],
                 [2, 3]]], dtype=oneflow.int64)
    """
)