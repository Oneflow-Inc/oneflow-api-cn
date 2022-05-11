import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.amax,
    r"""
    oneflow.amax(input, dim=None, keepdim=False) -> Tensor

    这个函数相当于 PyTorch 的 amax 函数。它返回沿维度的最大值。

    参数:
        - **input** (oneflow.Tensor) - 输入张量。
        - **dim** (int or List of int, optional) - 要减少的维度。dim 的默认值为 None. 
        - **keepdim** (bool, optional) - 是否保留维度。keepdim 默认为 False。

    返回值类型:
        oneflow.Tensor: 输入张量的最大值。

    示例:

    .. code-block:: python
    
        >>> import oneflow as flow
               
        >>> x = flow.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> flow.amax(x, 1)
        tensor([[2, 3],
                [6, 7]], dtype=oneflow.int64)
        >>> flow.amax(x, 0)
        tensor([[4, 5],
                [6, 7]], dtype=oneflow.int64)
        >>> flow.amax(x)
        tensor(7, dtype=oneflow.int64)
        >>> flow.amax(x, 0, True)
        tensor([[[4, 5],
                 [6, 7]]], dtype=oneflow.int64)
    """
)