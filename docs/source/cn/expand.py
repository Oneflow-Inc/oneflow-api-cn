import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.expand,
    r"""expand(input, *sizes) -> Tensor
    
    此运算符将输入张量扩展到更大的尺寸。
    将 -1 作为 size 意味着不更改该维度的大小。
    Tensor :attr:`input` 可以扩展到大的维度，新的维度将附加在前面。
    对于新维度，:attr:`sizes` 不能设置为 -1。

    参数：
        - **input** (oneflow.Tensor): 输入张量
        - ***sizes** (oneflow.Size or int): 所需的展开尺寸。

    返回类型：
        oneflow.Tensor

    示例：
    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([[[[0, 1]], [[2, 3]], [[4, 5]]]], dtype=flow.float32)
        >>> input.shape
        oneflow.Size([1, 3, 1, 2])
        >>> out = input.expand(1, 3, 2, 2)
        >>> out.shape
        oneflow.Size([1, 3, 2, 2])
    
    """
)
