import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.index_select,
    r"""index_select(input, dim, index) -> Tensor

    此接口与 PyTorch 一致。
    文档参考自： https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch/#torchindex_select

    沿指定维度 `dim` 选择值。

    :attr:`index` 必须是一个 1-D 张量，数据类型为 Int32。 :attr:`dim` 必须在输入维度的范围内。
    :attr:`index` 的值必须在输入的第 :attr:`dim` 维度的范围内。
    注意 ``input`` 和 ``index`` 不会互相广播。

    参数：
        - **input** (Tensor): 源张量
        - **dim** (int): :attr:`index` 沿的维度
        - **index** (Tensor): 包含要索引的 :attr:`index` 的一维张量
    
    示例：

    .. code-block:: python
    
        >>> import oneflow as flow
        >>> input = flow.tensor([[1,2,3],[4,5,6]], dtype=flow.int32)
        >>> input 
        tensor([[1, 2, 3],
                [4, 5, 6]], dtype=oneflow.int32)
        >>> index = flow.tensor([0,1], dtype=flow.int32)
        >>> output = flow.index_select(input, 1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
        >>> output = input.index_select(1, index)
        >>> output
        tensor([[1, 2],
                [4, 5]], dtype=oneflow.int32)
    
    """
)
