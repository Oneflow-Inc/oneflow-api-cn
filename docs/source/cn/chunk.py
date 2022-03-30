import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.chunk,
    r"""chunk(input, chunks, dim) -> Tensor
    
    将张量 :attr:`input` 拆分为特定数量的块。
    每个块都是输入张量的一个视图。 
    最后一个块会小于其他的块如果在指定的维度 :attr:`dim` 上 :attr:`input` 的大小不能被 :attr:`chunks` 整除。

    参数：
        - **input** (oneflow.tensor): 要拆分的张量
        - **chunks** (int): 要返回的块数。
        - **dim** (int): 拆分张量的维度。

    返回类型：
        包含 oneflow.tensor 的 List。

    示例：

    .. code-block:: python
    
        >>> import oneflow as flow

        >>> input = flow.randn(5, 3, 6, 9, dtype=flow.float32)
        >>> of_out = []
        >>> of_out = flow.chunk(input, chunks=3, dim=2)
        >>> chunks = 3
        >>> of_out_shape = []
        >>> for i in range(0, chunks):
        ...     of_out_shape.append(of_out[i].numpy().shape)
        >>> of_out_shape
        [(5, 3, 2, 9), (5, 3, 2, 9), (5, 3, 2, 9)]

        >>> input = flow.randn(5, 3, 6, 9, dtype=flow.float32)
        >>> of_out = []
        >>> of_out = flow.chunk(input, chunks=4, dim=3)
        >>> chunks = 4
        >>> of_out_shape = []
        >>> for i in range(0, chunks):
        ...     of_out_shape.append(of_out[i].numpy().shape)
        >>> of_out_shape
        [(5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 2), (5, 3, 6, 3)]

    """
)
