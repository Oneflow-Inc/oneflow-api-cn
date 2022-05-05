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
        >>> import numpy as np

        >>> arr = np.random.randn(5, 3, 6, 9).astype(np.float32)
        >>> input = flow.tensor(arr)
        >>> output = []
        >>> chunks = 3
        >>> output = flow.chunk(input, chunks=chunks, dim=2)
        >>> out_shape = []
        >>> for i in range(0, chunks):
        ...     out_shape.append(output[i].numpy().shape)
        >>> out_shape
        [(5, 3, 2, 9), (5, 3, 2, 9), (5, 3, 2, 9)]

    """
)
