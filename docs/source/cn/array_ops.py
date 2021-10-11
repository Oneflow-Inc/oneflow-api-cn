import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.diag,
    r"""diag(x, diagonal=0) -> Tensor

    如果 :attr:`x` 是一个向量（一维张量），返回一个个二维平方张量，其中 :attr:`x` 的元素作为对角线。
    如果 :attr:`x` 是一个矩阵（二维张量），返回一个一维张量，其元素为 :attr:`x` 的对角线元素。

    参数：
        - **x** (Tensor): 输入张量
        - **diagonal** (Optional[Int32], 0): 要考虑的对角线
            如果 diagonal = 0，则考虑主对角线，如果 diagonal > 0，则考虑主对角线上方，如果 diagonal < 0，则考虑主对角线下方。默认为0。
    
    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor(
        ...     [
        ...        [1.0, 2.0, 3.0],
        ...        [4.0, 5.0, 6.0],
        ...        [7.0, 8.0, 9.0],
        ...     ], dtype=flow.float32)
        >>> flow.diag(input)
        tensor([1., 5., 9.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.tril,
    r"""tril(x, diagonal=0) -> Tensor
    
    返回输入矩阵（二维张量）或矩阵批的沿指定对角线的下三角部分，结果张量的其他元素设置为 0。
    
    .. note::
        如果 diagonal = 0，返回张量的对角线是主对角线，
        如果 diagonal > 0，返回张量的对角线在主对角线之上，
        如果 diagonal < 0，返回张量的对角线在主对角线之下。

    参数：
        - **x** (Tensor): 输入张量 
        - **diagonal** (Optional[Int64], 0): 要考虑的对角线

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.ones(3, 3, dtype=flow.float32)
        >>> flow.tril(x)
        tensor([[1., 0., 0.],
                [1., 1., 0.],
                [1., 1., 1.]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.triu,
    r"""triu(x, diagonal=0) -> Tensor

    返回输入矩阵（二维张量）或矩阵批的沿指定对角线的上三角部分，结果张量的其他元素设置为 0。
    
    参数：
        - **x** (Tensor): 输入张量 
        - **diagonal** (Optional[Int64], 0): 要考虑的对角线

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.ones(3, 3, dtype=flow.float32)
        >>> flow.triu(x)
        tensor([[1., 1., 1.],
                [0., 1., 1.],
                [0., 0., 1.]], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.batch_gather,
    r"""batch_gather(in, indices) -> Tensor
    
    用 `indices` 重新排列元素
    
    参数：
        - **in** (Tensor): 输入张量 
        - **indices** (Tensor): 索引张量，它的数据类型必须是 int32/64。

    示例：

    示例1:

    .. code-block:: python
       
        >>> import oneflow as flow 

        >>> x = flow.tensor([[1, 2, 3], [4, 5, 6]], dtype=flow.float32)
        >>> indices = flow.tensor([1, 0], dtype=flow.int64)
        >>> out = flow.batch_gather(x, indices)
        >>> out
        tensor([[4., 5., 6.],
                [1., 2., 3.]], dtype=oneflow.float32)

    示例2：
    
    .. code-block:: python
        
        >>> import oneflow as flow 

        >>> x = flow.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]], dtype=flow.float32)
        >>> indices = flow.tensor([[1, 0], [0, 1]], dtype=flow.int64)
        >>> out = flow.batch_gather(x, indices)
        >>> out
        tensor([[[4., 5., 6.],
                 [1., 2., 3.]],
        <BLANKLINE>         
                [[1., 2., 3.],
                 [4., 5., 6.]]], dtype=oneflow.float32)
                 
    """,
)

reset_docstr(
    oneflow.argwhere,
    r"""argwhere(input, dtype=flow.int32) -> Tensor
    
    返回一个包含所有 :attr:`input` 中非 0 元素的 `index` 的列表。返回列表为一个 tensor，其中元素为坐标值。

    参数：
        - **input** (oneflow.Tensor): 输入张量
        - **dtype** (Optional[flow.dtype], 可选): 输出的数据类型，默认为 flow.int32

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([[0, 1, 0], [2, 0, 2]], dtype=flow.int32)
        >>> output = flow.argwhere(input)
        >>> output
        tensor([[0, 1],
                [1, 0],
                [1, 2]], dtype=oneflow.int32)

    """
)

reset_docstr(
    oneflow.broadcast_like,
    r"""broadcast_like(x, like_tensor, broadcast_axes=None) -> Tensor
    
    将 :attr:`x` 沿着 :attr:`broadcast_axes` 广播为 :attr:`like_tensor` 的形式。

    参数：
        - **x** (Tensor): 输入张量
        - **like_tensor** (Tensor): 参考张量  
        - **broadcast_axes** (Optional[Sequence], 可选): 想要广播的轴，默认为None。

    返回类型：
        oneflow.Tensor: 广播输入张量。

    示例：

    .. code:: python

        >>> import oneflow as flow 

        >>> x = flow.randn(3, 1, 1)
        >>> like_tensor = flow.randn(3, 4, 5)
        >>> broadcast_tensor = flow.broadcast_like(x, like_tensor, broadcast_axes=[1, 2]) 
        >>> broadcast_tensor.shape
        oneflow.Size([3, 4, 5])

    """
)

reset_docstr(
    oneflow.cat,
    r"""cat(inputs, dim=0) -> Tensor

    在指定的维度 :attr:`dim` 上连接两个或以上 tensor。

    类似于 `numpy.concatenate <https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html>`_

    参数：
        - **inputs** : 一个包含要连接的 `Tensor` 的 `list` 。
        - **dim** (int): 要连接的维度。

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input1 = flow.randn(2, 6, 5, 3, dtype=flow.float32)
        >>> input2 = flow.randn(2, 6, 5, 3, dtype=flow.float32)
        >>> input3 = flow.randn(2, 6, 5, 3, dtype=flow.float32)

        >>> out = flow.cat([input1, input2, input3], dim=1)
        >>> out.shape
        oneflow.Size([2, 18, 5, 3])
    
    """
)
