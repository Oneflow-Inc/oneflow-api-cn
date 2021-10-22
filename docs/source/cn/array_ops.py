import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.diag,
    r"""diag(x, diagonal=0) -> Tensor

    如果 :attr:`x` 是一个向量（一维张量），返回一个二维平方张量，其中 :attr:`x` 的元素作为对角线。
    如果 :attr:`x` 是一个矩阵（二维张量），返回一个一维张量，其元素为 :attr:`x` 的对角线元素。

    参数 :attr:`diagonal` 决定要考虑哪条对角线：
        - 如果 diagonal = 0，则考虑主对角线
        - 如果 diagonal > 0，则考虑主对角线上方
        - 如果 diagonal < 0，则考虑主对角线下方
    参数：
        - **x** (Tensor): 输入张量
        - **diagonal** (Optional[Int32], 0): 要考虑的对角线（默认为0）
    
    返回类型：
        oneflow.tensor

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
    
    返回输入矩阵（二维张量）或矩阵批次的沿指定对角线的下三角部分，结果张量的其他元素设置为 0。
    
    .. note::
        - 如果 diagonal = 0，返回张量的对角线是主对角线
        - 如果 diagonal > 0，返回张量的对角线在主对角线之上
        - 如果 diagonal < 0，返回张量的对角线在主对角线之下

    参数：
        - **x** (Tensor): 输入张量 
        - **diagonal** (Optional[Int64], 0): 要考虑的对角线

    返回类型：
        oneflow.tensor

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

    返回输入矩阵（二维张量）或矩阵批次的沿指定对角线的上三角部分，结果张量的其他元素设置为 0。
    
    参数：
        - **x** (Tensor): 输入张量 
        - **diagonal** (Optional[Int64], 0): 要考虑的对角线

    返回类型：
        oneflow.tensor

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
        - **input** (oneflow.tensor): 输入张量
        - **dtype** (Optional[flow.dtype], 可选): 输出的数据类型，默认为 flow.int32

    返回类型：
        oneflow.tensor

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
        - **broadcast_axes** (Optional[Sequence], 可选): 想要广播的维度，默认为None。

    返回类型：
        oneflow.tensor: 广播输入张量。

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
        oneflow.tensor

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

reset_docstr(
    oneflow.eye,
    r"""eye(n, m=None, dtype=flow.float, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor
    
    返回一个二维 tensor ，对角线上的元素为 1 ，其他元素为 0 。

    参数：
        - **n** (int): 行数
        - **m** (Optional[int], 可选): 列数，如果为 None ，则默认与 `n` 相值。
    
    关键词参数：
        - **device** (flow.device, 可选): 返回张量的所需设备。如果为 None ，则使用当前设备作为默认张量。
        - **requires_grad** (bool, 可选): 使用 autograd 记录对返回张量的操作。默认值： `False` 。
    
    返回值：
        oneflow.tensor: 对角线上为 1，其他地方为 0 的 tensor 。
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> out = flow.eye(3, 3)
        >>> out
        tensor([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]], dtype=oneflow.float32)
    
    """
)

reset_docstr(
    oneflow.flatten,
    r"""flatten(input, start_dim=0, end_dim=-1) -> Tensor
    
    将 tensor 指定连续范围的维度展平。

    参数：
        - **start_dim** (int): 起始维度 (默认为 0)。
        - **end_dim** (int): 结束维度 (默认为 -1)。
    
    示例：

    .. code-block:: python 

        >>> import oneflow as flow
        >>> input = flow.Tensor(32, 1, 5, 5)
        >>> output = input.flatten(start_dim=1)
        >>> output.shape
        oneflow.Size([32, 25])

    """
)

reset_docstr(
    oneflow.flip, 
    r"""flip(input, dims) -> Tensor
    
    沿指定维度 :attr:`dims` 反转 n-D 张量的顺序。

    .. note::
        有别于 NumPy 的算子 `np.flip` 在一定时间内返回一个 :attr:`input` 的视图，
        `oneflow.flip` 创建一个 :attr:`input` 的备份。因为创建 tensor 的备份所需的工作量比查看数据 `oneflow.flip` 比 `np.flip` 慢。

    参数：
        - **input** (Tensor): 输入张量
        - **dims** (list 或者 tuple): 要翻转的维度
        
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.arange(0, 8, dtype=flow.float32).reshape(2, 2, -1)
        >>> input.shape
        oneflow.Size([2, 2, 2])
        >>> out = flow.flip(input, [0, 1])
        >>> out
        tensor([[[6., 7.],
                 [4., 5.]],
        <BLANKLINE>
                [[2., 3.],
                 [0., 1.]]], dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.masked_fill,
    r"""masked_fill(input, mask, value) -> Tensor

    如果参数 :attr:`mask` 为 True ，则在 :attr:`input` 中填充 :attr:`value` 。
    参数 :attr:`mask` 的形状必须能被广播为 :attr:`input` 的形状。

    参数：
        - **input** (Tensor): 被填充的张量
        - **mask** (BoolTensor): 决定是否填充的 boolean 张量
        - **value** (float): 要填充的值

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> fill_value = 8.7654321 # random value e.g. -1e9 3.1415
        >>> input = flow.tensor([[[-0.13169311,  0.97277078,  1.23305363,  1.56752789],
        ...                       [-1.51954275,  1.87629473, -0.53301206,  0.53006478],
        ...                       [-1.38244183, -2.63448052,  1.30845795, -0.67144869]],
        ...                      [[ 0.41502161,  0.14452418,  0.38968   , -1.76905653],
        ...                       [ 0.34675095, -0.7050969 , -0.7647731 , -0.73233418],
        ...                       [-1.90089858,  0.01262963,  0.74693893,  0.57132389]]], dtype=flow.float32)
        >>> mask = flow.gt(input, 0)
        >>> output = flow.masked_fill(input, mask, fill_value)
        >>> output
        tensor([[[-0.1317,  8.7654,  8.7654,  8.7654],
                 [-1.5195,  8.7654, -0.5330,  8.7654],
                 [-1.3824, -2.6345,  8.7654, -0.6714]],
        <BLANKLINE>
                [[ 8.7654,  8.7654,  8.7654, -1.7691],
                 [ 8.7654, -0.7051, -0.7648, -0.7323],
                 [-1.9009,  8.7654,  8.7654,  8.7654]]], dtype=oneflow.float32)

    """    
)

reset_docstr(
    oneflow.masked_select,
    r"""masked_select(input, mask) -> Tensor

    返回一个新的 tensor ，其元素为依据 :attr:`mask` 的真实值在 :attr:`input` 中索引的元素。

    :attr:`mask` 是一个 BoolTensor （在 oneflow 中， BoolTensor 被替换为 Int8Tensor ）

    参数 :attr:`mask` 的形状必须能被广播为 :attr:`input` 的形状。

    参数：
        - **input** (Tensor): 输入张量
        - **mask** (Tensor): 包含要索引的二进制掩码的张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]], dtype=flow.float32)
        >>> mask = input.gt(0.05)
        >>> out = flow.masked_select(input, mask)
        >>> out
        tensor([0.3139, 0.3898], dtype=oneflow.float32)
    
    """
)

reset_docstr(
    oneflow.nonzero,
    r"""nonzero(input, as_tuple=False) -> Tensor or tuple of Tensors

    .. note::
        当 :attr:`as_tuple` 默认为 ``False`` 时，返回一个 2-D tensor，每一行是一个非 0 值的索引（index）。

        当 :attr:`as_tuple` 为 ``True`` 时，返回一个包含 1-D 索引张量的元组，允许高级索引，
        因此 ``x[x.nonzero(as_tuple=True)]`` 给出 ``x`` 的所有非 0 值。在返回的元组中，每个
        索引张量包含特定维度的非 0 元素的索引。

        有关这两种情况的更多详细信息，请参见下文。

    **当** :attr:`as_tuple` **为** ``False`` **时（默认）** ：

    返回一个 tensor 包含所有 :attr:`input` 中非 0 元素的索引（index）。结果中的每一行包含 :attr:`input` 中一个
    非 0 元素的索引，结果按字典顺序排序，对最后一个索引的改变最快（C-style）。

    如果 :attr:`input` 的维度数为 :math:`n` ，则结果索引张量 :attr:`out` 的形状为 :math:`(z \\times n)` ，
    :math:`z` 是 :attr:`input` 中非 0 元素的数量。

    **当** :attr:`as_tuple` **为** ``True`` **时**：

    返回一个包含 1-D 张量的元组，每个张量分别对应 :attr:`input` 的每个维度，并包含 :attr:`input` 当前维度中所有非
    0 元素的索引。

    如果 :attr:`input` 的维度数为 :math:`n` ，则此返回元组包含 :math:`n` 个大小为 :math:`z` 的张量， :math:`z` 
    为 :attr:`input` 中非 0 元素的总数。

    有一种特殊情况是，当 :attr:`input` 是 0 维张量并且有一个非 0 的标量值，则被视为一个只有一个元素的 1 维张量

    参数：
        **input** (Tensor): 输入张量

    关键词参数：
        **out** (Tensor, optional): 包含索引的输出张量

    返回类型：
        如果 :attr:`as_tuple` 为 ``False`` ，则返回 oneflow.tensor ，其元素为 :attr:`input` 中的索引。
        如果 :attr:`as_tuple` 为 ``True`` ，则返回包含 oneflow.tensor 的元组，
        每个张量包含非 0 元素在当前维度的索引。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.nonzero(flow.tensor([1, 1, 1, 0, 1]))
        tensor([[0],
                [1],
                [2],
                [4]], dtype=oneflow.int32)
        >>> flow.nonzero(flow.tensor([[0.6, 0.0, 0.0, 0.0],
        ...                             [0.0, 0.4, 0.0, 0.0],
        ...                             [0.0, 0.0, 1.2, 0.0],
        ...                             [0.0, 0.0, 0.0,-0.4]]))
        tensor([[0, 0],
                [1, 1],
                [2, 2],
                [3, 3]], dtype=oneflow.int32)
        >>> flow.nonzero(flow.tensor([1, 1, 1, 0, 1]), as_tuple=True)
        (tensor([0, 1, 2, 4], dtype=oneflow.int32),)
        >>> flow.nonzero(flow.tensor([[0.6, 0.0, 0.0, 0.0],
        ...                             [0.0, 0.4, 0.0, 0.0],
        ...                             [0.0, 0.0, 1.2, 0.0],
        ...                             [0.0, 0.0, 0.0,-0.4]]), as_tuple=True)
        (tensor([0, 1, 2, 3], dtype=oneflow.int32), tensor([0, 1, 2, 3], dtype=oneflow.int32))
        >>> flow.nonzero(flow.tensor(5), as_tuple=True)
        (tensor([0], dtype=oneflow.int32),)

    """
)

reset_docstr(
    oneflow.reshape,
    r"""reshape(input, shape=None) -> Tensor

    返回一个新张量，此张量的内容为输入 :attr:`input`，形状为指定的 :attr:`shape`。

    我们可以将 :attr:`shape` 中的某一个维度设置为 `-1` ，算子会自动推断出完整的形状。

    参数：
        - **input**: 输入张量
        - **shape**: 输出张量的形状

    返回类型：
        oneflow.tensor: 数据类型与 :attr:`input` 相同的张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor(
        ...    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=flow.float32
        ... )

        >>> y = flow.reshape(input, shape=[2, 2, 2, -1]).shape
        >>> y
        oneflow.Size([2, 2, 2, 2])

    """
)
