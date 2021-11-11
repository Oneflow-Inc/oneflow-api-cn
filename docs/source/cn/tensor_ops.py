import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.is_floating_point,
    r"""is_floating_point(input) -> boolean

    如果 :attr:`input` 的数据类型是浮点数据类型，则返回 True 。浮点数据类型为 flow.float64、flow.float32 或者 flow.float16 。

    参数：
        **input** (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1, 2, 3, 4, 5], dtype=flow.int)
        >>> output = flow.is_floating_point(input)
        >>> output
        False
    
    """
)

reset_docstr(
    oneflow.is_nonzero,
    r"""is_nonzero(input) -> Tensor
    
    如果 :attr:`input` 是转换类型后不等于值为 0 的单元素张量的张量
    （ :attr:`flow.tensor([0.])` 或者 :attr:`flow.tensor([0])` ）返回 True 。

    报告 :attr:`RuntimeError` 如果 :attr:`input.shape.numel()!=1`

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> flow.is_nonzero(flow.tensor([0.]))
        False
        >>> flow.is_nonzero(flow.tensor([1.5]))
        True
        >>> flow.is_nonzero(flow.tensor([3]))
        True

    """
)

reset_docstr(
    oneflow.negative,
    r"""negative(input) -> Tensor

    返回 :attr:`input` 的负值。

    参数：
        - **input** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1.0, -1.0, 2.3], dtype=flow.float32)
        >>> out = flow.negative(input)
        >>> out
        tensor([-1.0000,  1.0000, -2.3000], dtype=oneflow.float32)

    """
)


reset_docstr(
    oneflow.Tensor.add,
    r"""add(input, other) -> Tensor
    
    计算 `input` 和 `other` 的和。支持 element-wise、标量和广播形式的加法。

    公式为：

    .. math::
        out = input + other

    参数：
        - **input** (Tensor): 输入张量
        - **other** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        # element-wise 加法
        >>> x = flow.randn(2, 3, dtype=flow.float32)
        >>> y = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.add(x, y)
        >>> out.shape
        oneflow.Size([2, 3])

        # 标量加法
        >>> x = 5
        >>> y = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.add(x, y)
        >>> out.shape
        oneflow.Size([2, 3])

        # 广播加法
        >>> x = flow.randn(1, 1, dtype=flow.float32)
        >>> y = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.add(x, y)
        >>> out.shape
        oneflow.Size([2, 3])

    """,
)

reset_docstr(
    oneflow.Tensor.cosh,
    r"""cosh(x) -> Tensor

    返回一个包含 :attr:`x` 中元素的双曲余弦值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \cosh(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([ 0.1632,  1.1835, -0.6979, -0.7325], dtype=flow.float32)
        >>> output = flow.cosh(input)
        >>> output
        tensor([1.0133, 1.7860, 1.2536, 1.2805], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.Tensor.cos,
    r"""cos(x) -> Tensor
    返回一个包含 :attr:`x` 中元素的余弦值的新 tensor。

    公式为：

    .. math::
        \text{out}_{i} = \cos(\text{input}_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([1.4309,  1.2706, -0.8562,  0.9796], dtype=flow.float32)
        >>> output = flow.cos(input)

    """,
)

reset_docstr(
    oneflow.Tensor.div,
    r"""div(input, other) -> Tensor
    
    计算 `input` 除以 `other`，支持 element-wise、标量和广播形式的除法。

    公式为：

    .. math::
        out = \frac{input}{other}
    
    参数：
        - **input** (Union[int, float, oneflow.tensor]): input.
        - **other** (Union[int, float, oneflow.tensor]): other.

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        # element-wise 除法
        >>> input = flow.randn(2, 3, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.div(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 标量除法
        >>> input = 5
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.div(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 广播除法
        >>> input = flow.randn(1, 1, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.div(input,other)
        >>> out.shape 
        oneflow.Size([2, 3])

    """,
)

reset_docstr(
    oneflow.Tensor.le,
    r"""
    返回 :math:`input <= other` 的 element-wise 真实值。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **other** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor: 数据类型为 int8 的张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input1 = flow.tensor([1, 2, 3], dtype=flow.float32)
        >>> input2 = flow.tensor([1, 1, 4], dtype=flow.float32)

        >>> out = flow.le(input1, input2)
        >>> out
        tensor([1, 0, 1], dtype=oneflow.int8)

    
    """
)

reset_docstr(
    oneflow.Tensor.log,
    r"""log(x) -> Tensor

    返回一个新 tensor 包含 :attr:`x` 中元素的自然对数。
    公式为：

    .. math::
        y_{i} = \log_{e} (x_{i})

    参数：
        **x** (Tensor): 输入张量

    返回类型：
        oneflow.tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow  

        >>> input = flow.randn(2, 3, 4, 5, dtype=flow.float32)
        >>> output = flow.log(input)

    """,
)

reset_docstr(
    oneflow.Tensor.long,
    r"""long(input) -> Tensor

    `Tensor.long()` 等价于 `Tensor.to(flow.int64)` 。 参考 :func:`oneflow.to` 。

    参数：
        - **input**  (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.randn(1, 2, 3, dtype=flow.float32)
        >>> input = input.long()
        >>> input.dtype
        oneflow.int64
    """
)

reset_docstr(
    oneflow.rsqrt,
    r"""rsqrt(input) -> Tensor
        
        返回一个新的张量，它的元素是 :attr:`input` 的每个元素的平方根的倒数。

        .. math::
            \text{out}_{i} = \frac{1}{\sqrt{\text{input}_{i}}}

        参数：
            - **input** (Tensor): 输入张量

        示例：

        .. code-block:: python

            >>> import oneflow as flow
            
            >>> a = flow.tensor([1.0, 2.0, 3.0], dtype=flow.float32)
            >>> out = flow.rsqrt(a)
            >>> out
            tensor([1.0000, 0.7071, 0.5774], dtype=oneflow.float32)
    """,    
)

reset_docstr(
    oneflow.Tensor.sort,
    r"""sort(input, dim=-1, descending=False) -> tuple(values, indices)
    
    按值升序沿给定维度 :attr:`dim` 对张量 :attr:`input` 的元素进行排序。

    参数：
        - **input** (oneflow.tensor): 输入张量
        - **dim** (int, 可选): 要排序的维度，默认为（dim = -1）
        - **descending** (bool, 可选): 控制排序方式（升序或降序）

    返回类型：
        Tuple(oneflow.tensor, oneflow.tensor(dtype=int32)): 一个元素为 (values, indices) 的元组
        元素为排序后的 :attr:`input` 的元素，索引是原始输入张量 :attr:`input` 中元素的索引。

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([[1, 3, 8, 7, 2], [1, 9, 4, 3, 2]], dtype=flow.float32)
        >>> (values, indices) = flow.sort(input)
        >>> values
        tensor([[1., 2., 3., 7., 8.],
                [1., 2., 3., 4., 9.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 4, 1, 3, 2],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> (values, indices) = flow.sort(input, descending=True)
        >>> values
        tensor([[8., 7., 3., 2., 1.],
                [9., 4., 3., 2., 1.]], dtype=oneflow.float32)
        >>> indices
        tensor([[2, 3, 1, 4, 0],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> (values, indices) = flow.sort(input, dim=0)
        >>> values
        tensor([[1., 3., 4., 3., 2.],
                [1., 9., 8., 7., 2.]], dtype=oneflow.float32)
        >>> indices
        tensor([[0, 0, 1, 1, 0],
                [1, 1, 0, 0, 1]], dtype=oneflow.int32)
 
    """
)

reset_docstr(
    oneflow.Tensor.split,
    r"""split(x, split_size_or_sections, dim=0) -> Tensor
    
    将张量分成块。

    如果 :attr:`split_size_or_sections` 为一个整数，则 :attr:`x` 会被分成等大的块。
    如果给定维度 :attr:`dim` 上的 tensor 大小不能被 split_size 整除，则最后一块的大小会小于其它块。

    如果 :attr:`split_size_or_sections` 是一个列表，
    那么 :attr:`x` 将根据 :attr:`split_size_or_sections` 被拆分为 :attr:`len(split_size_or_sections)` 个大小为 :attr:`dim` 的块。

    参数：
        - **x** (Tensor): 要拆分的张量
        - **split_size_or_sections** (Union[int, List[int]]): 单个块的大小或包含每个块大小的列表
        - **dim** (int): 拆分张量所沿的维度。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> a = flow.arange(10).view(5, 2)
        >>> flow.split(a, 2)
        (tensor([[0, 1],
                [2, 3]], dtype=oneflow.int64), tensor([[4, 5],
                [6, 7]], dtype=oneflow.int64), tensor([[8, 9]], dtype=oneflow.int64))
        >>> flow.split(a, [1, 4])
        (tensor([[0, 1]], dtype=oneflow.int64), tensor([[2, 3],
                [4, 5],
                [6, 7],
                [8, 9]], dtype=oneflow.int64))

    """
)

reset_docstr(
    oneflow.Tensor.sqrt,
    r"""返回一个元素为 :attr:`input` 元素平方根的新 tensor 。
        公式为：

        .. math::
            \text{out}_{i} = \sqrt{\text{input}_{i}}

        参数：
            - **input** (Tensor): 输入张量

        示例：

        .. code-block:: python

            >>> import oneflow as flow
            
            >>> input = flow.tensor([1.0, 2.0, 3.0], dtype=flow.float32)
            >>> output = flow.sqrt(input)
            >>> output
            tensor([1.0000, 1.4142, 1.7321], dtype=oneflow.float32)
    """,
)

reset_docstr(
    oneflow.Tensor.square,
    r"""square(x)  -> Tensor

    返回一个新的张量，其元素为 :attr:`x` 中元素的的平方。

    .. math::
        \text{out}_{i} = \sqrt{\text{input}_{i}}

    参数：
        **x** (Tensor): 输入张量
    
    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
            
        >>> input = flow.tensor([1.0, 2.0, 3.0], dtype=flow.float32)
        >>> output = flow.square(input)
        >>> output
        tensor([1., 4., 9.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.Tensor.sub,
    r"""sub(input, other) -> Tensor
    
    计算 `input` 和 `other` 的差，支持 element-wise、标量和广播形式的减法。

    公式为：

    .. math::
        out = input - other

    参数：
        - **input** (Tensor): 输入张量
        - **other** (Tensor): 输入张量

    返回类型：   
        oneflow.tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        # element-wise 减法
        >>> input = flow.randn(2, 3, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.sub(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 标量减法
        >>> input = 5
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.sub(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

        # 广播减法
        >>> input = flow.randn(1, 1, dtype=flow.float32)
        >>> other = flow.randn(2, 3, dtype=flow.float32)
        >>> out = flow.sub(input,other)
        >>> out.shape
        oneflow.Size([2, 3])

    """,
)

reset_docstr(
    oneflow.Tensor.tile,
    r"""tile(input, reps) -> Tensor
    
    
    此接口与 PyTorch 一致。
    文档参考自：
    https://pytorch.org/docs/stable/generated/torch.tile.html


    
    通过重复 :attr:`input` 的元素构造一个新张量。 :attr:`reps` 参数指定每个维度的重复次数。

    如果 :attr:`reps` 的长度小于 :attr:`input` 的维度，则在 :attr:`reps` 前添加 1 。直到 :attr:`reps` 的长度
    等于 :attr:`input` 的维度。例如： :attr:`input` 的形状为  (8, 6, 4, 2)  ，而 :attr:`reps` 为 (2, 2) ，
    则 :attr:`reps` 被认为是 (1, 1, 2, 2) 。

    类似地，如果 :attr:`input` 的维度少于 :attr:`reps` 指定的维度，则 :attr:`input` 被视为在维度 0 处未压缩，
    直到它的维度与 :attr:`reps` 指定的一样多。例如，如果 :attr:`input` 的形状为 (4, 2) 而 ``reps`` 为 (3, 3, 2, 2)，
    则视 :attr:`input` 形状为 (1, 1, 4, 2)。

    .. note::
        这个函数类似于 NumPy 的 tile 函数。

    参数：
        - **input** (oneflow.tensor): 要重复元素的张量
        - **reps** (元组): 每个维度要重复的次数

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([1, 2], dtype=flow.int32)
        >>> out = input.tile(reps=(2,))
        >>> out
        tensor([1, 2, 1, 2], dtype=oneflow.int32)

        >>> input = flow.randn(5, 2, 1)
        >>> out = input.tile(reps=(3, 4))
        >>> out.size()
        oneflow.Size([5, 6, 4])
    """
)

reset_docstr(
    oneflow.Tensor.to,
    r"""to(input, *args, **kwargs) -> Tensor
    
    执行张量 dtype 和 device 转换。
        flow.dtype 和 flow.device 由 `input.to(*args, **kwargs)` 的参数推导而来。

    .. note::
        如果 tensor :attr:`input` 的 :class:`flow.dtype` 已经与参数一致，则返回 :attr:`input` 。
        否则创建一个符合条件的 :attr:`input` 备份。

    参数：
        - **input** (oneflow.tensor): 输入张量

    返回类型：
        oneflow.tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.randint(1, 9, size=(1, 2, 3, 4))
        >>> output = input.to(dtype=flow.float32)
        >>> flow.eq(input, output)
        tensor([[[[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]],
        <BLANKLINE>  
                 [[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 1]]]], dtype=oneflow.int8)
        >>> cuda0 = flow.device('cuda:0')
        >>> output = input.to(device=cuda0)
        
    """
)

reset_docstr(
    oneflow.Tensor.to_consistent,
    r"""to_consistent(placement=None, sbp=None, grad_sbp=None) -> Tensor

    将 local tensor 转换为 consistent tensor 或者将 consistent tensor 转化为
    具有不同 sbp 或 placement 的另一个 consistent tensor 。

    参数：
        - **input** (Tensor): 输入张量
        - **placement** (flow.placement, 可选): 设置返回张量的 placement 属性。如果为None，则 :attr:`input` 必须为  consistent tensor ，输出的 placement 将被设置为 :attr:`input` 的 placement 。默认： None
        - **sbp** (flow.sbp.sbp or tuple of flow.sbp.sbp, 可选): 返回的 consistent tensor 的 sbp 属性。如果为 None ，输入张量必须是 consistent tensor 的并使用它自己的 sbp 。默认：None

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([0.5, 0.6, 0.7], dtype=flow.float32)
        >>> placement = flow.placement("cpu", {0:range(1)})
        >>> output_tensor = input.to_consistent(placement, [flow.sbp.split(0)])
        >>> output_tensor.is_consistent
        True

    """
)

reset_docstr(
    oneflow.to_local,
    r"""to_local(input) -> Tensor

    返回 consistent tensor :attr:`input` 的 local tensor 。


    参数：
        - **input** (Tensor): 输入张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.tensor([0.5, 0.6, 0.7], dtype=flow.float32)
        >>> placement = flow.placement("cpu", {0:range(1)})
        >>> consistent_tensor = input.to_consistent(placement, [flow.sbp.split(0)])
        >>> consistent_tensor.to_local()
        tensor([0.5000, 0.6000, 0.7000], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.Tensor.tolist,
    r"""tolist(input) -> Tensor

    将 tensor 作为（嵌套）列表返回。对于标量，返回一个标准的 Python 数字，和 `oneflow.Tensor.item()` 一样。
    必要情况下 tensor 会被自动移动到 GPU 。

    此操作不可导的。

    参数：
        - **input** (Tensor): 输入张量

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> input = flow.tensor([[1,2,3], [4,5,6]])
        >>> input.tolist()
        [[1, 2, 3], [4, 5, 6]]
    """
)

reset_docstr(
    oneflow.Tensor.view,
    r"""view(input, *shape) -> Tensor
    
    此接口与 PyTorch 一致。 文档参考自：https://pytorch.org/docs/stable/generated/torch.Tensor.view.html

    
    返回一个新的 tensor ，其数据与 :attr:`input` 相同，但形状 :attr:`shape` 不同。

    返回的 tensor 与 :attr:`input` 共享相同的数据并且必须具有相同数量的元素，但是形状可以不同。
    对于要被查看的 tensor ，新的视图大小必须与其原始大小和 step 兼容，每个新视角的维度必须为原始维度的子空间，
    或者为跨越原始维度 :math:`d, d+1, \dots, d+k` 的 span 满足以下类似邻接条件 :math:`\forall i = d, \dots, d+k-1` 。

    .. math::

      \text{stride}[i] = \text{stride}[i+1] \times \text{size}[i+1]

    
    否则将无法以视图查看  为形状 :attr:`shape` 且不复制 :attr:`input` （例如通过 :meth:`contiguous`）。
    当不清楚是否可以执行 :meth:`view` 时，建议使用 :meth:`reshape` ，因为 :meth:`reshape` 在兼容的时候返回
    :attr:`input` 的视图，不兼容的时候复制 :attr:`input` （相当于调用 :meth:`contiguous` ）。

    参数：
        - **input** (Tensor)
        - **shape** (flow.Size 或 int...)

    返回类型：
        与 :attr:`input` 数据类型相同的 tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=flow.float32)

        >>> y = input.view(2, 2, 2, -1).shape
        >>> y
        oneflow.Size([2, 2, 2, 2])

    """

)

reset_docstr(
    oneflow.Tensor.where, 
    r"""where(condition, x=None, y=None) -> Tensor
    
    返回一个 tensor 其元素为从 :attr:`x` 或 :attr:`y` 中依据 :attr:`condition` 的真实值选择的元素，
    如果 :attr:`condition` 中的元素大于 0 ，则取 :attr:`x` 中的元素，否则取 :attr:`y` 的元素。

    .. note::
        如果 :attr:`x` 为 None 并且 :attr:`y` 为 None ，则 flow.where(condition) 等同于 
        flow.nonzero(condition, as_tuple=True) 。
        
        :attr:`condition` 、 :attr:`x` 、 :attr:`y` 必须可互相广播。

    参数：
        - **condition** (IntTensor): 如果不为 0 则 yield x ，否则 yield y
        - **x** (Tensor 或 Scalar): 当 :attr:`condition` 为 True 时，如果 :attr:`x` 为标量则为值，如果 :attr:`x` 不为标量则为在索引处选择的值
        - **y** (Tensor 或 Scalar): 当 :attr:`condition` 为 False 时，如果 :attr:`x` 为标量则为值，如果 :attr:`x` 不为标量则为在索引处选择的值
    
    返回类型：
        oneflow.tensor: 与 :attr:`condition` 、 :attr:`x` 、 :attr:`y` 广播形状相同的 tensor 。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]], dtype=flow.float32)
        >>> y = flow.ones(3, 2, dtype=flow.float32)
        >>> condition = flow.tensor([[0, 1], [1, 0], [1, 0]], dtype=flow.int32)
        >>> out = condition.where(x, y)
        >>> out #doctest: +ELLIPSIS
        tensor([[1.0000, 0.3139],
                ...
                [0.0478, 1.0000]], dtype=oneflow.float32)

    """
)
