import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.ones_like,
    r"""ones_like(x) -> Tensor

    返回一个元素全部为值为 1 的标量，且形状与 `x` 相同的 tensor。
    flow.ones_like(x) 等价于 flow.ones(x.shape, dtype=x.dtype)

    参数：
        **x** (Tensor): 输入的形状将会决定输出的形状

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(5, dtype=flow.float32)
        >>> y = flow.ones_like(x)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.zeros_like,
    r"""zeros_like(x) -> Tensor

    返回一个元素全部为值为 0 的标量，形状和 `x` 相同的 tensor。
    flow.zeros_like(x) 等价于 flow.zeros(x.shape, dtype=x.dtype)

    参数：
        **x** (Tensor): 输入的形状将决定输出的形状

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(5, dtype=flow.float32)
        >>> y = flow.zeros_like(x)
        >>> y
        tensor([0., 0., 0., 0., 0.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.empty,
    r"""empty(*size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    返回一个填充了未初始化数据的张量。 张量的形状由变量参数 ``size`` 定义。

    参数：
        - **size** (int... 或 oneflow.Size):  定义输出张量的形状。可以是可变数量的参数或集合，如列表或元组或 oneflow.Size
        - **dtype** (flow.dtype, 可选的): 返回张量的数据类型。默认：flow.float32
        - **device** (torch.device, 可选的): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选的): 设置返回张量的 placement 属性。如果为None，则构造 local tensor
        - **sbp** (flow.sbp 或 List[flow.sbp], 可选的): 返回的consistent tensor的所需 sbp
        - **requires_grad** (bool, 可选的): 用 autograd 记录对返回张量的操作，默认为 False

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.empty(4, 5)  # 构造空 local tensor
        >>> y.shape
        oneflow.Size([4, 5])
        >>> y.is_consistent
        False
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> y = flow.empty(4, 5, placement=placement, sbp=flow.sbp.broadcast)  # 构造空 consistent tensor
        >>> y.is_consistent
        True

    """)

reset_docstr(
    oneflow.ones,
    r"""ones(*size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False,) -> Tensor
    
    返回一个元素全部为标量 1 ，形状由参数 :attr:`size` 决定的 tensor。

    参数：
        - **size** (一个整数或包含整数的元组)): 决定输出张量的形状，可以是数字变量或集合例如列表或元组
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 设置返回张量的 placement 属性。如果为 None，则构造 local tensor 
        - **sbp** (flow.sbp.sbp 或包含 flow.sbp.sbp 的元组, 可选): 返回的consistent tensor的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.ones(5)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)
        >>> y = flow.ones(2,3) # 构造 local tensor
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> y = flow.ones(4, 5, placement=placement, sbp=flow.sbp.broadcast) # 构造 consistent tensor
        >>> y.is_consistent
        True


    """
)

reset_docstr(
    oneflow.Tensor.new_ones,
    r"""new_ones(size=None, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor
 
    放回一个 :attr:`size` 大小的张量，其元素全部为 1 。默认情况下，返回张量的 `dtype` 和 `device` 和输入张量的相同。

    参数：
        - **size** (一个整数或包含整数的元组): 决定输出张量的形状，可以是数字变量或集合例如列表或元组
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型。
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 设置返回张量的 placement 属性。如果为None，则构造 local tensor 。
        - **sbp** (flow.sbp.sbp 或包含 flow.sbp.sbp 的元组, 可选): 返回的consistent tensor的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量。
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False。

    参数：

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.ones((1, 2, 3), dtype=flow.float32)
        >>> y = x.new_ones((2, 2), dtype=flow.float32)
        >>> y
        tensor([[1., 1.],
                [1., 1.]], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.linalg.norm,
    r"""linalg.norm(input, ord=None, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor

    返回 :attr:`input` 的矩阵范数或向量范数。

    此函数可以计算八种不同类型的矩阵范数之一，或无限数量的向量范数之一。具体取决于减少维度的数量和 :attr:`ord` 参数的值。

    参数：
        - **input** (Tensor): 输入张量。如果 :attr:`dim` 是 None，则输入必须是一维或二维的，除非 :attr:`ord` 也是 None。
            
            如果 :attr:`dim` 和 :attr:`ord` 都是 None ，则将返回扁平化为 1-D 的输入二范数。它的数据类型必须是浮点数或复数类型。
            对于复杂输入，范数是根据每个元素的绝对值计算的。
            
            如果输入是复数并且既没有指定 :attr:`dtype` 也没有指定 :attr:`out` ，
            返回的数据类型将是相应的浮点类型（例如如果 :attr:`input` 是 complexfloat，则为 float）

        - **ord** (int, inf, -inf, 'fro', 'nuc', 可选): 范数的顺序。默认为 `'None'` 。
            可以计算以下范数：

            ==============  ============================  =================================
            :attr:`ord`              矩阵范数                          向量范数
            ==============  ============================  =================================
            None             Frobenius norm                `2`-norm
            `'fro'`          Frobenius norm                -- not supported --
            `'nuc'`          -- 尚不支持 --                 -- not supported --
            `inf`            `max(sum(abs(x), dim=1))`     `max(abs(x))`
            `-inf`           `min(sum(abs(x), dim=1))`     `min(abs(x))`
            `0`              -- 尚不支持 --                 `sum(x != 0)`
            `1`              `max(sum(abs(x), dim=0))`     as below
            `-1`             `min(sum(abs(x), dim=0))`     as below
            `2`              -- 尚不支持 --                 as below
            `-2`             -- 尚不支持 --                 as below
            其他             -- 不支持 --                  `sum(abs(x)^{ord})^{(1 / ord)}`
            ==============  ============================  =================================

            其中 `inf` 指的是 `float('inf')` 、 NumPy 的 `inf` 对象或任何等效对象。

        - **dim** (int, 2-tuple of ints, 2-list of ints, 可选): 如果 :attr:`dim` 是一个 int，向量范数将在指定的维度上计算。
            
            如果 :attr:`dim` 是一个整数的二元组，矩阵范数将在指定的维度上计算。
            
            如果 :attr:`dim` 为 None ，则输入张量是二维张量时计算矩阵范数，输入张量为一维时计算向量范数。默认值：``None``

        - **keepdim** (bool, 可选): 如果设定为 True ，则减少的维度将作为大小为 1 的维度保留在结果中。默认值：``False``

        - **out** (Tensor, 可选): 输出张量

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> from oneflow import linalg as LA
        >>> a = flow.arange(9, dtype=flow.float32) - 4
        >>> a
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=oneflow.float32)
        >>> b = a.reshape(3, 3)
        >>> b
        tensor([[-4., -3., -2.],
                [-1.,  0.,  1.],
                [ 2.,  3.,  4.]], dtype=oneflow.float32)
        >>> LA.norm(a)
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(b)
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(b, 'fro')
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(a, float('inf'))
        tensor(4., dtype=oneflow.float32)
        >>> LA.norm(b, float('inf'))
        tensor(9., dtype=oneflow.float32)
        >>> LA.norm(a, -float('inf'))
        tensor(0., dtype=oneflow.float32)
        >>> LA.norm(b, -float('inf'))
        tensor(2., dtype=oneflow.float32)
        >>> LA.norm(a, 1)
        tensor(20., dtype=oneflow.float32)
        >>> LA.norm(b, 1)
        tensor(7., dtype=oneflow.float32)
        >>> LA.norm(a, -1)
        tensor(0., dtype=oneflow.float32)
        >>> LA.norm(b, -1)
        tensor(6., dtype=oneflow.float32)
        >>> LA.norm(a, 2)
        tensor(7.7460, dtype=oneflow.float32)
        >>> LA.norm(a, -2)
        tensor(0., dtype=oneflow.float32)
        >>> LA.norm(a, 3)
        tensor(5.8480, dtype=oneflow.float32)
        >>> LA.norm(a, -3)
        tensor(0., dtype=oneflow.float32)
        >>> c = flow.tensor([[1., 2., 3.],
        ...                   [-1, 1, 4]])
        >>> LA.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000], dtype=oneflow.float32)
        >>> LA.norm(c, dim=1, keepdim = True)
        tensor([[3.7417],
                [4.2426]], dtype=oneflow.float32)
        >>> LA.norm(c, ord=1, dim=1)
        tensor([6., 6.], dtype=oneflow.float32)

    """,
)

reset_docstr(
    oneflow.zeros,
    r"""zeros(*size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    返回一个用标量值 0 填充的 tensor ，其形状由变量参数 :attr:`size` 定义。

    参数：
        - **size** (整数或整数元组): 定义输出张量的形状。可以是可变数量的参数或是像列表或元组这样的集合。
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 设置返回张量的 placement 属性。如果为None，则构造 local tensor 
        - **sbp** (flow.sbp.sbp 或 tuple of flow.sbp.sbp, 可选): 返回的consistent tensor的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False
  
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.zeros(5)
        >>> y
        tensor([0., 0., 0., 0., 0.], dtype=oneflow.float32)
        >>> y = flow.zeros(2,3)
        >>> y
        tensor([[0., 0., 0.],
                [0., 0., 0.]], dtype=oneflow.float32)

    """
)
