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
        - **sbp** (flow.sbp 或 List[flow.sbp], 可选的): 返回的global tensor的所需 sbp
        - **requires_grad** (bool, 可选的): 用 autograd 记录对返回张量的操作，默认为 False

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.empty(4, 5)  # 构造空 local tensor
        >>> y.shape
        oneflow.Size([4, 5])
        >>> y.is_global
        False
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> y = flow.empty(4, 5, placement=placement, sbp=flow.sbp.broadcast)  # 构造空 global tensor
        >>> y.is_global
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
        - **sbp** (flow.sbp.sbp 或包含 flow.sbp.sbp 的元组, 可选): 返回的global tensor的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量
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
        >>> y = flow.ones(4, 5, placement=placement, sbp=flow.sbp.broadcast) # 构造 global tensor
        >>> y.is_global
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
        - **sbp** (flow.sbp.sbp 或包含 flow.sbp.sbp 的元组, 可选): 返回的global tensor的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量。
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
        - **sbp** (flow.sbp.sbp 或 tuple of flow.sbp.sbp, 可选): 返回的global tensor的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量
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

reset_docstr(
    oneflow.linalg.matrix_norm,
    r"""linalg.matrix_norm(input, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None) -> Tensor
    计算一个矩阵的范数。

    支持 float, double, cfloat 和 cdouble 的输入属性。

    该函数同时支持矩阵的 batch 。将通过由二元组的 :attr:`dim` 参数指定的维度计算范数，而其他的维度将被当作 batch 维度。输出矩阵将有相同的 batch 维度。

    :attr:`ord` 定义了被计算出的矩阵范数。支持下列类型的范数：
    
    ======================   ========================================================
    :attr:`ord`              矩阵范数
    ======================   ========================================================
    `'fro'` (默认)           Frobenius 范数
    `'nuc'`                  -- 暂未支持 --
    `inf`                    `max(sum(abs(x), dim=1))`
    `-inf`                   `min(sum(abs(x), dim=1))`
    `1`                      `max(sum(abs(x), dim=0))`
    `-1`                     `min(sum(abs(x), dim=0))`
    `2`                      -- 暂未支持 --
    `-2`                     -- 暂未支持 --
    ======================   ========================================================

    此处 `inf` 指代 `float('inf')`, NumPy 的 `inf` 对象， 或者任意等价的对象。

    参数：
        - **input** (Tensor): 拥有两个或更多维度的张量。默认情况下，其形状被解释为 `(*, m, n)` ，其中 `*` 是零个或更多 batch 维度，但这个解释方法可以通过 :attr:`dim` 控制。
        - **ord** (int, inf, -inf, 'fro', 'nuc', 可选): 范数的类型。默认： `'fro'`
        - **dim** (Tuple[int, int], 可选): 用于计算范数的维度。默认： `(-2, -1)`
        - **keepdim** (bool, 可选): 如果设置为 `True` ，被减少的维度将以大小为一的维度保留。默认： `False`

    返回值：
        一个真实值的张量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> from oneflow import linalg as LA
        >>> import numpy as np
        >>> a = flow.tensor(np.arange(9, dtype=np.float32)).reshape(3,3)
        >>> a
        tensor([[0., 1., 2.],
                [3., 4., 5.],
                [6., 7., 8.]], dtype=oneflow.float32)
        >>> LA.matrix_norm(a)
        tensor(14.2829, dtype=oneflow.float32)
        >>> LA.matrix_norm(a, ord=-1)
        tensor(9., dtype=oneflow.float32)
        >>> b = a.expand(2, -1, -1)
        >>> b
        tensor([[[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]],
        <BLANKLINE>
                [[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]]], dtype=oneflow.float32)
        >>> LA.matrix_norm(b, dim=(0, 2))
        tensor([ 3.1623, 10.0000, 17.2627], dtype=oneflow.float32)
    
    """,
)

reset_docstr(
    oneflow.linalg.vector_norm,
    r"""
    计算一个矢量范数。

    支持 float 和 double 的输入类型。

    这个函数不总是将多维张量 :attr:`input` 作为矢量的 batch ，而是：

    - 如果 :attr:`dim` \ `= None`, :attr:`input` 将在计算范数前被扁平化。
    - 如果 :attr:`dim` 是一个 `int` 或者 `tuple`, 范数将在这些维度上计算，而其他维度将被当作 batch 维度。

    此行为是为了与 :func:`flow.linalg.norm` 保持一致。

    :attr:`ord` 定义了被计算出的矩阵范数。支持下列类型的范数：

    ======================   ========================================================
    :attr:`ord`              矢量范数
    ======================   ========================================================
    `2` (默认)               `2`-norm (见下)
    `inf`                    `max(abs(x))`
    `-inf`                   `min(abs(x))`
    `0`                      `sum(x != 0)`
    其他 `int` 或 `float`     `sum(abs(x)^{ord})^{(1 / ord)}`
    ======================   ========================================================

    此处 `inf` 指代 `float('inf')`, NumPy 的 `inf` 对象， 或者任意等价的对象。

    参数：
        - **input** (Tensor): 输入张量，默认情况下会被扁平化，但是此行为可以用 :attr:`dim` 控制。
        - **ord** (int, float, inf, -inf, 'fro', 'nuc', 可选): 范数的类型。默认： `2`
        - **dim** (int, Tuple[int], 可选): 用于计算范数的维度。关于 :attr:`dim`\ `= None` 时的行为见之前段落。默认： `None`
        - **keepdim** (bool, 可选): 如果设置为 `True` ，被减少的维度将以大小为一的维度保留。默认： `False`

    返回值：
        一个真实值的张量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> from oneflow import linalg as LA
        >>> import numpy as np
        >>> a = flow.tensor(np.arange(9, dtype=np.float32) - 4)
        >>> a
        tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.], dtype=oneflow.float32)
        >>> b = a.reshape(3, 3)
        >>> b
        tensor([[-4., -3., -2.],
                [-1.,  0.,  1.],
                [ 2.,  3.,  4.]], dtype=oneflow.float32)
        >>> LA.vector_norm(a, ord=3.5)
        tensor(5.4345, dtype=oneflow.float32)
        >>> LA.vector_norm(b, ord=3.5)
        tensor(5.4345, dtype=oneflow.float32)

    """
)

reset_docstr(
    oneflow.set_num_threads,
    r"""
    设置在 cpu 上用于 intraop 并行计算的线程数量。
    
    .. WARNING::
        为了保证正确数量的线程被使用， set_num_threads 必须在运行 eager ， eager globe 或 ddp 之前被调用。

    """
)

reset_docstr(
    oneflow.vsplit,
    r"""
    根据 indices_or_sections 的值，将输入的二维或更高维张量垂直地切分成多个张量。
    每个切分出的张量都是输入张量的一个 view 。此算子等价于调用 torch.tensor_split(input, indices_or_sections, dim=0) （切分维度为 0 ），除了在 indices_or_sections 是
    一个整型的情况下，此时它必须平均划分切分维度，否则将抛出一个运行时错误。此文档参考自：https://pytorch.org/docs/stable/generated/torch.vsplit.html#torch.vsplit
    

    参数：
        - **input** (Tensor) - 输入张量
        - **indices_or_sections** (int 或者一个列表) - 如果 indices_or_sections 是一个整型 n , input 将沿着维度 dim 被切分为 n 个部分。
            如果在维度 dim 上 input 能被 n 整除，则每个部分的大小相同，都为 ``input.size(dim)/n``。如果 input 无法被 n 整除，则前 int(input.size(dim) % n) 个部分（section）的大小都为 int(input.size(dim) / n) + 1，
            而余下的部分大小则是 int(input.size(dim) / n)。如果 indices_or_sections 是一个整型的列表或元组，则输入将从维度 dim 根据列表或元组内的每个索引被分裂。
            比如， indices_or_sections=[2, 3] 和 dim=0 将使输入张量被分裂为 input[:2], input[2:3], 和 input[3:] 。如果 indices_or_sections 是一个张量，它必须是一个位于 cpu 的
            零维或者一维张量。
            

    返回值：
        输出的 TensorTuple
    
    返回类型：
        oneflow.TensorTuple

    示例：

    .. code-block:: python

        >>> import oneflow as flow

        >>> input = flow.rand(3,4,5,6)
        >>> output = flow.vsplit(input,(1,3))
        >>> output[0].size()
        oneflow.Size([1, 4, 5, 6])
        >>> output[1].size()
        oneflow.Size([2, 4, 5, 6])
        >>> output[2].size()
        oneflow.Size([1, 4, 5, 6])
    """,
)

reset_docstr(
    oneflow.t,
    r"""
    oneflow.t(input) → Tensor.

        要求 `input` 小于二维，并交换第零维和第一维。

        零维和一维张量将返回其本身。对于二维张量，此算子等价于 `transpose(input, 0, 1)` 。

    参数：
        - **input** (oneflow.Tensor) - 输入张量  
 
    For example:

    .. code-block:: python

        >>> import oneflow as flow

        >>> x = flow.rand()
        >>> flow.t(x).shape
        oneflow.Size([])
        >>> x = flow.rand(3)
        >>> flow.t(x).shape
        oneflow.Size([3])
        >>> x = flow.rand(2,3)
        >>> flow.t(x).shape
        oneflow.Size([3, 2])
    
    """,
)

reset_docstr(
    oneflow.swapaxes,
    r"""此函数等价于 NumPy 的 swapaxes 函数。

    示例：

    .. code-block:: python
    
        >>> import oneflow as flow
               
        >>> x = flow.tensor([[[0,1],[2,3]],[[4,5],[6,7]]])
        >>> x.shape
        oneflow.Size([2, 2, 2])
        >>> flow.swapaxes(x, 0, 1).shape
        oneflow.Size([2, 2, 2])
        >>> flow.swapaxes(x, 0, 2).shape
        oneflow.Size([2, 2, 2])

    """,
)

reset_docstr(
    oneflow.select,
    r"""
    将原张量根据给定的索引在指定维度上切分。此算子返回移除了给定维度后原张量的 view 。

    参数：
        - **input** (Tensor) - 输入张量
        - **dim**  (int) - 切分的维度
        - **select** (int) - 选择的索引

    返回：
        oneflow.Tensor: 输出张量。

    示例：
    
    .. code-block:: python
    
        >>> import oneflow as flow
        >>> input = flow.rand(3, 4, 5)
        >>> out = flow.select(input, 0, 1)
        >>> out.size()
        oneflow.Size([4, 5])
        >>> out = flow.select(input, 1, 1)
        >>> out.size()
        oneflow.Size([3, 5])
    """,
)

reset_docstr(
    oneflow.roll,
    r"""将张量根据指定维度滚动。
    
    在末端位置的元素将被移动至首端位置。
    
    如果维度没有被指定，张量将在滚动前被扁平化，然后被恢复至原形状。

    参数：
        - **input** (oneflow.Tensor) - 输入张量
        - **shifts** (int or int 元组 ) - 张量偏移的位置数量。如果 shifts 是一个元组，则 dims 必须也是一个相同大小的元组，且每个维度将会根据对应的值滚动。
        - **dims** (int or tuple of python:ints) - 滚动的轴向维度

    返回：
        oneflow.Tensor: 结果张量。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([[1, 2],
        ...               [3, 4],
        ...               [5, 6],
        ...               [7, 8]], dtype=flow.float32)
        >>> input = flow.Tensor(x)
        >>> input.shape
        oneflow.Size([4, 2])
        >>> out = flow.roll(input, 1, 0)
        >>> out
        tensor([[7., 8.],
                [1., 2.],
                [3., 4.],
                [5., 6.]], dtype=oneflow.float32)
        >>> input.roll(-1, 1)
        tensor([[2., 1.],
                [4., 3.],
                [6., 5.],
                [8., 7.]], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.permute,
    r"""
    permute(input, *dims) -> Tensor

    返回原张量的维度换位后的 view 。

    参数：
        - **dims** (int 的元组 ) - 需要的维度顺序

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.rand(2, 6, 5, 3)
        >>> output = flow.permute(input, (1, 0, 2, 3)).shape
        >>> output
        oneflow.Size([6, 2, 5, 3])

    """,
)

reset_docstr(
    oneflow.ne,
    r"""ne(input, other) -> Tensor

    计算 element-wise 的不等性。
    第二个参数是一个可以与第一个参数广播的数字或张量。

    参数：
        - **input** (oneflow.Tensor): 用于比较的张量。
        - **other** (oneflow.Tensor, float 或者 int): 与输入比较的张量。

    返回：
        一个布尔值构成的张量，当 :attr:`input` 不等于 :attr:`other` 时为真，否则为假。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> input = flow.tensor([2, 3, 4, 5], dtype=flow.float32)
        >>> other = flow.tensor([2, 3, 4, 1], dtype=flow.float32)

        >>> y = flow.ne(input, other)
        >>> y
        tensor([False, False, False,  True], dtype=oneflow.bool)

    """,

)