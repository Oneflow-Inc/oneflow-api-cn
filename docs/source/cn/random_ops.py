import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.rand,
    r"""rand(*size, out=None, generator=None, dtype=None, layout=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    返回一个由在区间 [0, 1) 上均匀分布的随机数填充的新 tensor 。

    输出 tensor 的形状由变量 :attr:`size` 决定。

    参数：
        - **size** (int... 或 oneflow.Size): 定义输出张量的形状。可以是数字变量，或者集合例如列表或元组，或者 oneflow.Size 
        - **out** (可选): 输出张量
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型。默认： ``flow.float32`` 
        - **layout** (可选): 返回的 Tensor 的 layout。
        - **generator** (flow.Generator, 可选): 用于采样的伪随机数生成器
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 返回consistent tensor的所需设备。如果为None，则构造 local tensor
        - **sbp** (flow.sbp, 可选): 返回的consistent tensor的所需 sbp 描述符。必须和 placement 的数量相等
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.rand(3,3) # 构造 local tensor
        >>> x.shape
        oneflow.Size([3, 3])
        >>> x.is_consistent
        False
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.rand(3, 3, placement=placement, sbp=sbp) # 构造 consistent tensor
        >>> x.is_consistent
        True

    """

)

reset_docstr(
    oneflow.randn,
    r"""randn(*size, out=None, generator=None, dtype=None, layout=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    返回一个由符合期望为 0 ，方差为 1 的正态分布的随机数填充的新 tensor 。 
    
    输出 tensor 的形状由变量 :attr:`size` 决定。
    

    参数：
        - **size** (int... 或 oneflow.Size): 定义输出张量的形状。可以是数字变量，或者集合例如列表或元组，或者 oneflow.Size 

    关键词参数：
        - **out** (可选): 输出张量
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型。默认： ``flow.float32`` 
        - **layout** (可选): 返回的 Tensor 的 layout。
        - **generator** (flow.Generator, 可选): 用于采样的伪随机数生成器
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 返回consistent tensor的所需设备。如果为None，则构造 local tensor
        - **sbp** (flow.sbp, 可选): 返回的consistent tensor的所需 sbp 描述符。必须和 placement 的数量相等
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False

    返回类型：
        oneflow.Tensor
    
    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.randn(3,3) # 构造 local tensor
        >>> x.shape
        oneflow.Size([3, 3])
        >>> x.is_consistent
        False
        >>> placement = flow.placement("cpu", {0:[0]})
        >>> sbp = flow.sbp.broadcast
        >>> x = flow.randn(3,3,placement=placement,sbp=sbp) # 构造 consistent tensor
        >>> x.is_consistent
        True

    """
)

reset_docstr(
    oneflow.randint,
    r"""randint(low, high, size, out=None, generator=None, dtype=None, layout=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    返回一个张量，其中填充了在 :attr:`low` （包括）和 :attr:`high` （不包括）之间均匀生成的随机整数。

    输出 tensor 的形状由变量 :attr:`size` 决定。

    参数：
        - **low** (flow.int64): 返回张量中的最小值（包括）
        - **high** (flow.int64): 返回张量中的最大值（不包括）
        - **size** (int... 或 oneflow.Size): 定义输出张量的形状。可以是数字变量，或者集合例如列表或元组，或者 oneflow.Size 

    关键词参数：
        - **out** (可选): 输出张量
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型。默认： ``flow.float32`` 
        - **layout** (可选): 返回的 Tensor 的 layout。
        - **generator** (flow.Generator, 可选): 用于采样的伪随机数生成器
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 返回consistent tensor的所需设备。如果为None，则构造 local tensor
        - **sbp** (flow.sbp, 可选): 返回的consistent tensor的所需 sbp 描述符。必须和 placement 的数量相等
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False

    返回类型：
        oneflow.Tensor

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> y = flow.randint(0, 5, (3,3), generator=generator)
        >>> y
        tensor([[2, 2, 3],
                [4, 3, 4],
                [2, 4, 2]], dtype=oneflow.int64)
        >>> y.is_consistent
        False
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> y = flow.randint(0, 5, (3,3), generator=generator, placement=placement, sbp=flow.sbp.broadcast) # 构造 consistent tensor
        >>> y.is_consistent
        True

    """
)

reset_docstr(
    oneflow.randperm,
    r"""randperm (n, generator=None, out=None, dtype=None, layout=None, device=None, placement=None, sbp=None, requires_grad=False, pin_memory=False,) -> Tensor

    返回从 ``0`` 到 ``n - 1`` （不包括）的整数的随机排列。

    参数：
        - **n** (int): 最大值（不包括）
    
    返回类型：
        oneflow.Tensor

    关键词参数：
        - **generator** (flow.Generator, 可选): 用于采样的伪随机数生成器
        - **out** (可选): 输出张量
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型。默认： ``flow.float32`` 
        - **layout** (可选): 返回的 Tensor 的 layout。
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 返回consistent tensor的所需设备。如果为None，则构造 local tensor
        - **sbp** (flow.sbp, 可选): 返回的consistent tensor的所需 sbp 描述符。必须和 placement 的数量相等
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False
        - **pin_memory** (bool, 可选)：目前不支持 pin_memory

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> generator = flow.Generator()
        >>> generator.manual_seed(0)
        >>> y = flow.randperm(5, generator=generator)
        >>> y
        tensor([2, 4, 3, 0, 1], dtype=oneflow.int32)
        >>> y.is_consistent
        False
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> y = flow.randperm(5, generator=generator, placement=placement, sbp=flow.sbp.broadcast) # 构造 consistent tensor
        >>> y.is_consistent
        True
        
    """
)
