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

    返回一个元素全部为值为 0 的标量，形状和 `x` 相同的 Tensor。
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
        - **size** (int... 或 oneflow.Size):  定义输出张量的形状。可以是可变数量的参数或集合，如列表或元组或 oneflow.Size。
        - **dtype** (flow.dtype, 可选的): 返回张量的数据类型。默认：flow.float32
        - **device** (torch.device, 可选的): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选的): 返回一致张量的所需设备。如果为None，则构造局部张量。
        - **sbp** (flow.sbp 或 List[flow.sbp], 可选的): 返回的一致张量的所需 sbp。
        - **requires_grad** (bool, 可选的): 用 autograd 记录对返回张量的操作，默认为 False。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.empty(4, 5)  # 构造局部空张量
        >>> y.shape
        oneflow.Size([4, 5])
        >>> y.is_consistent
        False
        >>> placement = flow.placement("cpu", {0: [0]})
        >>> y = flow.empty(4, 5, placement=placement, sbp=flow.sbp.broadcast)  # 构造一致空张量
        >>> y.is_consistent
        True

    """)

reset_docstr(
    oneflow.ones,
    r"""ones(*size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False,) -> Tensor
    
    返回一个元素全部为标量 1 ，形状由参数 :attr:`size` 决定的 tensor 。

    参数：
        - **size** (一个整数或包含整数的元组)): 决定输出张量的形状，可以是数字变量或集合例如列表或元组
        - **dtype** (flow.dtype, 可选): 返回张量的数据类型。
        - **device** (flow.device, 可选): 返回的本地张量的所需设备。默认使用当前设备
        - **placement** (flow.placement, 可选): 返回一致张量的所需设备。如果为None，则构造局部张量。
        - **sbp** (flow.sbp.sbp 或包含 flow.sbp.sbp 的元组, 可选): 返回的一致张量的所需 sbp 描述符。如果为 None ，则返回的张量是使用参数 `device` 的本地张量。
        - **requires_grad** (bool, 可选): 用 autograd 记录对返回张量的操作，默认为 False。

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> y = flow.ones(5)
        >>> y
        tensor([1., 1., 1., 1., 1.], dtype=oneflow.float32)
        >>> y = flow.ones(2,3)
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)

    """
)
