import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.tensor,
    r"""
    用数据构造一个张量，如果设置了 :attr:`placement` 和 :attr:`sbp` ，则返回consistent tensor，
        否则返回一个 local tensor 。
       
    参数：
        - **data**: 张量的初始数据。可以是列表、元组、NumPy ndarray、标量或张量。

    关键词参数：
        - **dtype** (oneflow.dtype, 可选)：返回张量的所需数据类型。默认值：如果没有，则从数据推断数据类型。
        - **device** (oneflow.device, 可选)：返回张量的所需设备。如果 placement 和 sbp 为 None，则使用当前 cpu 作为默认设备。
        - **placement** (oneflow.placement, 可选)：设置返回张量的 placement 属性。
        - **sbp** (oneflow.sbp 或 oneflow.sbp 中的元组, 可选)：返回张量的所需 sbp。
        - **requires_grad** (bool, 可选)：如果已经自动求导则记录对返回张量的操作。默认值：False。

    注意：
        关键词参数 device 与 placement 、 sbp 是互斥的。
        consistent tensor只能由张量构造。


    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1,2,3])
        >>> x
        tensor([1, 2, 3], dtype=oneflow.int64)

    """
)

reset_docstr(
    oneflow.Tensor.atan2,
    r"""
    参考 :func:`oneflow.atan2`
    """
)

reset_docstr(
    oneflow.Tensor.expand,
    r"""
    Tensor.expand() -> Tensor

    参考 :func:`oneflow.expand`
    """
)

reset_docstr(
    oneflow.Tensor.expand_as,
    r"""
    expand_as(other) -> Tensor

    将输入张量扩展到与 :attr:`other` 相同的大小。
    ``self.expand_as(other)`` 等价于 ``self.expand(other.size())`` 。

    更多有关 ``expand`` 的细节请参考 :meth:`~Tensor.expand`。

    参数：
        - **other** (:class:`oneflow.Tensor`): 返回张量与 :attr:`other` 大小相同。
    """
)

reset_docstr(
    oneflow.Tensor.flatten,
    r"""
    参考 :func:`oneflow.flatten`
    """
)

reset_docstr(
    oneflow.Tensor.floor,
    r"""
    参考 :func:`oneflow.floor`
    """
)

reset_docstr(
    oneflow.Tensor.flip,
    r"""
    参考 :func:`oneflow.flip`
    """
)

reset_docstr(
    oneflow.Tensor.in_top_k,
    r"""
    Tensor.in_top_k(targets, predictions, k) -> Tensor

    参考 :func:`oneflow.in_top_k`
    """
)

reset_docstr(
    oneflow.Tensor.index_select,
    r"""
    Tensor.index_select(dim, index) -> Tensor

    参考 :func:`oneflow.index_select`
    """
)

reset_docstr(
    oneflow.Tensor.numel,
    r"""
    参考 :func:`oneflow.numel`
    """
)

reset_docstr(
    oneflow.Tensor.new_ones,
    r"""
    Tensor.new_ones() -> Tensor

    参考 :func:`oneflow.new_ones`
    """
)

reset_docstr(
    oneflow.Tensor.to_global,
    r"""
    Tensor.to_global() -> Tensor

    参考 :func:`oneflow.to_global`
    """
)

reset_docstr(
    oneflow.Tensor.transpose,
    r"""
    参考 :func:`oneflow.transpose`
    """
)

reset_docstr(
    oneflow.Tensor.logical_not,
    r"""
    logical_not() -> Tensor
    参考 :func:`oneflow.logical_not`
    """
)

reset_docstr(
    oneflow.Tensor.std,
    r"""
    参考 :func:`oneflow.std`
    """
)

reset_docstr(
    oneflow.Tensor.var,
    r"""
    参考 :func:`oneflow.var`
    """
)

reset_docstr(
    oneflow.Tensor.squeeze,
    r"""
    参考 :func:`oneflow.squeeze`
    """
)

reset_docstr(
    oneflow.Tensor.matmul,
    r"""
    参考 :func:`oneflow.matmul`
    """
)

reset_docstr(
    oneflow.Tensor.narrow,
    r"""
    参考 :func:`oneflow.narrow`
    """
)

reset_docstr(
    oneflow.Tensor.unsqueeze,
    r"""
    参考 :func:`oneflow.unsqueeze`
    """
)

reset_docstr(
    oneflow.Tensor.permute,
    r"""
    参考 :func:`oneflow.permute`
    """
)

reset_docstr(
    oneflow.Tensor.abs,
    r"""
    参考 :func:`oneflow.abs`
    """
)

reset_docstr(
    oneflow.Tensor.acos,
    r"""
    参考 :func:`oneflow.acos`
    """
)

reset_docstr(
    oneflow.Tensor.arccos,
    r"""
    参考 :func:`oneflow.arccos`
    """
)

reset_docstr(
    oneflow.Tensor.acosh,
    r"""
    参考 :func:`oneflow.acosh`
    """
)

reset_docstr(
    oneflow.Tensor.arccosh,
    r"""
    参考 :func:`oneflow.arccosh`
    """
)

reset_docstr(
    oneflow.Tensor.arctanh,
    r"""
    参考 :func:`oneflow.arctanh`
    """
)

reset_docstr(
    oneflow.Tensor.argmax,
    r"""
    参考 :func:`oneflow.argmax`
    """
)

reset_docstr(
    oneflow.Tensor.argmin,
    r"""
    参考 :func:`oneflow.argmin`
    """
)

reset_docstr(
    oneflow.Tensor.argwhere,
    r"""
    参考 :func:`oneflow.argwhere`
    """
)

reset_docstr(
    oneflow.Tensor.atanh,
    r"""
    参考 :func:`oneflow.atanh`
    """
)

reset_docstr(
    oneflow.Tensor.bmm,
    r"""
    参考 :func:`oneflow.bmm`
    """
)

reset_docstr(
    oneflow.Tensor.chunk,
    r"""
    参考 :func:`oneflow.chunk`
    """
)

reset_docstr(
    oneflow.Tensor.split,
    r"""
    参考 :func:`oneflow.split`
    """
)

reset_docstr(
    oneflow.Tensor.swapaxes,
    r"""
    参考 :func:`oneflow.swapaxes`
    """
)

reset_docstr(
    oneflow.Tensor.cast,
    r"""
    参考 :func:`oneflow.cast`
    """
)

reset_docstr(
    oneflow.Tensor.diag,
    r"""
    参考 :func:`oneflow.diag`
    """
)

reset_docstr(
    oneflow.Tensor.exp,
    r"""
    参考 :func:`oneflow.exp`
    """
)

reset_docstr(
    oneflow.Tensor.erf,
    r"""
    Tensor.erf() -> Tensor

    参考 :func:`oneflow.erf`
    """
)

reset_docstr(
    oneflow.Tensor.erfc,
    r"""
    Tensor.erfc() -> Tensor

    参考 :func:`oneflow.erfc`
    """
)

reset_docstr(
    oneflow.Tensor.erfinv,
    r"""
    参考 :func:`oneflow.erfinv`
    """
)

reset_docstr(
    oneflow.Tensor.eq,
    r"""
    参考 :func:`oneflow.eq`
    """
)

reset_docstr(
    oneflow.Tensor.lt,
    r"""
    参考 :func:`oneflow.lt`
    """
)

reset_docstr(
    oneflow.Tensor.le,
    r"""
    参考 :func:`oneflow.le`
    """
)

reset_docstr(
    oneflow.Tensor.ne,
    r"""
    参考 :func:`oneflow.ne`
    """
)

reset_docstr(
    oneflow.Tensor.fill_,
    r"""
    Tensor.fill_(value) → Tensor

    用指定的值填充 self 张量。
    """
)

reset_docstr(
    oneflow.Tensor.ge,
    r"""
    参考 :func:`oneflow.ge`
    """
)

reset_docstr(
    oneflow.Tensor.gelu,
    r"""
    参考 :func:`oneflow.gelu`
    """
)

reset_docstr(
    oneflow.Tensor.gt,
    r"""
    参考 :func:`oneflow.gt`
    """
)

reset_docstr(
    oneflow.Tensor.log1p,
    r"""
    参考 :func:`oneflow.log1p`
    """
)

reset_docstr(
    oneflow.Tensor.mish,
    r"""
    参考 :func:`oneflow.mish`
    """
)

reset_docstr(
    oneflow.Tensor.mul,
    r"""Tensor.mul(value) -> Tensor
    参考 :func:`oneflow.mul`
    """
)

reset_docstr(
    oneflow.Tensor.negative,
    r"""
    参考 :func:`oneflow.negative`
    """
)

reset_docstr(
    oneflow.Tensor.pow,
    r"""
    参考 :func:`oneflow.pow`
    """
)

reset_docstr(
    oneflow.Tensor.relu,
    r"""
    参考 :func:`oneflow.relu`
    """
)

reset_docstr(
    oneflow.Tensor.roll,
    r"""
    参考 :func:`oneflow.roll`
    """
)

reset_docstr(
    oneflow.Tensor.round,
    r"""
    参考 :func:`oneflow.round`
    """
)

reset_docstr(
    oneflow.Tensor.reciprocal,
    r"""
    参考 :func:`oneflow.reciprocal`
    """
)

reset_docstr(
    oneflow.Tensor.asin,
    r"""
    参考 :func:`oneflow.asin`
    """
)

reset_docstr(
    oneflow.Tensor.arcsin,
    r"""
    参考 :func:`oneflow.arcsin`
    """
)

reset_docstr(
    oneflow.Tensor.arcsinh,
    r"""
    参考 :func:`oneflow.arcsinh`
    """
)

reset_docstr(
    oneflow.Tensor.sin,
    r"""
    sin() -> Tensor

    参考 :func:`oneflow.sin`
    """
)

reset_docstr(
    oneflow.Tensor.cos,
    r"""
    参考 :func:`oneflow.cos`
    """
)

reset_docstr(
    oneflow.Tensor.atan,
    r"""
    参考 :func:`oneflow.atan`
    """
)

reset_docstr(
    oneflow.Tensor.arctan,
    r"""
    参考 :func:`oneflow.arctan`
    """
)

reset_docstr(
    oneflow.Tensor.selu,
    r"""
    参考 :func:`oneflow.selu`
    """
)

reset_docstr(
    oneflow.Tensor.sigmoid,
    r"""
    参考 :func:`oneflow.sigmoid`
    """
)

reset_docstr(
    oneflow.Tensor.sign,
    r"""
    参考 :func:`oneflow.sign`
    """
)

reset_docstr(
    oneflow.Tensor.silu,
    r"""
    参考 :func:`oneflow.silu`
    """
)

reset_docstr(
    oneflow.Tensor.sinh,
    r"""
    参考 :func:`oneflow.sinh`
    """
)

reset_docstr(
    oneflow.Tensor.softmax,
    r"""
    参考 :func:`oneflow.softmax`
    """
)

reset_docstr(
    oneflow.Tensor.softplus,
    r"""
    参考 :func:`oneflow.softplus`
    """
)

reset_docstr(
    oneflow.Tensor.softsign,
    r"""
    参考 :func:`oneflow.softsign`
    """
)

reset_docstr(
    oneflow.Tensor.tan,
    r"""
    参考 :func:`oneflow.tan`
    """
)

reset_docstr(
    oneflow.Tensor.tanh,
    r"""
    参考 :func:`oneflow.tanh`
    """
)

reset_docstr(
    oneflow.Tensor.tril,
    r"""
    参考 :func:`oneflow.tril`
    """
)

reset_docstr(
    oneflow.Tensor.triu,
    r"""
    参考 :func:`oneflow.triu`
    """
)

reset_docstr(
    oneflow.Tensor.gather,
    r"""
    oneflow.Tensor.gather(dim, index) -> Tensor

    参考 :func:`oneflow.gather`

    """
)

reset_docstr(
    oneflow.Tensor.clamp,
    r"""
    参考 :func:`oneflow.clamp`.
    """
)

reset_docstr(
    oneflow.Tensor.repeat,
    r"""
    Tensor.repeat(*size) -> Tensor

    参考 :func:`oneflow.repeat`
    """
)

reset_docstr(
    oneflow.Tensor.t,
    r"""
    Tensor.t() → Tensor

    参考 :func:`oneflow.t`
    """
)

reset_docstr(
    oneflow.Tensor.tile,
    r"""
    Tensor.tile(*dims) -> Tensor

    参考 :func:`oneflow.tile`
    """
)

reset_docstr(
    oneflow.Tensor.fmod,
    r"""
    Tensor.fmod(other) -> Tensor

    参考 :func:`oneflow.fmod`

    """
)

reset_docstr(
    oneflow.Tensor.logical_and,
    r"""
    logical_and() -> Tensor

    参考 :func:`oneflow.logical_and`

    """
)

reset_docstr(
    oneflow.Tensor.logical_or,
    r"""

    logical_or() -> Tensor

    参考 :func:`oneflow.logical_or`

    """
)

reset_docstr(
    oneflow.Tensor.logical_xor,
    r"""
    logical_xor() -> Tensor

    参考 :func:`oneflow.logical_xor`

    """
)

reset_docstr(
    oneflow.Tensor.masked_fill,
    r"""
    参考 :func:`oneflow.masked_fill`
    """
)

reset_docstr(
    oneflow.Tensor.masked_select,
    r"""
    参考 :func:`oneflow.masked_select`
    """
)

reset_docstr(
    oneflow.Tensor.sub,
    r"""
    参考 :func:`oneflow.sub`
    """
)

reset_docstr(
    oneflow.Tensor.div,
    r"""
    参考 :func:`oneflow.div`

    """
)

reset_docstr(
    oneflow.Tensor.ceil,
    r"""
    参考 :func:`oneflow.ceil`
    """
)

reset_docstr(
    oneflow.Tensor.expm1,
    r"""
    参考 :func:`oneflow.expm1`
    """
)

reset_docstr(
    oneflow.Tensor.topk,
    r"""
    参考 :func:`oneflow.topk`
    """
)

reset_docstr(
    oneflow.Tensor.nms,
    r"""
    参考 :func:`oneflow.nms`
    """
)

reset_docstr(
    oneflow.Tensor.nonzero,
    r"""
    nonzero(input, as_tuple=False) -> Tensor

    参考 :func:`oneflow.nonzero`
    """
)

reset_docstr(
    oneflow.Tensor.max,
    r"""
    input.max(dim, index) -> Tensor

    参考 :func:`oneflow.max`
    """
)

reset_docstr(
    oneflow.Tensor.min,
    r"""
    input.min(dim, index) -> Tensor

    参考 :func:`oneflow.min`
    """
)

reset_docstr(
    oneflow.Tensor.sum,
    r"""
    input.sum(dim, index) -> Tensor

    参考 :func:`oneflow.sum`
    """
)

reset_docstr(
    oneflow.Tensor.mean,
    r"""
    input.mean(dim, index) -> Tensor

    参考 :func:`oneflow.mean`
    """
)

reset_docstr(
    oneflow.Tensor.prod,
    r"""
    input.prod(dim, index) -> Tensor

    参考 :func:`oneflow.prod`
    """
)

reset_docstr(
    oneflow.Tensor.reshape,
    r"""
    参考 :func:`oneflow.reshape`
    """
)

reset_docstr(
    oneflow.Tensor.sort,
    r"""
    参考 :func:`oneflow.sort`
    """
)

reset_docstr(
    oneflow.Tensor.is_floating_point,
    r"""
    参考 :func:`oneflow.is_floating_point`
    """
)

reset_docstr(
    oneflow.Tensor.where,
    r"""
    参考 :func:`oneflow.where`
    """
)

reset_docstr(
    oneflow.Tensor.unfold,
    r"""
    该接口与PyTorch一致。
    该文件的参考来源是: https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html#torch.Tensor.unfold.

    返回一个原始张量的视图，其中包含在维度 `dimension` 中来自 `self` 张量大小 `size` 的所有片段。

    两片段之间的步长由 `step` 给出。

    如果 sizedim 是 `self` 维度 `dimension` 的大小，那么返回的张量中的维度大小将是（sizedim - size / step + 1）。返回的张量将是（sizedim - size）/ step + 1。

    在返回的张量中附加一个大小为 `size` 的额外维度。

    参数:
        - **dimension** (int) - 展开的维度
        - **size** (int) - 展开的每一片段的大小
        - **step** (int) - 每片段之间的步骤

    例如:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.arange(1., 8)
        >>> x
        tensor([1, 2, 3, 4, 5, 6, 7], dtype=oneflow.int64)
        >>> x.unfold(0, 2, 1)
        tensor([[1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7]], dtype=oneflow.int64)
        >>> x.unfold(0, 2, 2)
        tensor([[1, 2],
                [3, 4],
                [5, 6]], dtype=oneflow.int64)
    """
)

reset_docstr(
    oneflow.Tensor.argsort,
    r"""这个算子将输入的张量按指定的 dim 进行排序，并返回排序后的张量索引。

    参数:
        - **input** (oneflow.Tensor) - 输入张量。
        - **dim** (int, optional) - 要排序的维度。默认为最后一个维度（-1）。
        - **descending (bool, optional)** - 控制排序的顺序（升序或降序）。

    返回:
        - **oneflow.Tensor** - 排序后的张量索引。

    例如:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> x = np.array([[10, 2, 9, 3, 7],
        ...               [1, 9, 4, 3, 2]]).astype("float32")
        >>> input = flow.Tensor(x)
        >>> output = flow.argsort(input)
        >>> output
        tensor([[1, 3, 4, 2, 0],
                [0, 4, 3, 2, 1]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, descending=True)
        >>> output
        tensor([[0, 2, 4, 3, 1],
                [1, 2, 3, 4, 0]], dtype=oneflow.int32)
        >>> output = flow.argsort(input, dim=0)
        >>> output
        tensor([[1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0]], dtype=oneflow.int32)

    """
)

reset_docstr(
    oneflow.Tensor.backward,
    r"""
    该接口与PyTorch一致。
    该文件的参考来源是: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward.

    计算当前张量即 graph leaves 的梯度。

    Graph 是用链式规则来区分的。如果张量是非标量（即它的数据有一个以上的元素）并且需要梯度，则该函数还需要指定梯度。它应该是一个类型和位置相匹配的张量，它包含了被微分函数（即 self）的梯度。

    这个函数在 leaves 中累积梯度 - 你可能需要在调用它之前将 .grad 的属性归零或将它们设置为 None。关于累积梯度的内存结构的细节，参阅默认梯度结构。

    注意:
        如果你在用户指定的 CUDA 流上下文中运行任何前向操作、创建梯度或后向调用，参阅后向传递的流语义。
    注意:
        当提供了输入，并且给定的输入不是 leaf 时，当前的实现将调用它的 grad_fn（尽管严格来说不需要获得这个梯度）。这是一个实现的细节，用户不应该依赖它。更多细节见 https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 。

    参数:
        - **gradient** (Tensor or None) - 张量的梯度。如果它是一个张量，除非 create_graph 为 True ，否则它将被自动转换为一个不需要 grad 的张量。对于标量张量或不需要 grad 的张量，可以指定为 None 值。如果一个 None 值是可以接受的，那么这个参数是可选的。

        - **retain_graph** (bool, optional) - 如果为 False ，用于计算 grads 的 graph 将被释放。请注意，在几乎所有的情况下，不需要将这个选项设置为 True ，通常可以用一种更有效的方式来解决。默认为 create_graph 的值。

        - **create_graph** (bool, optional) - 如果为 True, 将构建导数的 graph ，允许计算高阶导数。默认为False。
    """
)

reset_docstr(
    oneflow.Tensor.dim,
    r"""
    Tensor.dim() → int

    返回 self 张量的维数。
    """
)

reset_docstr(
    oneflow.Tensor.element_size,
    r"""
    Tensor.element_size() → int

    返回单个元素的字节大小。

    """
)

reset_docstr(
    oneflow.Tensor.fill_,
    r"""
    Tensor.fill_(value) → Tensor

    用指定的值填充 self 张量。
    """
)

reset_docstr(
    oneflow.Tensor.get_device,
    r"""
    Tensor.get_device() -> Device ordinal (Integer)

    对于 CUDA 张量，该函数返回张量所在 GPU 的设备序号。对于 CPU 张量，会产生一个错误。


    """
)

reset_docstr(
    oneflow.Tensor.mul_,
    r"""Tensor.mul_(value) -> Tensor

    :func:`oneflow.Tensor.mul` 的 Inplace 版本。
    """
)

reset_docstr(
    oneflow.Tensor.div_,
    r"""Tensor.div_(value) -> Tensor
    :func:`oneflow.Tensor.div` 的 Inplace 版本。
    """
)

reset_docstr(
    oneflow.Tensor.sub_,
    r"""Tensor.sub_(value) -> Tensor
    :func:`oneflow.Tensor.sub` 的 Inplace 版本。
    """
)

reset_docstr(
    oneflow.Tensor.nelement,
    r"""
    Tensor.nelement() → int

    numel() 的别名。
    """
)

reset_docstr(
    oneflow.Tensor.floor_,
    r"""
    :func:`oneflow.floor` 的 Inplace 版本。

    """
)

reset_docstr(
    oneflow.Tensor.normal_,
    """
    normal_(mean=0, std=1, *, generator=None) -> Tensor

    用正态分布的元素样本填充 :attr:`self` 张量，参数为 :attr:`mean` 和 :attr:`std`。
    """
)

reset_docstr(
    oneflow.Tensor.numpy,
    r"""
    Tensor.numpy() → numpy.ndarray

    将 self 张量作为一个 NumPy ndarray 返回。这个张量和返回的 ndarray 共享相同的底层存储。对 self 张量的变化将反映在 ndarray 中，反之亦然。
    """
)

reset_docstr(
    oneflow.Tensor.add_,
    r"""
    :func:`oneflow.Tensor.add` 的 Inplace 版本。
    """
)

reset_docstr(
    oneflow.Tensor.size,
    r"""
    该接口与 PyTorch 一致。

    返回 self 张量的大小。如果没有指定 dim ，返回值是 oneflow.Size ，是 tuple 的子类。如果指定了 dim ，返回一个持有该维度大小的 int 。

    参数:
        - **idx** (int, optional) - 用于检索尺寸的维度。


    """,
)

reset_docstr(
    oneflow.Tensor.uniform_,
    r"""
    Tensor.uniform_(from=0, to=1) → Tensor

    用从连续均匀分布中采样的数字填充 self 张量。

    .. math::
        P(x)=1/(to-from)

    """
)

reset_docstr(
    oneflow.Tensor.copy_,
    r"""
    该接口与 PyTorch 一致。

    Tensor.copy_(src, non_blocking=False) → Tensor

    将 src 中的元素复制到 self 张量中，并返回 self 。

    src 张量必须可以与 self 张量一起广播。它可以是不同的数据类型，或者位于不同的设备上。

    参数:

        - **src** (Tensor) - 要复制的源张量

        - **non_blocking** (bool) - 如果为 True ，并且是在 CPU 和 GPU 之间的拷贝，那么相对于主机来说，拷贝可能会异步发生。对于其他情况，这个参数没有影响。
    """
)

reset_docstr(
    oneflow.Tensor.to,
    r"""执行 Tensor dtype 或设备转换。
        flow.dtype 和 flow.device 是由 `input.to(*args, **kwargs)` 的参数推断出来的。

    .. note::
        如果 ``input`` 张量已经
        有正确的 :class:`flow.dtype` 和 :class:`flow.device` ，那么将返回 ``input`` 。
        否则，返回的张量是所需的 ``input`` 副本。

    参数:
        - **input** (oneflow.Tensor) - 一个输入张量。
        - **args** (oneflow.Tensor or oneflow.device or oneflow.dtype) - 位置参数。
        - **kwargs** (oneflow.device or oneflow.dtype) - 键值参数。

    返回:
        oneflow.Tensor: 一个张量。

    例如:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> arr = np.random.randint(1, 9, size=(1, 2, 3, 4))
        >>> input = flow.Tensor(arr)
        >>> output = input.to(dtype=flow.float32)
        >>> np.array_equal(arr.astype(np.float32), output.numpy())
        True

    """
)

reset_docstr(
    oneflow.Tensor.clamp_,
    r"""
    :func:`oneflow.Tensor.clamp` 的 Inplace 版本。
    """
)

reset_docstr(
    oneflow.Tensor.clip,
    r"""
    :func:`oneflow.Tensor.clamp` 的别名。
    """
)

reset_docstr(
    oneflow.Tensor.clip_,
    r"""
    :func:`oneflow.Tensor.clamp_` 的别名。
    """
)

reset_docstr(
    oneflow.Tensor.T,
    r"""
    此张量所有维度转置后的张量。

    如果 `n` 是 `x` 的维数，`x.T` 等同于 `x.permute(n-1, n-2, ..., 0)` 。
    """
)

reset_docstr(
    oneflow.Tensor.int,
    r"""`Tensor.int()` 和 `Tensor.to(flow.int32)` 是等同的，参考 to() 函数。
    
    参数:
        - **input**  (Tensor) - 输入张量。

    例如:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> input = input.int()
        >>> input.dtype
        oneflow.int32
    """
)

reset_docstr(
    oneflow.Tensor.long,
    r"""`Tensor.long()` 和 `Tensor.to(flow.int64)` 是等同的， 参考 to() 函数。

    参数:
        - **input**  (Tensor) - 输入张量。

    例如:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.float32)
        >>> input = input.long()
        >>> input.dtype
        oneflow.int64
    """
)

reset_docstr(
    oneflow.Tensor.float,
    r"""`Tensor.float()` 和 `Tensor.to(flow.float32)` 是等同的， 查阅 to() 函数。

    参数:
        - **input**  (Tensor) - 输入张量。

    例如:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.int)
        >>> input = input.float()
        >>> input.dtype
        oneflow.float32
    """
)

reset_docstr(
    oneflow.Tensor.double,
    r"""`Tensor.double()` 和 `Tensor.to(flow.float64)` 是等同的，查阅 to() 函数。

    参数:
        - **input**  (Tensor) - 输入张量。

    例如:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np

        >>> input = flow.tensor(np.random.randn(1, 2, 3), dtype=flow.int)
        >>> input = input.double()
        >>> input.dtype
        oneflow.float64
    """
)

reset_docstr(
    oneflow.Tensor.erfinv_,
    r"""
    :func:`oneflow.erfinv` 的 Inplace 版本。
    """
)

reset_docstr(
    oneflow.Tensor.item,
    r"""将这个张量的值作为一个标准的 Python 数字返回，这只对有一个元素的张量有效。
    对于其他情况，参考 tolist() 函数。

    这个运算是不可微分的。

    参数:
        - **input**  (Tensor) - 输入张量。

    例如:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.tensor([1.0])
        >>> x.item()
        1.0
    """
)

reset_docstr(
    oneflow.device,
    r"""
    该文档参考自：
    https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device

    :class:`oneflow.device` 是一个对象，用于指代一个 :class:`oneflow.Tensor` 将被分配到的设备。

    :class:`oneflow.device` 包括一个设备类型（ 'cpu' 或 'cuda' ）和一个可选的设备类型下的设备序号。如果设备序号不存在，则此对象将总是指代当前设备类型下的当前设备。

    :class:`oneflow.device` 所指代的设备可以通过 Tensor.device 属性获取。

    :class:`oneflow.device` 可以通过一个 string 构建，也可以通过 string 和设备序号共同构建。

    通过 string 构建的示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.device('cuda:0')
        device(type='cuda', index=0)

        >>> flow.device('cpu')
        device(type='cpu', index=0)

        >>> flow.device('cuda')  # 当前 cuda 设备
        device(type='cuda', index=0)
    
    通过 string 和设备序号构建的示例：

    .. code-block:: python

        >>> import oneflow as flow
        >>> flow.device('cuda', 0)
        device(type='cuda', index=0)

        >>> flow.device('cpu', 0)
        device(type='cpu', index=0)
    
    Note:
        The :class:`oneflow.device` 参数在函数中通常可以被一个字符串替代。这保证了代码的快速原型化。
        
        .. code-block:: python

            >>> import oneflow as flow
            >>> # 一个使用了 oneflow.device 的示例函数
            >>> cuda0 = flow.device('cuda:0')
            >>> x = flow.randn(2,3, device=cuda0)
        
        .. code-block:: python

            >>> # 可以将 flow.device 用 string 替代
            >>> x = flow.randn(2,3, device='cuda:0')
    """
)

reset_docstr(
    oneflow.Tensor.addcmul,
    """
    参考 :func:`oneflow.addcmul`
    """
)

reset_docstr(
    oneflow.Tensor.addcmul_,
    """
    :func:`oneflow.Tensor.addcmul` 的 In-place 版本。
    """
)

reset_docstr(
    oneflow.Tensor.addmm,
    """
    参考 :func:`oneflow.addmm`
    """
)

reset_docstr(
    oneflow.Tensor.amax,
    """
    参考 :func:`oneflow.amax`
    """
)

reset_docstr(
    oneflow.Tensor.amin,
    """
    参考 :func:`oneflow.amin`
    """
)

reset_docstr(
    oneflow.Tensor.asinh,
    """
    参考 :func:`oneflow.asinh`
    """
)

reset_docstr(
    oneflow.Tensor.byte,
    """
    self.byte() 和 self.to(oneflow.uint8) 是等价的。
    参考 :func:`oneflow.Tensor.to`
    """
)

reset_docstr(
    oneflow.Tensor.cosh,
    """
    参考 :func:`oneflow.cosh`
    """
)



reset_docstr(
    oneflow.Tensor.global_to_global,
    """
    Tensor.global_to_global(placement=None, sbp=None, *, grad_sbp=None, check_meta=False) -> Tensor

    执行张量 placement 和/或 sbp 转换。

    Note:
        这个张量必须是 global tensor。

        至少需要 placement 和 sbp 中的一个操作。

        如果 placement 和 sbp 都与这个张量的 placement 和 sbp 相同，那么返回自己的这个张量。
    
    参数：
        - **placement** (flow.placement, optional) - 返回 global tensor 的所需 placement，默认值：None。
        - **sbp** (flow.sbp.sbp or tuple of flow.sbp.sbp, optional) - 返回 global tensor 的所需 sbp，默认值：None。
    关键参数：
        - **grad_sbp** (flow.sbp.sbp or tuple of flow.sbp.sbp, optional) - 手动指定在后向传播中该张量梯度张量的 sbp。
            如果为 None，将自动推断出梯度张量的 sbp。默认值：None。
        - **check_meta** (bool, optional) - 注明是否要检查元信息。如果设置为 True，则检查每个 rank 上的输入元信息
           （placement 和 sbp）的一致性。默认值：False。

    .. code-block:: python

        >>> # 在 2 个 rank 上各自运行。
        >>> import oneflow as flow
        >>> input = flow.tensor([0., 1.], dtype=flow.float32, placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.broadcast]) # doctest: +SKIP
        >>> output = input.global_to_global(placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.split(0)]) # doctest: +SKIP
        >>> print(output.size()) # doctest: +SKIP
        >>> print(output) # doctest: +SKIP

    .. code-block:: python

        >>> # 在 rank 0 上运行的结果。
        oneflow.Size([2])
        tensor([0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(axis=0),), dtype=oneflow.float32)

    .. code-block:: python

        >>> # 在 rank 1 上运行的结果。
        oneflow.Size([2])
        tensor([0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(axis=0),), dtype=oneflow.float32)
    """
)


reset_docstr(
    oneflow.Tensor.half,
    """
    self.half() 和 self.to(dtype=oneflow.float16) 是等价的。

    参考 :func:`oneflow.Tensor.to`

    """
)

reset_docstr(
    oneflow.Tensor.local_to_global,
    """
    Tensor.local_to_global(placement=None, sbp=None, *, check_meta=Ture) -> Tensor

    从一个 local tensor 构造一个 global tensor。

    Note:
        这个张量必须是 local tensor。

        placement 和 sbp 属性都需要。

        返回的 global tensor 将该张量作为其在当前 rank 中的 local tensor。

        通常数据是不会被广播的，但是当 sbp 为 ``oneflow.sbp.broadcast`` 时，rank 0 的数据将被广播到其他等级。
    
    参数：
        - **placement** (flow.placement, optional) - 返回 global tensor 的所需 placement，默认值：None。
        - **sbp** (flow.sbp.sbp or tuple of flow.sbp.sbp, optional) - 返回 global tensor 的所需 sbp，默认值：None。
    关键参数：
        - **check_meta** (bool, optional) - 注明当从 local tensor 构建 global tensor 时是否要检查元信息。只有设置为 False 时，每
            个 rank 上输入的 local tensor的形状和类型都相同，如果设置为 False，则加速执行 local_to_global。默认值：True。

    .. code-block:: python

        >>> # 在 2 个 rank 上各自运行。
        >>> import oneflow as flow
        >>> input = flow.tensor([0., 1.], dtype=flow.float32) # doctest: +SKIP
        >>> output = input.local_to_global(placement=flow.placement("cpu", ranks=[0, 1]), sbp=[flow.sbp.split(0)], check_meta=False) # doctest: +SKIP
        >>> print(output.size()) # doctest: +SKIP
        >>> print(output) # doctest: +SKIP

    .. code-block:: python

        >>> # 在 rank 0 上运行的结果。
        oneflow.Size([4])
        tensor([0., 1., 0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(axis=0),), dtype=oneflow.float32) 
 
    .. code-block:: python

        >>> # 在 rank 1 上运行的结果。
        oneflow.Size([4])
        tensor([0., 1., 0., 1.], placement=oneflow.placement(type="cpu", ranks=[0, 1]), sbp=(oneflow.sbp.split(axis=0),), dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.Tensor.log,
    """
    参考 :func:`oneflow.log`
    """
)

reset_docstr(
    oneflow.Tensor.new_empty,
    """
    Tensor.new_empty(*size, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    返回一个充满未初始化数据大小为 :attr:`size` 的张量。默认情况下，返回的张量具有与此张量相同的 :attr:`flow.dtype` 和 :attr:`flow.device`。

    参数：
        - **size** (int...) - 一个 list，tuple 或整数的 flow.Size，定义输出张量的形状。
        - **dtype** (flow.dtype, optional) - 返回张量的所需类型，默认值如果为 None，则返回与此张量相同的 flow.dtype。
        - **device** (flow.device, optional) - 返回张量的所需设备，默认值如果为 None，则返回与此张量相同的 flow.device。
        - **placement** (flow.placement, optional) - 返回的 global tensor 的所需的 placement，默认值如果为 None，则返回的张量是使用参数 `device` 的 local tensor。
        - **sbp** (flow.sbp.sbp or tuple of flow.sbp.sbp, optional) -返回的 global tensor 所需的 sbp 描述，默认值如果为 None，则返回的张量是使用参数 `device` 的 local tensor。
        - **requires_grad** (bool, optional) - 如果 autograd 应该在返回的张量上记录算子，默认值：None。

    示例：


    .. code-block:: python

        >>> import oneflow as flow
        >>> x = flow.ones(())
        >>> y = x.new_empty((2, 2))
        >>> y.shape
        oneflow.Size([2, 2])
    """
)

reset_docstr(
    oneflow.Tensor.new_zeros,
    """
    Tensor.new_zeros(size=None, dtype=None, device=None, placement=None, sbp=None, requires_grad=False) -> Tensor

    返回一个大小为 0 的张量。默认情况下，返回的张量与该张量具有相同的 flow.dtype 和 flow.device。

    参数：
        - **size** (int...) - 一个 list，tuple 或整数的 flow.Size，定义输出张量的形状。
        - **dtype** (flow.dtype, optional) - 返回张量的所需类型，默认值如果为 None，则返回与此张量相同的 flow.dtype。
        - **device** (flow.device, optional) - 返回张量的所需设备，默认值如果为 None，则返回与此张量相同的 flow.device。
        - **placement** (flow.placement, optional) - 返回的 global tensor 的所需的 placement，默认值如果为 None，则返回的张量是使用参数 `device` 的 local tensor。
        - **sbp** (flow.sbp.sbp or tuple of flow.sbp.sbp, optional) -返回的 global tensor 所需的 sbp 描述，默认值如果为 None，则返回的张量是使用参数 `device` 的 local tensor。
        - **requires_grad** (bool, optional) - 如果 autograd 应该在返回的张量上记录算子，默认值：None。

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> x = flow.Tensor(np.ones((1, 2, 3)))
        >>> y = x.new_zeros((2, 2))
        >>> y
        tensor([[0., 0.],
                [0., 0.]], dtype=oneflow.float32)
    """
)

reset_docstr(
    oneflow.Tensor.sin_,
    r"""
    :func:`oneflow.sin` 的 In-place 版本。

    """
)

reset_docstr(
    oneflow.Tensor.sqrt,
    """
    参考 :func:`oneflow.sqrt`
    """
)

reset_docstr(
    oneflow.Tensor.square,
    """
    参考 :func:`oneflow.square`
    """
)

reset_docstr(
    oneflow.Tensor.swapdims,
    """
    参考 :func:`oneflow.swapdims`
    """
)

reset_docstr(
    oneflow.Tensor.to_consistent,
    """
    这个接口将不再有效，请使用 :func:`oneflow.Tensor.to_global`。
    """
)

reset_docstr(
    oneflow.Tensor.unbind,
    """
    参考 :func:`oneflow.unbind`
    """
)

reset_docstr(
    oneflow.Tensor.view_as,
    """
    Tensor.view_as(other) -> Tensor

    将这个张量扩大到与 :attr:`other` 相同的大小。
    ``self.view_as(other)`` 与 ``self.view(other.size())`` 是等价的。
    
    查阅 :meth:`~Tensor.view` 来获得更多关于 ``view`` 的信息。

    参数：
        - **other** (:class:`oneflow.Tensor`) - 结果张量与 :attr:`other` 的大小相同。
    """
)

reset_docstr(
    oneflow.from_numpy,
    r"""
    从一个 :attr:`numpy.ndarray` 创建一个 :attr:`Tensor`
    
    返回的 tensor 和 ndarray 共享相同的内存。对 tensor 的修改将反映在 ndarray 中，反之亦然。

    它目前所接受 ndarray 的数据类型为 numpy.float64、numpy.float32、numpy.float16、numpy.int64、numpy.int32、numpy.int8、numpy.uint8。

    例如:
        >>> import numpy as np
        >>> np_arr = np.arange(6).reshape(2, 3)
        >>> t = flow.from_numpy(np_arr)
        >>> t
        tensor([[0, 1, 2],
                [3, 4, 5]], dtype=oneflow.int64)
        >>> np_arr[0, 0] = -1
        >>> t
        tensor([[-1,  1,  2],
                [ 3,  4,  5]], dtype=oneflow.int64)
    """
)
