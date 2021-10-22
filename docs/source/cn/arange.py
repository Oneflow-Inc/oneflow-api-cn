import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.arange,
    r"""arange(start=0, end, step=1, *, dtype=kInt64, device=None, requires_grad=Flase) -> Tensor

    返回一个大小为 :math:`\left\lfloor \frac{\text{end} - \text{start}}{\text{step}} \right\rfloor + 1`
    ， 其元素为从 :attr:`start` （包括）到 :attr:`end` （不包括）跨度为 :attr:`step` 的所有整数。

    公式为：

    .. math::
        \text{out}_{i+1} = \text{out}_i + \text{step}.

    参数：
        - **start** (int): 返回的集合的起始值。默认为 0。
        - **end** (int): 返回的集合的结束值。
        - **step** (int): 相邻点的跨度。默认为 1。

    关键词参数：
        - **dtype** (flow.dtype, 可选): 如果未设定 `dtype` ，则  `dtype` 为 `flow.int64`.
        - **device** (flow.device, 可选): 返回张量的所需设备。当前设备作为默认张量。
        - **requires_grad** (bool, 可选): 如果 autograd 为 `True` 则记录对返回张量的操作。默认为 `False`.

    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> y = flow.arange(0, 5)
        >>> y
        tensor([0, 1, 2, 3, 4], dtype=oneflow.int64)
    
    """
)
