import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.as_tensor,
    r"""
    as_tensor(data, dtype=None, device=None) -> Tensor

    界面与 PyTorch 一致。

    如果可能，将数据转换为张量，共享数据并保留 autograd 历史记录

    如果 data 已经是具有请求的 dtype 和 device 的张量，则返回数据本身，但如果 data 是具有不同 dtype 或 device 的张量，则将其复制为使用 data.to(dtype=dtype, device=device)。

    如果 data 是具有相同 dtype  和 device  的 NumPy 数组（一个 ndarray），则使用 oneflow.from_numpy 构造一个张量。

    参数:
        - **data** (array_like) - 张量的初始数据。可以是 list, tuple, NumPy,``ndarray``, scalar和其他类型。
        - **dtype** (oneflow.dtype, optional) - 返回张量的所需数据类型。默认值： if ``None``，从数据推断数据类型。
        - **device** (oneflow.device, optional) - 构造张量的设备。如果``None`` 并且 :attr:`data` 是张量，则使用 :attr:`data` 的设备。如果 None 并且 :attr:`data` 不是张量，则在 CPU 上构建结果张量。

    示例:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> a = np.array([1, 2, 3])
        >>> t = flow.as_tensor(a, device=flow.device('cuda'))
        >>> t
        tensor([1, 2, 3], device='cuda:0', dtype=oneflow.int64)
        >>> t[0] = -1
        >>> a
        array([1, 2, 3])
    """
)