import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.tensor,
    r"""    
    用数据构造一个张量，如果 placement 和 sbp 在 kwargs 中，则返回一致张量，
        否则返回一个局部张量。
       
    参数：
        - **data**: 张量的初始数据。可以是列表、元组、NumPy ndarray、标量或张量。

    关键词参数：
        - **dtype** (oneflow.dtype, 可选)：返回张量的所需数据类型。默认值：如果没有，则从数据推断数据类型。
        - **device** (oneflow.device, 可选)：返回张量的所需设备。如果placement 和sbp 为None，则使用当前cpu。
        - **placement** (oneflow.placement, 可选)：返回张量的理想位置。
        - **sbp** (oneflow.sbp or tuple of oneflow.sbp, 可选)：返回张量的所需 sbp。
        - **requires_grad** (bool, 可选)：如果已经自动求导则记录对返回张量的操作。默认值：False。

    Noted:
        关键词参数与placement 和sbp 是互斥的。
        一致张量只能由张量构造。


    示例：

    .. code-block:: python

        >>> import oneflow as flow
        
        >>> x = flow.tensor([1,2,3])
        >>> x
        tensor([1, 2, 3], dtype=oneflow.int64)

    """,
)
