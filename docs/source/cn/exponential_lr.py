import oneflow
from docreset import reset_docstr

reset_docstr(
    oneflow.optim.lr_scheduler.ExponentialLR,
    """
    在每个 epoch 中对每个参数组的学习率进行伽玛衰减操作。
    当 last_step = -1 时，设置初始学习率为 lr。

    参数：
        - **optimizer** (Optimizer) - 被包装的优化器。
        - **gamma** (float) - 学习率衰减的乘法系数。
        - **last_step** (int) - 最后一个 epoch 的索引，默认值：-1。
        - **verbose** (bool) - 如果为 ``True`` ，则会在每次更新时打印一条信息到标准输出。默认为 ``False``。
    """
)
